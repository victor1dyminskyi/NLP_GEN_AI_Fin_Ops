"""Step 3 — build the shared state and FOCUS billing for DemoShop. One row = one resource × one day.

  quantity(day) = base_qty × count × weekday_factor × (1 + growth_per_month)^months × noise × window_factor
  cost(day)     = quantity(day) × unit price            (price from catalog.csv = real FOCUS rows)

Rules
- noise: multiplicative, normal(1, sigma × noise_scale); one random stream PER RESOURCE (seeded by
  seed + resource id) → adding a resource never changes the numbers of the others.
- hourly resources cannot run more than 24 h × count per day, unless the resource sets its own
  `noise` in world.yaml (an auto-scaling group can add instances).
- storage is billed per GB-month: daily quantity = GB stored / days in that month.
- Lifecycle: optional active_from / active_to per resource → no usage (and no bill rows) outside it.
- qty_windows (Step 5): [{from, to, factor}] multiplies usage in a window;
  [{from, to, ramp_to}] ramps linearly from ×1 to ×ramp_to and holds it after `to`.
- Savings Plan: covered rows get a discounted unit price only inside the plan's [start, end).
  Simplification: BilledCost = EffectiveCost = discounted cost (no separate commitment-fee rows).

Inputs : data/config/world_calibrated.yaml, data/config/catalog.csv
Output : data/synthetic/resource_days.parquet  (shared state: one row per resource per day)
         data/synthetic/billing.parquet        (derived: FOCUS rows where quantity > 0)
         reports/figures/step3_*.png

Sync rule: every table (billing, usage metrics, change log) is DERIVED from resource_days.
To switch something off, change the world (active_from / active_to, commitment end) and regenerate —
never patch costs directly.

Run:  uv run python src/data_gen/generate_billing.py
"""

import calendar
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
FIG = ROOT / "reports/figures"

REGION_NAMES = {"us-east-1": "US East (N. Virginia)", "us-west-2": "US West (Oregon)"}


def stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def resource_rng(world: dict, rid: str, stream: int = 0) -> np.random.Generator:
    """Independent, reproducible random stream per resource."""
    return np.random.default_rng([world["period"]["seed"], stream, stable_int(rid)])


def resource_id(r: dict, account_id: str) -> str:
    """Realistic-looking, deterministic AWS resource id."""
    h = hashlib.md5(r["id"].encode()).hexdigest()
    sku = r["sku"]
    if sku.startswith("ec2_"):
        return f"i-0{h[:16]}"
    if sku.startswith("rds_"):
        return f"arn:aws:rds:us-east-1:{account_id}:db:{r['id']}"
    if sku.startswith("ebs_gp3"):
        return f"vol-0{h[:16]}"
    if sku.startswith("ebs_snapshot"):
        return f"snap-0{h[:16]}"
    if sku.startswith("s3_"):
        return f"demoshop-{r['id']}"
    if sku.startswith("nat_"):
        return f"nat-0{h[:16]}"
    if sku.startswith("alb_"):
        return f"arn:aws:elasticloadbalancing:us-east-1:{account_id}:loadbalancer/app/{r['id']}/{h[:16]}"
    return f"{r['id']}-{h[:8]}"


def window_factor(days: pd.DatetimeIndex, windows: list) -> np.ndarray:
    f = np.ones(len(days))
    for w in windows or []:
        a, b = pd.Timestamp(w["from"]), pd.Timestamp(w["to"])
        inside = (days >= a) & (days <= b)
        if "factor" in w:
            f[inside] *= w["factor"]
        if "ramp_to" in w:
            span = max((b - a).days, 1)
            prog = np.clip((days - a).days / span, 0, 1)
            ramp = 1 + (w["ramp_to"] - 1) * prog
            f[days >= a] *= ramp[days >= a]  # holds ramp_to after `to`
    return f


def build_state(world: dict, cat: pd.DataFrame) -> pd.DataFrame:
    p = world["period"]
    days = pd.date_range(p["start"], p["end"], freq="D")
    months_elapsed = (days - days[0]).days / 30.44
    dow = days.day_of_week.to_numpy()  # Monday = 0
    days_in_month = np.array([calendar.monthrange(d.year, d.month)[1] for d in days])
    noise_scale = world.get("noise_scale", 1.0)
    sp_of = {r: c for c in world["commitments"] for r in c["covers"]}
    states = []

    for r in world["resources"]:
        item = cat.loc[r["sku"]]
        acct = world["accounts"][r["account"]]
        count = r.get("count", 1)
        sigma = r.get("noise", world["noise_defaults"][item.kind]) * noise_scale
        rng = resource_rng(world, r["id"])

        weekday = np.array(world["weekday_profiles"][r["profile"]])[dow]
        growth = (1 + r.get("growth_per_month", 0.0)) ** months_elapsed
        noise = np.clip(rng.normal(1.0, sigma, len(days)), 0.05, None)
        base = r["base_qty"] * count * growth * noise * window_factor(days, r.get("qty_windows"))

        if item.kind == "hourly":
            qty = base * weekday
            if "noise" not in r:  # fixed-size servers: physical cap of 24 h per instance
                qty = np.minimum(qty, 24 * count)
        elif item.kind == "storage":
            qty = base / days_in_month  # GB stored -> GB-months for one day
        else:  # transfer
            qty = base * weekday

        # lifecycle: a resource only exists between active_from and active_to (inclusive)
        alive = np.ones(len(days), bool)
        if "active_from" in r:
            alive &= days >= pd.Timestamp(r["active_from"])
        if "active_to" in r:
            alive &= days <= pd.Timestamp(r["active_to"])
        qty = np.where(alive, np.round(qty, 6), 0.0)

        # Savings Plan applies only on days inside its term [start, end)
        list_price = float(item.ListUnitPrice)
        sp = sp_of.get(r["id"])
        covered = np.zeros(len(days), bool)
        if sp:
            covered = days >= pd.Timestamp(sp["start"])
            if sp.get("end"):
                covered &= days < pd.Timestamp(sp["end"])
        unit_price = np.where(covered, list_price * (1 - (sp or {}).get("discount", 0)), list_price)

        states.append(pd.DataFrame({
            "date": days, "ResourceName": r["id"], "ResourceId": resource_id(r, acct["id"]),
            "sku": r["sku"], "kind": item.kind, "SubAccountName": r["account"], "team": acct["team"],
            "count": count, "quantity": qty,
            "running_hours": qty if item.kind == "hourly" else np.nan,
            "list_unit_price": list_price, "unit_price": unit_price,
            "pricing_category": np.where(covered, "Committed", "Standard"),
            "commitment_id": np.where(covered, sp["id"] if sp else None, None),
        }))
    return pd.concat(states, ignore_index=True)


def build_billing(state: pd.DataFrame, world: dict, cat: pd.DataFrame) -> pd.DataFrame:
    """FOCUS rows = state rows where something was used."""
    company = world["company"]
    by_id = {r["id"]: r for r in world["resources"]}
    sp_by_id = {c["id"]: c for c in world["commitments"]}
    frames = []
    for name, s in state[state.quantity > 0].groupby("ResourceName", sort=False):
        r = by_id[name]
        item = cat.loc[r["sku"]]
        acct = world["accounts"][r["account"]]
        d = pd.DatetimeIndex(s.date)
        qty = s.quantity.to_numpy()
        list_price = s.list_unit_price.to_numpy()
        unit_price = s.unit_price.to_numpy()
        committed = s.pricing_category.to_numpy() == "Committed"
        sp_type = np.array([sp_by_id[c]["type"] if isinstance(c, str) else None for c in s.commitment_id], dtype=object)
        region = "us-west-2" if r["sku"] == "dt_inter_region" else "us-east-1"
        tags = None
        if r.get("tags", True):
            tags = json.dumps({"team": acct["team"], "application": r.get("app", ""), "environment": acct["env"]})

        frames.append(pd.DataFrame({
            "BillingAccountId": company["billing_account_id"],
            "BillingAccountName": company["name"],
            "BillingCurrency": company["currency"],
            "BillingPeriodStart": d.to_period("M").start_time,
            "BillingPeriodEnd": (d.to_period("M") + 1).start_time,
            "ChargePeriodStart": d,
            "ChargePeriodEnd": d + pd.Timedelta(days=1),
            "ChargeCategory": "Usage",
            "ChargeFrequency": "Usage-Based",
            "ChargeDescription": item.ChargeDescription,
            "ProviderName": "AWS",
            "PublisherName": "Amazon Web Services, Inc.",
            "InvoiceIssuerName": "Amazon Web Services, Inc.",
            "RegionId": region,
            "RegionName": REGION_NAMES[region],
            "ServiceName": item.ServiceName,
            "ServiceCategory": item.ServiceCategory,
            "SkuId": item.SkuId,
            "ResourceId": s.ResourceId.to_numpy(),
            "ResourceName": name,
            "ResourceType": item.ResourceType if isinstance(item.ResourceType, str) else None,
            "SubAccountId": acct["id"],
            "SubAccountName": r["account"],
            "ConsumedQuantity": qty,
            "ConsumedUnit": item.ConsumedUnit,
            "PricingQuantity": qty,
            "PricingUnit": item.PricingUnit,
            "ListUnitPrice": list_price,
            "ListCost": np.round(qty * list_price, 6),
            "ContractedUnitPrice": unit_price,
            "ContractedCost": np.round(qty * unit_price, 6),
            "BilledCost": np.round(qty * unit_price, 6),
            "EffectiveCost": np.round(qty * unit_price, 6),
            "PricingCategory": s.pricing_category.to_numpy(),
            "CommitmentDiscountId": s.commitment_id.to_numpy(),
            "CommitmentDiscountType": sp_type,
            "CommitmentDiscountStatus": np.where(committed, "Used", None),
            "Tags": tags,
        }))
    return pd.concat(frames, ignore_index=True)


def charts(df: pd.DataFrame, world: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    team = df.SubAccountName.map({k: v["team"] for k, v in world["accounts"].items()})
    daily = df.assign(team=team).pivot_table(index="ChargePeriodStart", columns="team",
                                             values="EffectiveCost", aggfunc="sum")
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    daily.sum(axis=1).plot(ax=ax[0], color="black", lw=1.2)
    ax[0].set_title("DemoShop — total daily cost (normal history, no anomalies)")
    ax[0].set_ylabel("$ / day")
    ax[0].set_ylim(bottom=0)
    daily.plot.area(ax=ax[1], lw=0, alpha=0.85)
    ax[1].set_title("Daily cost by team")
    ax[1].set_ylabel("$ / day")
    ax[1].legend(loc="upper left", ncol=6, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "step3_daily_cost.png", dpi=110)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for a, rid in zip(axs.flat, ["sf-web-asg", "sf-egress", "ml-notebooks", "data-lake"]):
        s = df[df.ResourceName == rid].set_index("ChargePeriodStart").EffectiveCost.iloc[:56]
        s.plot(ax=a, marker=".", lw=1)
        a.set_title(rid)
        a.set_ylabel("$ / day")
        a.set_ylim(bottom=0)
    fig.suptitle("Example resources, first 8 weeks (weekday pattern + noise)")
    fig.tight_layout()
    fig.savefig(FIG / "step3_examples.png", dpi=110)


def main() -> None:
    world = yaml.safe_load((ROOT / "data/config/world_calibrated.yaml").read_text())
    cat = pd.read_csv(ROOT / "data/config/catalog.csv").set_index("sku_key")
    state = build_state(world, cat)
    df = build_billing(state, world, cat)
    SYN.mkdir(parents=True, exist_ok=True)
    state.to_parquet(SYN / "resource_days.parquet", index=False)
    df.to_parquet(SYN / "billing.parquet", index=False)

    month = df.ChargePeriodStart.dt.to_period("M")
    team = df.SubAccountName.map({k: v["team"] for k, v in world["accounts"].items()})
    print(f"Rows: {len(df):,} · resources: {df.ResourceName.nunique()} · days: {state.date.nunique()}")
    print("\nMonthly bill ($, EffectiveCost):")
    print(df.assign(month=month, team=team).pivot_table(index="month", columns="team", values="EffectiveCost",
                                                       aggfunc="sum", margins=True).round(0).to_string())
    charts(df, world)
    print("\nWrote data/synthetic/{resource_days,billing}.parquet and reports/figures/step3_*.png")


if __name__ == "__main__":
    main()
