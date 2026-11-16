"""Step 4 — side tables, all DERIVED from the shared state (resource_days.parquet) so they stay in sync.

  usage_metrics : one row per EC2 server-fleet per day WITH running hours > 0
                  cpu_pct  = fleet level × daily noise; fleet level ~ log-normal fitted to REAL Azure
                             per-VM averages (p25/p50/p75 by workload class)
                  gpu_pct  = gpu_inference_median × daily noise (GPU servers only; level is a TESTED setting)
  change_log    : CloudTrail-style events — create (before window or on active_from), delete (active_to),
                  weekly scheduler stop/start for office_hours servers
  owners        : team directory (team → owner, email, Slack, accounts). No resource→owner map on purpose:
                  finding the owner is a task the agent is evaluated on.
  commitments   : Savings Plans (id, discount, term, covered resources)

Inputs : data/synthetic/resource_days.parquet, data/synthetic/billing.parquet,
         data/config/world_calibrated.yaml, data/raw/azure_vm_2019/cpu_utilization_summary.csv
Outputs: data/synthetic/{usage_metrics,change_log,owners,commitments}.parquet, reports/figures/step4_*.png

Run:  uv run python src/data_gen/generate_side_tables.py   (after generate_billing.py)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from generate_billing import resource_rng

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
FIG = ROOT / "reports/figures"
CPU_SUMMARY = ROOT / "data/raw/azure_vm_2019/cpu_utilization_summary.csv"
AZURE_CLASS = {"interactive": "Interactive", "batch": "Delay-insensitive"}

# CloudTrail event names per resource type: (create, delete, eventSource)
EVENTS = {
    "ec2_": ("RunInstances", "TerminateInstances", "ec2.amazonaws.com"),
    "rds_": ("CreateDBInstance", "DeleteDBInstance", "rds.amazonaws.com"),
    "ebs_gp3": ("CreateVolume", "DeleteVolume", "ec2.amazonaws.com"),
    "ebs_snapshot": ("CreateSnapshot", "DeleteSnapshot", "ec2.amazonaws.com"),
    "s3_": ("CreateBucket", "DeleteBucket", "s3.amazonaws.com"),
    "nat_hours": ("CreateNatGateway", "DeleteNatGateway", "ec2.amazonaws.com"),
    "alb_hours": ("CreateLoadBalancer", "DeleteLoadBalancer", "elasticloadbalancing.amazonaws.com"),
}


def event_names(sku: str):
    for prefix, ev in EVENTS.items():
        if sku.startswith(prefix):
            return ev
    return None  # traffic / log lines are usage, not resources → no create/delete events


def username(full_name: str) -> str:
    return full_name.lower().replace(" ", ".")


def window_level(dates: pd.Series, base: float, windows: list, key: str) -> np.ndarray:
    """Load level per day: `base`, overridden inside util_windows that set `key` (e.g. idle GPU ≈ 0.5%)."""
    level = np.full(len(dates), base, dtype=float)
    for w in windows or []:
        if key in w:
            inside = (dates >= pd.Timestamp(w["from"])) & (dates <= pd.Timestamp(w["to"]))
            level[inside.to_numpy()] = w[key]
    return level


def usage_metrics(state, world, cat) -> pd.DataFrame:
    cpu = pd.read_csv(CPU_SUMMARY).set_index("category")
    u = world["utilization"]
    daily_noise = u["daily_noise"] * world.get("noise_scale", 1.0)
    by_id = {r["id"]: r for r in world["resources"]}
    servers = state[state.sku.str.startswith("ec2_") & (state.running_hours > 0)]
    frames = []
    for name, s in servers.groupby("ResourceName", sort=False):
        r = by_id[name]
        rng = resource_rng(world, name, stream=4)
        q = cpu.loc[AZURE_CLASS[r.get("workload", "interactive")]]
        sigma = np.log(q.avg_cpu_p75 / q.avg_cpu_p25) / 1.349  # IQR of a normal = 1.349 σ
        level = float(np.exp(rng.normal(np.log(q.avg_cpu_p50), sigma)))
        n = len(s)
        cpu_level = window_level(s.date, r.get("cpu_level", level), r.get("util_windows"), "cpu_pct")
        cpu_pct = np.clip(cpu_level * rng.normal(1, daily_noise, n), 0.1, 100)
        gpu_pct = None
        if bool(cat.loc[r["sku"]].is_gpu):
            gpu_level = window_level(s.date, r.get("gpu_level", u["gpu_inference_median"]),
                                     r.get("util_windows"), "gpu_pct")
            gpu_pct = np.clip(gpu_level * rng.normal(1, daily_noise, n), 0.1, 100)
        frames.append(pd.DataFrame({
            "date": s.date.to_numpy(), "ResourceId": s.ResourceId.to_numpy(), "ResourceName": name,
            "SubAccountName": s.SubAccountName.to_numpy(),
            "instances_avg": np.round(s.running_hours.to_numpy() / 24, 2),
            "cpu_pct": np.round(cpu_pct, 1),
            "gpu_pct": np.round(gpu_pct, 1) if gpu_pct is not None else np.nan,
        }))
    return pd.concat(frames, ignore_index=True)


def change_log(state, world) -> pd.DataFrame:
    cl = world["change_log"]
    start = pd.Timestamp(world["period"]["start"])
    owners = {t: username(v["owner"]) for t, v in world["teams"].items()}
    rows = []
    for r in world["resources"]:
        ev = event_names(r["sku"])
        if ev is None:
            continue
        create, delete, source = ev
        acct = world["accounts"][r["account"]]
        s = state[state.ResourceName == r["id"]]
        rid = s.ResourceId.iloc[0]
        rng = resource_rng(world, r["id"], stream=5)
        default_actor = owners[acct["team"]] if r.get("tags", True) else cl["legacy_actor"]
        actor = r.get("created_by", default_actor)
        ticket = r.get("ticket")
        base = dict(ResourceName=r["id"], ResourceId=rid, SubAccountName=r["account"], event_source=source)

        if "active_from" in r:
            t = pd.Timestamp(r["active_from"]) + pd.Timedelta(hours=int(rng.integers(9, 12)))
        else:
            t = start - pd.Timedelta(minutes=int(rng.integers(60, cl["history_days_before_start"] * 1440)))
        rows.append({**base, "event_time": t, "event_name": create, "actor": actor, "actor_type": "user",
                     "ticket": ticket, "details": None})
        if "attached_to" in r:  # a disk attached to a server
            rows.append({**base, "event_time": t + pd.Timedelta(minutes=5), "event_name": "AttachVolume",
                         "actor": actor, "actor_type": "user", "ticket": ticket,
                         "details": f"attached to {r['attached_to']}"})

        if "active_to" in r:
            t = pd.Timestamp(r["active_to"]) + pd.Timedelta(hours=18)
            rows.append({**base, "event_time": t, "event_name": delete, "actor": r.get("deleted_by", actor),
                         "actor_type": "user", "ticket": ticket, "details": None})

        if r["profile"] == "office_hours" and r["sku"].startswith("ec2_"):
            alive = s[s.quantity > 0].date
            for d in alive[alive.dt.day_of_week == 5]:  # Saturday 00:00 stop
                rows.append({**base, "event_time": d, "event_name": "StopInstances",
                             "actor": cl["scheduler_actor"], "actor_type": "automation", "ticket": None,
                             "details": "scheduled weekend stop"})
            for d in alive[alive.dt.day_of_week == 0]:  # Monday 07:00 start
                rows.append({**base, "event_time": d + pd.Timedelta(hours=7), "event_name": "StartInstances",
                             "actor": cl["scheduler_actor"], "actor_type": "automation", "ticket": None,
                             "details": "scheduled weekday start"})

    # extra events (Step 5): e.g. an approved change ticket for a planned traffic increase
    rid_of = state.drop_duplicates("ResourceName").set_index("ResourceName").ResourceId
    acct_of = state.drop_duplicates("ResourceName").set_index("ResourceName").SubAccountName
    for e in world.get("extra_events", []):
        rows.append({"event_time": pd.Timestamp(e["time"]), "event_name": e["event_name"],
                     "event_source": e.get("event_source", "change-management"), "actor": e["actor"],
                     "actor_type": "user", "SubAccountName": acct_of[e["resource"]],
                     "ResourceName": e["resource"], "ResourceId": rid_of[e["resource"]],
                     "ticket": e.get("ticket"), "details": e.get("details")})
    df = pd.DataFrame(rows).sort_values("event_time", ignore_index=True)
    return df[["event_time", "event_name", "event_source", "actor", "actor_type",
               "SubAccountName", "ResourceName", "ResourceId", "ticket", "details"]]


def build_side_tables(state, world, cat) -> dict:
    return {"usage_metrics": usage_metrics(state, world, cat), "change_log": change_log(state, world),
            "owners": owners_table(world), "commitments": commitments_table(world)}


def owners_table(world) -> pd.DataFrame:
    accts = pd.Series({k: v["team"] for k, v in world["accounts"].items()})
    return pd.DataFrame([{
        "team": t, "owner": v["owner"], "username": username(v["owner"]), "email": v["email"],
        "slack": v["slack"], "accounts": ", ".join(accts[accts == t].index),
    } for t, v in world["teams"].items()])


def commitments_table(world) -> pd.DataFrame:
    return pd.DataFrame([{
        "commitment_id": c["id"], "type": c["type"], "discount": c["discount"],
        "start": pd.Timestamp(c["start"]), "end": pd.Timestamp(c["end"]) if c.get("end") else pd.NaT,
        "covers": ", ".join(c["covers"]), "managed_by": c.get("managed_by"),
    } for c in world["commitments"]])


def checks(state, bill, met, cl, own, com, world) -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    servers = state[state.sku.str.startswith("ec2_") & (state.running_hours > 0)]
    key = ["ResourceName", "date"]
    check("metrics exist exactly on billed server-days",
          set(map(tuple, met[key].values)) == set(map(tuple, servers[key].values)))
    check("no metrics on days with 0 running hours", not met.merge(state[state.quantity == 0], on=key).shape[0])
    check("cpu/gpu within 0–100 %", met.cpu_pct.between(0, 100).all() and met.gpu_pct.dropna().between(0, 100).all())
    gpu_servers = set(state[state.sku.str.startswith("ec2_g")].ResourceName)
    check("GPU metrics exactly on GPU servers", set(met[met.gpu_pct.notna()].ResourceName) == gpu_servers)
    check("billing: cost = quantity × unit price",
          np.allclose(bill.EffectiveCost, bill.PricingQuantity * bill.ContractedUnitPrice, atol=1e-5))
    check("billing: no negative quantities or costs", (bill.PricingQuantity >= 0).all() and (bill.EffectiveCost >= 0).all())
    n_days = state.date.nunique()
    always_on = [r["id"] for r in world["resources"] if "active_from" not in r and "active_to" not in r]
    always_on = [x for x in always_on if bill[bill.ResourceName == x].shape[0]]
    check(f"always-on resources have all {n_days} days in billing",
          (bill[bill.ResourceName.isin(always_on)].groupby("ResourceName").size() == n_days).all())
    creates = cl[cl.event_name.isin([e[0] for e in EVENTS.values()])]
    check("every change-log resource exists in state", cl.ResourceName.isin(state.ResourceName).all())
    check("one create event per (non-traffic) resource", creates.ResourceName.value_counts().max() == 1)
    first_bill = bill.groupby("ResourceName").ChargePeriodStart.min()
    check("create event happens before the first bill row",
          (creates.set_index("ResourceName").event_time <= first_bill.reindex(creates.ResourceName).values
           + pd.Timedelta(days=1)).all())
    check("every team has an owner", own.owner.notna().all() and len(own) == len(world["teams"]))
    cov = com.assign(r=com.covers.str.split(", ")).explode("r")
    check("commitments cover existing resources", cov.r.isin(state.ResourceName).all())
    b = bill.merge(cov[["r", "start", "end"]], left_on="ResourceName", right_on="r", how="inner")
    inside = (b.ChargePeriodStart >= b.start) & (b.end.isna() | (b.ChargePeriodStart < b.end))
    check("'Committed' price exactly on days inside the Savings Plan term",
          ((b.PricingCategory == "Committed") == inside).all())
    return ok


def charts(met, cl):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for name in ["sf-web-asg", "data-etl-workers", "ml-notebooks", "ml-inference"]:
        s = met[met.ResourceName == name].set_index("date").cpu_pct.iloc[:56]
        ax[0].plot(s.index, s.values, marker=".", lw=1, label=name)
    g = met[met.ResourceName == "ml-inference"].set_index("date").gpu_pct.iloc[:56]
    ax[0].plot(g.index, g.values, color="black", lw=1.6, label="ml-inference GPU %")
    ax[0].set_title("Daily average load, first 8 weeks")
    ax[0].set_ylabel("% busy")
    ax[0].set_ylim(0, None)
    ax[0].legend(fontsize=8)
    ax[0].tick_params(axis="x", rotation=30)
    counts = cl.event_name.value_counts()
    ax[1].barh(counts.index, counts.values)
    ax[1].set_title("Change-log events by type")
    fig.tight_layout()
    fig.savefig(FIG / "step4_side_tables.png", dpi=110)


def main() -> None:
    world = yaml.safe_load((ROOT / "data/config/world_calibrated.yaml").read_text())
    cat = pd.read_csv(ROOT / "data/config/catalog.csv").set_index("sku_key")
    state = pd.read_parquet(SYN / "resource_days.parquet")
    bill = pd.read_parquet(SYN / "billing.parquet")
    t = build_side_tables(state, world, cat)
    met, cl, own, com = t["usage_metrics"], t["change_log"], t["owners"], t["commitments"]
    for name, df in t.items():
        df.to_parquet(SYN / f"{name}.parquet", index=False)
        print(f"{name:14} {len(df):>6,} rows")

    print("\nMedian daily CPU % by server fleet:")
    print(met.groupby("ResourceName").cpu_pct.median().round(1).sort_values().to_string())
    print(f"ml-inference median GPU %: {met.gpu_pct.median():.1f}")

    print("\nSanity checks:")
    all_ok = checks(state, bill, met, cl, own, com, world)
    charts(met, cl)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} · chart: reports/figures/step4_side_tables.png")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
