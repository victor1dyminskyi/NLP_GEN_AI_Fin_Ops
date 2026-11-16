"""Step 2b — calibrate DemoShop's amounts so its SPENDING MIX matches the real FOCUS sample.

Prices are never touched (they are real). Only amounts change:
  amount for a group = target share × total bill ÷ real price

Groups (same rule applied to the real sample and to DemoShop):
  compute_cpu, compute_gpu, databases, network_fixed (NAT/ALB hours), block_storage (EBS, snapshots),
  object_storage (S3), data_transfer (egress, inter-AZ, inter-region, NAT GB), logs.
Real services we do not model (Datadog, SQS, ...) are excluded and the shares renormalised.

How the amounts are solved
- data_transfer, block_storage, object_storage, logs: quantities are continuous → scaled exactly
  to the target share.
- databases, network_fixed: whole machines (1 DB, 1 NAT) → kept as designed; they define the scale.
- compute (CPU and GPU separately): fleets (count > 1) are scaled by one factor k per group (rounded to whole servers);
  single machines (bastion, FTP) stay at 1.
- Total bill T is solved from: fixed groups must take (1 − compute share − continuous shares) of T.
  T itself is a free choice of company size (detection is relative), documented as such.

Inputs : data/raw/focus_sample_100000.csv.gz, data/config/world.yaml, data/config/catalog.csv
Output : data/config/world_calibrated.yaml (used by the generator), printed target-vs-achieved table

Run:  uv run python src/data_gen/calibrate_world.py
"""

from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/focus_sample_100000.csv.gz"
SRC = ROOT / "data/config/world.yaml"
OUT = ROOT / "data/config/world_calibrated.yaml"

COMPUTE = ["compute_cpu", "compute_gpu"]
CONTINUOUS = ["data_transfer", "block_storage", "object_storage", "logs"]
FIXED = ["databases", "network_fixed"]
DAYS = 30.44


def bucket_of_sku(sku: str) -> str:
    if sku in ("ec2_g5_4xlarge", "ec2_g5_12xlarge"):
        return "compute_gpu"
    if sku.startswith("ec2_"):
        return "compute_cpu"
    if sku.startswith("rds_"):
        return "databases"
    if sku in ("nat_hours", "alb_hours"):
        return "network_fixed"
    if sku.startswith("ebs_"):
        return "block_storage"
    if sku.startswith("s3_"):
        return "object_storage"
    if sku.startswith("cw_logs"):
        return "logs"
    return "data_transfer"  # dt_*, nat_gb


def real_targets() -> pd.Series:
    """Share of spend per group in the real AWS sample (modeled groups only)."""
    df = duckdb.sql(f"""
        with t as (select ServiceName s, ChargeDescription d, try_cast(BilledCost as double) cost
                   from read_csv_auto('{RAW}', all_varchar=true, sample_size=-1)
                   where ProviderName='AWS' and ChargeCategory='Usage'),
        b as (select case
            when s='Amazon Elastic Compute Cloud' and d like '%Instance Hour%'
                 and regexp_matches(d, '(g4dn|g5|p3|p4d)\\.') then 'compute_gpu'
            when s='Amazon Elastic Compute Cloud' and d like '%Instance Hour%' then 'compute_cpu'
            when s='Amazon Relational Database Service' and d ilike '%instance hour%' then 'databases'
            when d like '%Data Processed by NAT%' or d ilike '%data transfer%' then 'data_transfer'
            when d like '%NAT Gateway Hour%' or d like '%LoadBalancer-hour%' then 'network_fixed'
            when s='Amazon Elastic Compute Cloud' and d ilike '%GB-month%' then 'block_storage'
            when s='Amazon Simple Storage Service' and d like '%storage used%' then 'object_storage'
            when d like '%log data ingested%' then 'logs'
            else null end bucket, cost from t)
        select bucket, sum(cost) cost from b where bucket is not null group by 1""").df()
    return (df.set_index("bucket").cost / df.cost.sum()).sort_values(ascending=False)


def monthly_cost(r: dict, cat: pd.DataFrame, world: dict, disc: dict) -> float:
    item = cat.loc[r["sku"]]
    price = item.ListUnitPrice * (1 - disc.get(r["id"], 0))
    if item.kind == "storage":
        return r["base_qty"] * price  # GB × $/GB-month
    wk = sum(world["weekday_profiles"][r["profile"]]) / 7
    return r["base_qty"] * r.get("count", 1) * wk * price * DAYS


def main() -> None:
    world = yaml.safe_load(SRC.read_text())
    cat = pd.read_csv(ROOT / "data/config/catalog.csv").set_index("sku_key")
    disc = {r: c["discount"] for c in world["commitments"] for r in c["covers"]}
    target = real_targets()

    res = world["resources"]
    cost = {r["id"]: monthly_cost(r, cat, world, disc) for r in res}
    bucket = {r["id"]: bucket_of_sku(r["sku"]) for r in res}
    by_bucket = pd.Series(cost).groupby(pd.Series(bucket)).sum()

    fixed_cost = by_bucket[FIXED].sum()
    fixed_share = 1 - target[COMPUTE].sum() - target[CONTINUOUS].sum()
    T = fixed_cost / fixed_share  # total monthly bill implied by the fixed machines

    # one fleet factor per compute group (CPU, GPU); single machines stay at 1
    k = {}
    for g in COMPUTE:
        singles = sum(cost[r["id"]] for r in res if bucket[r["id"]] == g and r.get("count", 1) == 1)
        fleets = sum(cost[r["id"]] for r in res if bucket[r["id"]] == g and r.get("count", 1) > 1)
        k[g] = (target[g] * T - singles) / fleets

    for r in res:
        b = bucket[r["id"]]
        if b in COMPUTE and r.get("count", 1) > 1:
            r["count"] = int(max(1, round(r["count"] * k[b])))
        elif b in CONTINUOUS:
            f = target[b] * T / by_bucket[b]
            r["base_qty"] = float(round(r["base_qty"] * f, 2))
        r["bucket"] = b

    world["calibration"] = {
        "source": "data/raw/focus_sample_100000.csv.gz (AWS usage rows, modeled groups only)",
        "target_shares": {b: round(float(v), 4) for b, v in target.items()},
        "implied_total_monthly_usd": round(float(T), 0),
        "compute_fleet_factor_k": {g: round(float(v), 3) for g, v in k.items()},
        "note": "Generated by src/data_gen/calibrate_world.py — edit world.yaml, not this file.",
    }
    OUT.write_text("# GENERATED — do not edit. Source: world.yaml + real spending mix.\n"
                   + yaml.safe_dump(world, sort_keys=False, allow_unicode=True))

    # report achieved vs target
    new_cost = pd.Series({r["id"]: monthly_cost(r, cat, world, disc) for r in res})
    achieved = new_cost.groupby(pd.Series(bucket)).sum() / new_cost.sum()
    before = by_bucket / by_bucket.sum()
    rep = pd.DataFrame({"real_target_%": target * 100, "before_%": before * 100, "after_%": achieved * 100}).round(1)
    print(rep.sort_values("real_target_%", ascending=False).to_string())
    print(f"\nTotal bill: before ${by_bucket.sum():,.0f}/mo → after ${new_cost.sum():,.0f}/mo · fleet factors {', '.join(f'{g}={v:.2f}' for g, v in k.items())}")
    print("\nChanged amounts:")
    orig = {r["id"]: r for r in yaml.safe_load(SRC.read_text())["resources"]}
    for r in res:
        o = orig[r["id"]]
        if o.get("count", 1) != r.get("count", 1) or o["base_qty"] != r["base_qty"]:
            what = (f"count {o.get('count', 1)} → {r['count']}" if o.get("count", 1) != r.get("count", 1)
                    else f"base_qty {o['base_qty']} → {r['base_qty']}")
            print(f"  {r['id']:18} {r['bucket']:18} {what}")
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
