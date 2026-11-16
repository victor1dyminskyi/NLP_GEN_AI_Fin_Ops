"""Step 1 — build the price catalog from the real FOCUS sample.

Every price in the synthetic data must come from a real billing row.
For each catalog item we search the real AWS rows by a ChargeDescription
pattern, take the most frequent matching line, and copy its real price and units.
If a pattern matches nothing, the script fails (no silent invented prices).

Input : data/raw/focus_sample_100000.csv.gz  (FinOps Foundation, CC BY 4.0)
Output: data/config/catalog.csv

Run:  uv run python src/data_gen/build_catalog.py
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/focus_sample_100000.csv.gz"
OUT = ROOT / "data/config/catalog.csv"

# sku_key -> (ChargeDescription LIKE pattern, kind)
# kind decides how the generator treats the item:
#   hourly   : billed per hour a resource runs (servers, DBs, gateways)
#   storage  : billed per GB-month stored (daily cost = GB * price / days_in_month)
#   transfer : billed per GB moved or ingested (traffic-driven, noisy)
ITEMS = {
    # --- compute (EC2), real on-demand Linux prices ---
    "ec2_t3_medium":   ("$0.0416 per On Demand Linux t3.medium Instance Hour", "hourly"),
    "ec2_m5_large":    ("$0.096 per On Demand Linux m5.large Instance Hour", "hourly"),
    "ec2_m5_xlarge":   ("$0.192 per On Demand Linux m5.xlarge Instance Hour", "hourly"),
    "ec2_m5_2xlarge":  ("$0.384 per On Demand Linux m5.2xlarge Instance Hour", "hourly"),
    "ec2_c5_2xlarge":  ("$0.34 per On Demand Linux c5.2xlarge Instance Hour", "hourly"),
    "ec2_c5_4xlarge":  ("$0.68 per On Demand Linux c5.4xlarge Instance Hour", "hourly"),
    "ec2_r5_xlarge":   ("$0.252 per On Demand Linux r5.xlarge Instance Hour", "hourly"),
    # --- GPU (EC2) ---
    "ec2_g5_4xlarge":  ("$1.624 per On Demand Linux g5.4xlarge Instance Hour", "hourly"),
    "ec2_g5_12xlarge": ("$5.672 per On Demand Linux g5.12xlarge Instance Hour", "hourly"),
    # --- databases (RDS) ---
    "rds_m5_large_multiaz_mysql": ("%RDS db.m5.large Multi-AZ instance hour%MySQL", "hourly"),
    "rds_m5_large_pg":            ("%RDS db.m5.large Single-AZ instance hour%PostgreSQL", "hourly"),
    # --- storage ---
    "ebs_gp3":       ("%per GB-month of General Purpose (gp3) provisioned storage - US East%", "storage"),
    "ebs_snapshot":  ("%per GB-Month of snapshot data stored%", "storage"),
    "s3_standard":   ("$0.023 per GB - first 50 TB / month of storage used", "storage"),
    # --- networking ---
    "alb_hours":        ("$0.0225 per Application LoadBalancer-hour (or partial hour)", "hourly"),
    "nat_hours":        ("$0.045 per NAT Gateway Hour", "hourly"),
    "nat_gb":           ("$0.045 per GB Data Processed by NAT Gateways", "transfer"),
    "dt_out_internet":  ("$0.090 per GB - first 10 TB / month data transfer out beyond the global free tier", "transfer"),
    "dt_inter_az":      ("$0.010 per GB - regional data transfer - in/out/between EC2 AZs or using elastic IPs or ELB", "transfer"),
    "dt_inter_region":  ("%per GB - US West (Oregon) data transfer to US East (Northern Virginia)", "transfer"),
    # --- observability ---
    "cw_logs_ingest":   ("%per GB custom log data ingested in Standard log class - US East%", "transfer"),
}

GPU_KEYS = {"ec2_g5_4xlarge", "ec2_g5_12xlarge"}


def main() -> None:
    con = duckdb.connect()
    con.execute(
        f"""
        create table aws as
        select * from read_csv_auto('{RAW}', all_varchar=true, sample_size=-1)
        where ProviderName = 'AWS' and ChargeCategory = 'Usage'
          and try_cast(ListUnitPrice as double) > 0
        """
    )
    rows = []
    for key, (pattern, kind) in ITEMS.items():
        hit = con.execute(
            """
            select ServiceName, ServiceCategory, ChargeDescription,
                   try_cast(ListUnitPrice as double) as price,
                   PricingUnit, ConsumedUnit, any_value(ResourceType) as rtype,
                   any_value(SkuId) as sku, count(*) as n
            from aws where ChargeDescription like ?
            group by all order by n desc limit 1
            """,
            [pattern],
        ).fetchone()
        if hit is None:
            raise SystemExit(f"No real row matches {key!r} ({pattern!r}) — refusing to invent a price.")
        svc, cat, desc, price, punit, cunit, rtype, sku, n = hit
        rows.append((key, kind, key in GPU_KEYS, svc, cat, desc, price, punit, cunit, rtype, sku, n))

    con.execute(
        """create table catalog(sku_key text, kind text, is_gpu bool, ServiceName text,
           ServiceCategory text, ChargeDescription text, ListUnitPrice double,
           PricingUnit text, ConsumedUnit text, ResourceType text, SkuId text, rows_in_sample int)"""
    )
    con.executemany("insert into catalog values (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"copy catalog to '{OUT}' (header, delimiter ',')")
    print(con.sql("select sku_key, kind, ListUnitPrice, PricingUnit, rows_in_sample from catalog"))
    print(f"Wrote {len(rows)} items -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
