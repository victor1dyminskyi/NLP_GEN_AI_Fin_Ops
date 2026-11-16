"""Step 4 research — measure real CPU utilization of cloud servers from the Azure Public Dataset V2.

Source: Microsoft Azure VM trace 2019 (AzurePublicDataset, Cortez et al., SOSP'17 "Resource Central").
Table vmtable: one row per VM with avg / max / p95-max CPU utilization (%), lifetime, workload category.

We keep VMs that lived ≥ 7 days (always-on servers, like DemoShop's) and summarise average CPU %
per workload category:
  Interactive        ≈ web / API servers (user-facing)
  Delay-insensitive  ≈ batch / ETL workers
  Unknown            ≈ everything else

Input : data/raw/azure_vm_2019/vmtable.csv.gz   (downloaded if missing, ~440 MB, not kept in git)
Output: data/raw/azure_vm_2019/cpu_utilization_summary.csv   (small, used by Step 4)

Run:  uv run python src/data_gen/measure_cpu_utilization.py
"""

import urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data/raw/azure_vm_2019"
RAW = DIR / "vmtable.csv.gz"
OUT = DIR / "cpu_utilization_summary.csv"
URL = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/trace_data_vmtable_vmtable.csv.gz"
COLS = ("{'vmid':'VARCHAR','subscriptionid':'VARCHAR','deploymentid':'VARCHAR','vmcreated':'BIGINT',"
        "'vmdeleted':'BIGINT','maxcpu':'DOUBLE','avgcpu':'DOUBLE','p95maxcpu':'DOUBLE',"
        "'vmcategory':'VARCHAR','vmcorecountbucket':'VARCHAR','vmmemorybucket':'VARCHAR'}")
MIN_LIFETIME_S = 7 * 24 * 3600


def main() -> None:
    if not RAW.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} …")
        urllib.request.urlretrieve(URL, RAW)

    q = f"""
        with vm as (
            select vmcategory, avgcpu, p95maxcpu
            from read_csv('{RAW}', header=false, columns={COLS})
            where vmdeleted - vmcreated >= {MIN_LIFETIME_S}
        )
        select vmcategory as category, count(*) as vms,
               round(quantile_cont(avgcpu, 0.25), 1) as avg_cpu_p25,
               round(median(avgcpu), 1)              as avg_cpu_p50,
               round(quantile_cont(avgcpu, 0.75), 1) as avg_cpu_p75,
               round(median(p95maxcpu), 1)           as p95max_cpu_p50
        from vm group by all
        union all
        select 'ALL', count(*), round(quantile_cont(avgcpu,0.25),1), round(median(avgcpu),1),
               round(quantile_cont(avgcpu,0.75),1), round(median(p95maxcpu),1) from vm
        order by vms desc
    """
    res = duckdb.sql(q)
    print(res)
    res.write_csv(str(OUT))
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
