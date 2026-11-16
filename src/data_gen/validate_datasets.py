"""Step 6 — final validation of the evaluation datasets: does the spending mix still match reality?

Two checks:
 1. Realism of the NORMAL company: baseline (no planted items) spending mix within ±3 pp of the real
    FOCUS sample in every group.
 2. Anomalies are not unrealistically large: waste from planted ANOMALIES ≤ 29% of total spend —
    the average self-reported waste on IaaS/PaaS in Flexera's 2026 State of the Cloud survey.
The mix of each anomaly dataset is reported too; it is EXPECTED to shift (an idle GPU adds GPU spend).
A first version required every dataset within ±3 pp; 7 of 15 failed only because of planted GPU items,
which is the anomaly itself — so check 1 is applied to the baseline and check 2 bounds the anomalies.

Inputs : data/raw/focus_sample_100000.csv.gz, data/synthetic/<difficulty>/r<n>/billing.parquet
Outputs: reports/figures/step6_spending_mix.png, printed table (max deviation per dataset)

Run:  uv run python src/data_gen/validate_datasets.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from calibrate_world import bucket_of_sku, real_targets

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
FIG = ROOT / "reports/figures"
MAX_DEV_PP = 3.0     # baseline: every group within 3 percentage points of the real share (design choice)
MAX_WASTE_SHARE = 0.29  # Flexera 2026 State of the Cloud: self-reported wasted IaaS/PaaS spend
LABELS = ROOT / "data/labels/ground_truth.csv"  # allowed here: data_gen is not agent code


def mix(state_file: Path) -> pd.Series:
    s = pd.read_parquet(state_file, columns=["sku", "quantity", "unit_price"])
    cost = (s.quantity * s.unit_price).groupby(s.sku.map(bucket_of_sku)).sum()
    return cost / cost.sum()


def main() -> None:
    target = real_targets()
    rows = {"real FOCUS sample": target, "baseline (no items)": mix(SYN / "resource_days.parquet")}
    for f in sorted(SYN.glob("*/r*/resource_days.parquet")):
        rows[f"{f.parent.parent.name}/{f.parent.name}"] = mix(f)
    df = (pd.DataFrame(rows).T[target.index] * 100).round(1)
    dev = (df - df.loc["real FOCUS sample"]).abs().max(axis=1)
    df["max_dev_pp"] = dev.round(1)
    gt = pd.read_csv(LABELS)
    waste = {}
    for f in sorted(SYN.glob("*/r*/billing.parquet")):
        d, r = f.parent.parent.name, int(f.parent.name[1:])
        x = gt[(gt.difficulty == d) & (gt.replica == r) & (gt.label == "anomaly")]
        waste[f"{d}/r{r}"] = x.impact_usd.sum() / pd.read_parquet(f, columns=["EffectiveCost"]).EffectiveCost.sum()
    df["anomaly_waste_%"] = (pd.Series(waste) * 100).round(1)
    pd.set_option("display.width", 220)
    print(df.to_string())
    ok1 = df.loc["baseline (no items)", "max_dev_pp"] <= MAX_DEV_PP
    ok2 = max(waste.values()) <= MAX_WASTE_SHARE
    print(f"\n[{'PASS' if ok1 else 'FAIL'}] baseline mix within ±{MAX_DEV_PP} pp of real "
          f"(max {df.loc['baseline (no items)', 'max_dev_pp']} pp)")
    print(f"[{'PASS' if ok2 else 'FAIL'}] anomaly waste ≤ {MAX_WASTE_SHARE:.0%} of spend in every dataset "
          f"(max {max(waste.values()):.1%}, min {min(waste.values()):.1%})")
    ok = ok1 and ok2

    FIG.mkdir(parents=True, exist_ok=True)
    show = df.loc[["real FOCUS sample", "baseline (no items)", "easy/r1", "medium/r1", "hard/r1"], target.index]
    ax = show.T.plot.bar(figsize=(12, 4.5), width=0.8)
    ax.set_ylabel("% of total spend")
    ax.set_title("Step 6 — spending mix: real FOCUS sample vs DemoShop datasets")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(FIG / "step6_spending_mix.png", dpi=110)
    print(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} · chart: reports/figures/step6_spending_mix.png")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
