"""Evaluate + tune Component 1 (Detector) against the answer key.

Matching: an alert matches an answer-key row if its resource is one of the row's resources AND its
start date lies in [row start − 3 days, row end].
  caught anomaly  = anomaly row matched by ≥ 1 alert        → catch rate (recall)
  flagged decoy   = decoy row matched (EXPECTED: decoys are real cost increases; the agent must dismiss them)
  false alarm     = alert matching no row (anomaly or decoy)
  alert precision = alerts matching any row / all alerts ;  F1 = 2·P·R / (P + R)

Tuning (dev replicas 1–2 only): grid over z × k × min_usd, pick the best mean F1 (tie → fewer alerts),
write it to data/config/detector.yaml. Test replicas 3–5 are NOT touched here (final evaluation only).

Run:  uv run python src/eval/eval_detector.py            # tune on dev + report dev results
"""

import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agent"))
from detector import detect, load_params  # noqa: E402

LABELS = ROOT / "data/labels/ground_truth.csv"
SYN = ROOT / "data/synthetic"
OUT = ROOT / "reports/eval"
TOL = pd.Timedelta(days=3)
GRID = {"z": [3, 4, 5, 6], "k": [1, 2, 3], "min_usd": [0.1, 0.25, 0.5, 1.0, 5.0]}


def match(alerts: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    labels["caught"] = False
    alerts = alerts.copy()
    alerts["matched_case"] = None
    for i, a in alerts.iterrows():
        for j, lb in labels.iterrows():
            if (a.resource in lb.resources.split(";")
                    and pd.Timestamp(lb.start_date) - TOL <= a.start_date <= pd.Timestamp(lb.end_date)):
                labels.at[j, "caught"] = True
                alerts.at[i, "matched_case"] = lb.case_id
    return alerts, labels


def score(alerts, labels) -> dict:
    alerts, labels = match(alerts, labels)
    an = labels[labels.label == "anomaly"]
    de = labels[labels.label == "decoy"]
    tp_alerts = alerts.matched_case.notna().sum()
    recall = an.caught.mean() if len(an) else 1.0
    precision = tp_alerts / len(alerts) if len(alerts) else 1.0
    return {"anomalies": len(an), "caught": int(an.caught.sum()), "catch_rate": recall,
            "alerts": len(alerts), "false_alarms": int(alerts.matched_case.isna().sum()),
            "precision": precision, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "decoys_flagged": f"{int(de.caught.sum())}/{len(de)}",
            "_labels": labels, "_alerts": alerts}


def datasets(replicas):
    for d in ["easy", "medium", "hard"]:
        for r in replicas:
            yield d, r, SYN / d / f"r{r}"


def main() -> None:
    gt = pd.read_csv(LABELS)
    sc = yaml.safe_load((ROOT / "data/config/scenarios.yaml").read_text())
    dev = sc["dev_replicas"]
    base = load_params()

    # ---- tune on dev ----
    grid = []
    for z, k, m in itertools.product(*GRID.values()):
        p = {**base, "z": z, "k": k, "min_usd": m}
        res = [score(detect(path, p), gt[(gt.difficulty == d) & (gt.replica == r)]) for d, r, path in datasets(dev)]
        grid.append({"z": z, "k": k, "min_usd": m, "mean_f1": sum(x["f1"] for x in res) / len(res),
                     "mean_catch": sum(x["catch_rate"] for x in res) / len(res),
                     "alerts": sum(x["alerts"] for x in res)})
    g = pd.DataFrame(grid).sort_values(["mean_f1", "alerts"], ascending=[False, True])
    print("Top 8 settings on DEV (replicas 1–2, all difficulties):")
    print(g.head(8).round(3).to_string(index=False))
    best = g.iloc[0]
    params = {**base, "z": int(best.z), "k": int(best.k), "min_usd": float(best.min_usd)}
    text = (ROOT / "data/config/detector.yaml").read_text()
    for key in ["z", "k", "min_usd"]:
        text = "\n".join(f"{key}: {params[key]}{' ' * max(1, 15 - len(f'{key}: {params[key]}'))}# TUNED on dev r1–2 (eval_detector.py), frozen"
                         if ln.startswith(f"{key}:") else ln for ln in text.splitlines()) + "\n"
    (ROOT / "data/config/detector.yaml").write_text(text)

    # ---- report dev results with the chosen settings ----
    rows, per_type = [], []
    for d, r, path in datasets(dev):
        s = score(detect(path, params), gt[(gt.difficulty == d) & (gt.replica == r)])
        rows.append({"difficulty": d, "replica": r, **{k: v for k, v in s.items() if not k.startswith("_")}})
        per_type.append(s["_labels"][s["_labels"].label == "anomaly"][["difficulty", "type", "caught", "case_id"]])
    res = pd.DataFrame(rows)
    by_diff = res.groupby("difficulty")[["anomalies", "caught", "alerts", "false_alarms"]].sum()
    by_diff["catch_rate_%"] = (100 * by_diff.caught / by_diff.anomalies).round(0)
    by_diff["precision_%"] = (100 * (by_diff.alerts - by_diff.false_alarms) / by_diff.alerts).round(0)
    by_diff = by_diff.loc[["easy", "medium", "hard"]]
    pt = pd.concat(per_type)
    by_type = pt.pivot_table(index="type", columns="difficulty", values="caught", aggfunc="mean")[["easy", "medium", "hard"]]
    missed = pt[~pt.caught]

    print(f"\nChosen: z={params['z']}, k={params['k']}, min_usd=${params['min_usd']}  → written to data/config/detector.yaml")
    print("\nDEV results by difficulty:")
    print(by_diff.to_string())
    print("\nCatch rate by anomaly type (DEV):")
    print((100 * by_type).round(0).to_string())
    print(f"\nMissed anomalies ({len(missed)}):")
    print(missed.to_string(index=False) if len(missed) else "  none")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "detector_dev.json").write_text(json.dumps({
        "params": params, "grid": g.round(4).to_dict("records"),
        "by_difficulty": by_diff.reset_index().to_dict("records"),
        "by_type_catch_rate": (100 * by_type).round(1).reset_index().to_dict("records"),
        "missed": missed.to_dict("records"), "per_dataset": res.round(4).to_dict("records")}, indent=2, default=str))
    print("\nSaved reports/eval/detector_dev.json")


if __name__ == "__main__":
    main()
