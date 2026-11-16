"""FINAL TEST — run once on the test replicas (3–5) with everything frozen.

1. Audit: prove nothing was tuned on test data (detector tuning, SQL benchmark and agent runs used dev only).
2. Fingerprint: SHA-256 of all agent code + configs + model file → anyone can check the frozen version.
3. Detector on all test datasets (frozen settings) → catch rate, false alarms.
4. Agent (frozen v3, seed 42) on every alert of every test dataset → type/owner/action/end-to-end.
5. Save reports/eval/final_test.json and all incidents in reports/runs/final/.

Run:  uv run python -u src/eval/final_test.py
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agent"))
sys.path.insert(0, str(ROOT / "src/eval"))
from agent import MODEL, investigate  # noqa: E402
from detector import detect, load_params  # noqa: E402
from eval_agent import evaluate  # noqa: E402
from eval_detector import score  # noqa: E402

SC = yaml.safe_load((ROOT / "data/config/scenarios.yaml").read_text())
DEV, TEST = SC["dev_replicas"], SC["test_replicas"]
GT = pd.read_csv(ROOT / "data/labels/ground_truth.csv")
OUT = ROOT / "reports/runs/final"
FROZEN = ["src/agent/agent.py", "src/agent/tools.py", "src/agent/detector.py", "src/agent/llm.py",
          "data/config/detector.yaml", "data/config/playbooks.md", "data/config/data_dictionary.md",
          "ollama/Modelfile"]


def audit() -> list[str]:
    msgs = []
    det = json.loads((ROOT / "reports/eval/detector_dev.json").read_text())
    used = {int(r["replica"]) for r in det["per_dataset"]}
    assert used <= set(DEV), f"detector was tuned on non-dev replicas: {used}"
    msgs.append(f"detector tuned on replicas {sorted(used)} (dev) ✓")
    bench = yaml.safe_load((ROOT / "data/config/sql_benchmark.yaml").read_text())["dataset"]
    assert int(bench.split("/r")[-1]) in DEV, bench
    msgs.append(f"model benchmark on {bench} (dev) ✓")
    for tag in ["v1", "v2", "v3"]:
        f = ROOT / f"reports/eval/agent_dev_{tag}.json"
        if f.exists():
            reps = {int(Path(k).name.split("_r")[-1]) for k in json.loads(f.read_text())}
            assert reps <= set(DEV), f"agent {tag} evaluated on {reps}"
            msgs.append(f"agent {tag} developed on replicas {sorted(reps)} (dev) ✓")
    assert not OUT.exists() or not any(OUT.iterdir()), "final test already run — it runs ONCE"
    return msgs


def fingerprint() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16] for f in FROZEN}


def main() -> None:
    log = audit()
    print("AUDIT\n  " + "\n  ".join(log), flush=True)
    params = load_params()
    result = {"run_at": datetime.now().isoformat(timespec="seconds"), "model": MODEL, "detector_params": params,
              "fingerprint": fingerprint(), "audit": log, "detector": [], "agent": []}

    for d in ["easy", "medium", "hard"]:
        for r in TEST:
            ds = ROOT / "data/synthetic" / d / f"r{r}"
            labels = GT[(GT.difficulty == d) & (GT.replica == r)]
            alerts = detect(ds, params)
            s = score(alerts, labels)
            result["detector"].append({"difficulty": d, "replica": r,
                                       **{k: v for k, v in s.items() if not k.startswith("_")}})
            out = OUT / f"{d}_r{r}"
            out.mkdir(parents=True, exist_ok=True)
            print(f"\n{d}/r{r}: {len(alerts)} alerts, detector caught {s['caught']}/{s['anomalies']}", flush=True)
            for _, al in alerts.iterrows():
                inc = investigate(al.to_dict(), ds)
                (out / f"{al.alert_id}.json").write_text(json.dumps(inc, indent=2, default=str))
                print(f"  {al.alert_id} {al.resource:24} → {inc['form']['type']:18} [{inc['gate']['status']}]",
                      flush=True)
            ev = evaluate(out)
            result["agent"].append({"difficulty": d, "replica": r, **ev["summary"],
                                    "rows": ev["rows"].to_dict("records")})

    det = pd.DataFrame(result["detector"]).groupby("difficulty")[["anomalies", "caught", "alerts", "false_alarms"]].sum()
    det["catch_rate_%"] = (100 * det.caught / det.anomalies).round(0)
    det["precision_%"] = (100 * (det.alerts - det.false_alarms) / det.alerts).round(0)
    ag = []
    for d in ["easy", "medium", "hard"]:
        rs = [x for a in result["agent"] if a["difficulty"] == d for x in a["rows"]]
        df = pd.DataFrame(rs)
        an = df[df.kind == "anomaly"]
        n_anom = int((GT[(GT.difficulty == d) & GT.replica.isin(TEST)].label == "anomaly").sum())
        handled = an[an.all_ok].case.nunique()
        ag.append({"difficulty": d, "anomaly_alerts": len(an),
                   "type_acc_%": round(100 * an.type_ok.mean()), "owner_acc_%": round(100 * an.owner_ok.mean()),
                   "action_acc_%": round(100 * an.action_ok.mean()), "tier_acc_%": round(100 * an.tier_ok.mean()),
                   "decoys_dismissed": f"{int(df[df.kind == 'decoy'].dismissed_ok.sum())}/{int((df.kind == 'decoy').sum())}",
                   "false_alarms_dismissed": f"{int(df[df.kind == 'unmatched'].dismissed_ok.sum())}/{int((df.kind == 'unmatched').sum())}",
                   "end_to_end_%": round(100 * handled / n_anom), "escalated": int((df.status == "escalated_to_human").sum()),
                   "risky_without_approval": sum(a["risky_actions_without_approval"] for a in result["agent"] if a["difficulty"] == d)})
    result["detector_by_difficulty"] = det.reset_index().to_dict("records")
    result["agent_by_difficulty"] = ag
    (ROOT / "reports/eval/final_test.json").write_text(json.dumps(result, indent=2, default=str))
    print("\nDETECTOR (test)\n", det.to_string())
    print("\nAGENT (test)\n", pd.DataFrame(ag).to_string(index=False))
    print("\nSaved reports/eval/final_test.json")


if __name__ == "__main__":
    main()
