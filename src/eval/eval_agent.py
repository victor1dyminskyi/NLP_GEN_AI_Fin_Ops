"""Evaluate Component 2 + 3 (agent + action gate) from saved runs, against the answer key.

Each incident (one per detector alert) is matched to an answer-key row with the detector's rule
(same resource, start within [row start − 3 days, row end]). Then:

  anomaly alerts : type correct? owner correct? action correct? approval tier correct?
  decoy alerts   : dismissed? (type planned_change / no_issue AND action no_action)
  unmatched      : (detector false alarms) dismissed?
  end-to-end     : anomaly fully handled = caught by detector AND type + owner + action correct
  safety         : risky actions executed without approval (must be 0 — the gate is code)

Run:  uv run python src/eval/eval_agent.py reports/runs/medium_r1 [--tag v1]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agent"))
from agent import NEEDS_APPROVAL  # noqa: E402

LABELS = ROOT / "data/labels/ground_truth.csv"
TOL = pd.Timedelta(days=3)
DISMISS_TYPES = {"planned_change", "no_issue"}


def match_label(resource, start, labels):
    for _, lb in labels.iterrows():
        if (resource in lb.resources.split(";")
                and pd.Timestamp(lb.start_date) - TOL <= pd.Timestamp(start) <= pd.Timestamp(lb.end_date)):
            return lb
    return None


def evaluate(run_dir: Path) -> dict:
    diff, rep = run_dir.name.split("_")
    gt = pd.read_csv(LABELS)
    labels = gt[(gt.difficulty == diff) & (gt.replica == int(rep[1:]))]
    rows = []
    for f in sorted(run_dir.glob("AL-*.json")):
        inc = json.loads(f.read_text())
        form, g = inc["form"], inc["gate"]
        lb = match_label(inc["resource"], inc["alert"]["start_date"], labels)
        kind = "unmatched" if lb is None else lb.label
        r = {"alert": f.stem, "resource": inc["resource"], "kind": kind, "case": None if lb is None else lb.case_id,
             "true_type": None if lb is None else lb.type, "pred_type": form["type"],
             "true_owner": None if lb is None else lb.owner_team, "pred_owner": form["owner_team"],
             "true_action": None if lb is None else lb.expected_action, "pred_action": form["action"],
             "confidence": form["confidence"], "status": g["status"], "seconds": inc["seconds"]}
        dismissed = form["type"] in DISMISS_TYPES and form["action"] == "no_action"
        if kind == "anomaly":
            r.update(type_ok=form["type"] == lb.type, owner_ok=form["owner_team"] == lb.owner_team,
                     action_ok=form["action"] == lb.expected_action,
                     tier_ok=(form["action"] in NEEDS_APPROVAL) == (lb.action_tier == "approval"))
            r["all_ok"] = r["type_ok"] and r["owner_ok"] and r["action_ok"]
        else:
            r["dismissed_ok"] = dismissed
        rows.append(r)
    df = pd.DataFrame(rows)
    an = df[df.kind == "anomaly"]
    handled = set(an[an.all_ok].case) if len(an) else set()
    n_anom = int((labels.label == "anomaly").sum())
    risky_unapproved = int(((df.pred_action.isin(NEEDS_APPROVAL)) & (df.status == "auto_executed (simulated)")).sum())
    summary = {
        "alerts": len(df),
        "anomaly_alerts": len(an),
        "type_acc_%": round(100 * an.type_ok.mean(), 0) if len(an) else None,
        "owner_acc_%": round(100 * an.owner_ok.mean(), 0) if len(an) else None,
        "action_acc_%": round(100 * an.action_ok.mean(), 0) if len(an) else None,
        "tier_acc_%": round(100 * an.tier_ok.mean(), 0) if len(an) else None,
        "decoys_dismissed": f"{int(df[df.kind == 'decoy'].dismissed_ok.sum())}/{int((df.kind == 'decoy').sum())}",
        "false_alarms_dismissed": f"{int(df[df.kind == 'unmatched'].dismissed_ok.sum())}/{int((df.kind == 'unmatched').sum())}",
        "end_to_end_%": round(100 * len(handled) / n_anom, 0) if n_anom else None,
        "escalated": int((df.status == "escalated_to_human").sum()),
        "risky_actions_without_approval": risky_unapproved,
        "median_seconds": float(df.seconds.median()),
    }
    return {"summary": summary, "rows": df}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    out = {}
    for d in a.run_dirs:
        res = evaluate(ROOT / d)
        print(f"\n=== {d} {a.tag}")
        cols = ["alert", "resource", "kind", "true_type", "pred_type", "true_owner", "pred_owner",
                "pred_action", "confidence", "status"]
        print(res["rows"][cols].to_string(index=False))
        print(json.dumps(res["summary"], indent=1))
        out[d] = {"summary": res["summary"], "rows": res["rows"].to_dict("records")}
    path = ROOT / "reports/eval" / f"agent_dev{('_' + a.tag) if a.tag else ''}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
