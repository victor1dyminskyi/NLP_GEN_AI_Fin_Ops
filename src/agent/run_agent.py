"""Run the full system on one dataset: detector → agent on every alert → incident records.

Output: reports/runs/<difficulty>_<replica>/<alert_id>.json (full trace) + incidents.csv (one row per alert)

Run:  uv run python -u src/agent/run_agent.py data/synthetic/medium/r1 [--limit 3] [--alerts AL-004,AL-010]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from agent import MODEL, investigate
from detector import detect

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--alerts")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    ds = ROOT / a.dataset
    out = ROOT / "reports/runs" / f"{ds.parent.name}_{ds.name}"
    out.mkdir(parents=True, exist_ok=True)
    alerts = detect(ds)
    if a.alerts:
        alerts = alerts[alerts.alert_id.isin(a.alerts.split(","))]
    if a.limit:
        alerts = alerts.head(a.limit)
    print(f"{len(alerts)} alerts on {a.dataset} · model {a.model}", flush=True)

    rows = []
    for _, al in alerts.iterrows():
        inc = investigate(al.to_dict(), ds, model=a.model)
        (out / f"{al.alert_id}.json").write_text(json.dumps(inc, indent=2, default=str))
        f, g = inc["form"], inc["gate"]
        rows.append({"alert_id": al.alert_id, "resource": al.resource, "rule": al.rule,
                     "start_date": str(al.start_date)[:10], "type": f["type"], "owner_team": f["owner_team"],
                     "action": f["action"], "confidence": f["confidence"], "status": g["status"],
                     "tool_calls": inc["tool_calls"], "seconds": inc["seconds"]})
        print(f"  {al.alert_id} {al.resource:24} → {f['type']:18} {str(f['owner_team']):11} {f['action']:28} "
              f"conf={f['confidence']} [{g['status']}] {inc['tool_calls']} tools, {inc['seconds']}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "incidents.csv", index=False)
    print(f"Saved {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
