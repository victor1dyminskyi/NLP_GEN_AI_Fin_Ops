"""Re-run saved investigations and check you get the SAME result (same model file, seed 42, temperature 0).

Run:  uv run python src/eval/verify_reproducibility.py [reports/runs/final/medium_r3] [--n 5]
Prints, per alert, whether the new run's incident form and every step are identical to the saved one.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agent"))
from agent import investigate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="reports/runs/final/medium_r3")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    run = ROOT / a.run_dir
    diff, rep = run.name.split("_")
    ds = ROOT / "data/synthetic" / diff / rep
    same = 0
    files = sorted(run.glob("AL-*.json"))[: a.n]
    for f in files:
        saved = json.loads(f.read_text())
        alert = {**saved["alert"]}
        alert.pop("resource_kind", None)
        new = investigate(alert, ds)
        ok_form = new["form"] == saved["form"]
        ok_steps = [e["raw"] for e in new["trace"]] == [e["raw"] for e in saved["trace"]]
        same += ok_form and ok_steps
        print(f"{f.stem} {saved['resource']:24} form identical: {ok_form} · every step identical: {ok_steps}")
    print(f"\n{same}/{len(files)} investigations reproduced exactly.")


if __name__ == "__main__":
    main()
