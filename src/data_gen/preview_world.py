"""Step 2 helper — show the expected monthly bill of world.yaml before generating anything.

No randomness, no growth: just base_qty × count × avg weekday factor × price × 30 days.
Use it to sanity-check the fictional company while editing world.yaml.

Run:  uv run python src/data_gen/preview_world.py
"""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    world = yaml.safe_load((ROOT / "data/config/world.yaml").read_text())
    cat = pd.read_csv(ROOT / "data/config/catalog.csv").set_index("sku_key")
    covered = {r for c in world["commitments"] for r in c["covers"]}
    disc = {r: c["discount"] for c in world["commitments"] for r in c["covers"]}

    rows = []
    for r in world["resources"]:
        item = cat.loc[r["sku"]]
        wk = sum(world["weekday_profiles"][r["profile"]]) / 7
        qty_day = r["base_qty"] * r.get("count", 1) * wk
        if item.kind == "storage":
            qty_day = r["base_qty"] / 30  # GB-month spread over the month
        price = item.ListUnitPrice * (1 - disc.get(r["id"], 0))
        team = world["accounts"][r["account"]]["team"]
        rows.append({
            "team": team, "resource": r["id"], "sku": r["sku"],
            "monthly_$": round(qty_day * price * 30, 0),
            "savings_plan": r["id"] in covered, "tagged": r.get("tags", True),
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(df.sort_values("monthly_$", ascending=False).to_string(index=False))
    print("\nMonthly baseline by team ($):")
    print(df.groupby("team")["monthly_$"].sum().sort_values(ascending=False).to_string())
    print(f"\nTOTAL ≈ ${df['monthly_$'].sum():,.0f}/month · {len(df)} resources")


if __name__ == "__main__":
    main()
