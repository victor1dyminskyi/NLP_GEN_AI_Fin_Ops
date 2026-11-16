"""Component 1 — Detector (code, no LLM). Finds suspicious resources in ONE dataset folder.

Reads only agent-visible data: data/synthetic/<difficulty>/r<n>/{billing,change_log}.parquet.
Design (approved, see docs/CRISP_DM.md §4):

 (a) usage   score_t = (x_t − median_28d) / (1.4826 · MAD_28d); alert if score > Z for K days in a row
             AND extra cost ≥ $min per day
 (b) price   price_t = EffectiveCost / PricingQuantity; alert if |price / median_28d − 1| ≥ 5% for K days
 (c) split   usage effect = (q − q_normal)·p_normal ; price effect = q·(p − p_normal)
 (d) new     first billed day after the first 28 days of history AND average daily cost ≥ $min
 (e) orphan  a disk whose attached server was terminated, still billed K days later   [added — see doc]

Only upward changes are alerts (cost going down is a saving, not a problem).
One alert per episode: consecutive alert days on a resource are merged; the start date is the first day.

Usage:  from detector import detect;  alerts = detect(dataset_dir, params)
Run  :  uv run python src/agent/detector.py data/synthetic/medium/r1
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "data/config/detector.yaml"
MAD_TO_SD = 1.4826


def load_params() -> dict:
    return yaml.safe_load(PARAMS.read_text())


def daily_panel(bill: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cost and quantity per resource (columns) per day (rows); 0 on days a resource has no rows."""
    days = pd.date_range(bill.ChargePeriodStart.min(), bill.ChargePeriodStart.max())
    cost = bill.pivot_table(index="ChargePeriodStart", columns="ResourceName", values="EffectiveCost",
                            aggfunc="sum").reindex(days).fillna(0.0)
    qty = bill.pivot_table(index="ChargePeriodStart", columns="ResourceName", values="PricingQuantity",
                           aggfunc="sum").reindex(days).fillna(0.0)
    return cost, qty


def runs(mask: pd.Series, k: int) -> list[pd.Timestamp]:
    """Start dates of runs of ≥ k consecutive True values."""
    starts, n = [], 0
    for day, v in mask.items():
        n = n + 1 if v else 0
        if n == k:
            starts.append(day - pd.Timedelta(days=k - 1))
    return starts


def usage_alerts(cost, qty, first_seen, p) -> list[dict]:
    w, out = p["window_days"], []
    for r in cost.columns:
        x = cost[r]
        hist = x.where(x.index >= first_seen[r])  # history only while the resource exists
        normal = hist.shift(1).rolling(w, min_periods=w).median()
        mad = (hist.shift(1) - normal).abs().rolling(w, min_periods=w).median()
        spread = np.maximum(MAD_TO_SD * mad, np.maximum(0.01 * normal, 0.01))  # floor: avoid ÷0
        score = (x - normal) / spread
        extra = x - normal
        flag = (score > p["z"]) & (extra >= p["min_usd"])
        for s in runs(flag.fillna(False), p["k"]):
            q = qty[r]
            q_norm = q.shift(1).rolling(w, min_periods=w).median()[s]
            price = (x / q.replace(0, np.nan))
            p_norm = price.shift(1).rolling(w, min_periods=w).median()[s]
            out.append({"resource": r, "rule": "usage", "start_date": s, "score": round(float(score[s]), 1),
                        "normal_usd_day": round(float(normal[s]), 2), "current_usd_day": round(float(x[s]), 2),
                        "extra_usd_day": round(float(extra[s]), 2),
                        "usage_effect": round(float((q[s] - q_norm) * p_norm), 2) if pd.notna(p_norm) else None,
                        "price_effect": round(float(q[s] * (price[s] - p_norm)), 2) if pd.notna(p_norm) else None})
    return out


def price_alerts(cost, qty, first_seen, p) -> list[dict]:
    w, out = p["window_days"], []
    for r in cost.columns:
        q = qty[r]
        price = (cost[r] / q.replace(0, np.nan)).where(q.index >= first_seen[r])
        normal = price.shift(1).rolling(w, min_periods=w // 2).median()
        change = price / normal - 1
        flag = (change.abs() >= p["price_change"]) & (change > 0)
        for s in runs(flag.fillna(False), p["k"]):
            q_norm = q.shift(1).rolling(w, min_periods=w // 2).median()[s]
            out.append({"resource": r, "rule": "price", "start_date": s,
                        "score": round(float(change[s]) * 100, 1),  # % price change
                        "normal_usd_day": round(float(q_norm * normal[s]), 2),
                        "current_usd_day": round(float(cost[r][s]), 2),
                        "extra_usd_day": round(float(cost[r][s] - q_norm * normal[s]), 2),
                        "usage_effect": round(float((q[s] - q_norm) * normal[s]), 2),
                        "price_effect": round(float(q[s] * (price[s] - normal[s])), 2)})
    return out


def new_resource_alerts(cost, first_seen, p) -> list[dict]:
    start, out = cost.index[0], []
    for r, d in first_seen.items():
        if d < start + pd.Timedelta(days=p["window_days"]):
            continue  # existed from the start of history → cannot tell if new
        avg = float(cost[r][d:d + pd.Timedelta(days=p["k"] - 1)].mean())
        if avg >= p["min_usd"]:
            out.append({"resource": r, "rule": "new_resource", "start_date": d, "score": None,
                        "normal_usd_day": 0.0, "current_usd_day": round(avg, 2), "extra_usd_day": round(avg, 2),
                        "usage_effect": round(avg, 2), "price_effect": 0.0})
    return out


def orphan_alerts(cost, change_log, p) -> list[dict]:
    att = change_log[change_log.event_name == "AttachVolume"]
    term = change_log[change_log.event_name == "TerminateInstances"].groupby("ResourceName").event_time.max()
    out = []
    for _, a in att.iterrows():
        server = str(a.details).replace("attached to ", "")
        if server not in term.index or a.ResourceName not in cost.columns:
            continue
        gone = term[server].normalize() + pd.Timedelta(days=1)
        after = cost[a.ResourceName][gone:]
        billed = after[after > 0]
        if len(billed) >= p["k"] and billed.mean() >= p["min_usd"]:
            out.append({"resource": a.ResourceName, "rule": "orphan", "start_date": gone, "score": None,
                        "normal_usd_day": 0.0, "current_usd_day": round(float(billed.mean()), 2),
                        "extra_usd_day": round(float(billed.mean()), 2),
                        "usage_effect": None, "price_effect": None, "related": server})
    return out


def merge_episodes(df: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """One alert per episode: same resource, same family (usage+price | new | orphan), within `gap_days`.
    Keeps the earliest start, joins the rule names, keeps numbers from the strongest (largest extra $)."""
    if df.empty:
        return df
    # usage + price alerts describe the same cost change → one family; new / orphan are separate events
    df["family"] = df.rule.map({"usage": "change", "price": "change"}).fillna(df.rule)
    df = df.sort_values(["resource", "family", "start_date"]).reset_index(drop=True)
    gap = df.groupby(["resource", "family"]).start_date.diff().dt.days
    df["episode"] = ((gap.isna()) | (gap > gap_days)).cumsum()
    rows = []
    for _, g in df.groupby("episode"):
        best = g.loc[g.extra_usd_day.fillna(0).idxmax()].to_dict()
        best["start_date"] = g.start_date.min()
        best["rule"] = "+".join(sorted(set(g.rule)))
        rows.append(best)
    return pd.DataFrame(rows).drop(columns=["episode", "family"])


def detect(dataset_dir, params: dict | None = None) -> pd.DataFrame:
    p = params or load_params()
    d = Path(dataset_dir)
    bill = pd.read_parquet(d / "billing.parquet")
    cl = pd.read_parquet(d / "change_log.parquet")
    cost, qty = daily_panel(bill)
    first_seen = bill.groupby("ResourceName").ChargePeriodStart.min()
    alerts = (usage_alerts(cost, qty, first_seen, p) + price_alerts(cost, qty, first_seen, p)
              + new_resource_alerts(cost, first_seen, p)
              + (orphan_alerts(cost, cl, p) if p.get("orphan_rule", True) else []))
    df = merge_episodes(pd.DataFrame(alerts), p.get("merge_days", 14))
    if df.empty:
        return df
    acct = bill.drop_duplicates("ResourceName").set_index("ResourceName").SubAccountName
    df.insert(1, "account", df.resource.map(acct))
    df = df.sort_values(["start_date", "resource"], ignore_index=True)
    df.insert(0, "alert_id", [f"AL-{i:03d}" for i in range(1, len(df) + 1)])
    return df


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/medium/r1"
    pd.set_option("display.width", 220)
    print(detect(ROOT / path).to_string(index=False))
