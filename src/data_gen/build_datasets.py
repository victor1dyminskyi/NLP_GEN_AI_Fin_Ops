"""Step 5 — plant anomalies & decoys and build the evaluation datasets.

3 difficulties (easy / medium / hard) × 5 replicas. The number of planted items varies per replica
(totals per type in scenarios.yaml are spread randomly), so nobody can rely on a fixed count.
A replica has the same items, resources and dates in every difficulty; only the TESTED difficulty
settings change (normal GPU load, noise, spike/drift size, size of the untagged server).

How an item is planted — never by editing costs:
  new resource      → added to the world with active_from / active_to (idle GPU, untagged, orphan, decoys)
  usage change      → qty_windows on an existing resource (spike ×factor, drift ramp, campaign)
  price change      → Savings Plan `end` moved into the window (expired plan)
  load change       → util_windows / gpu_level (training busy, then idle)
  context           → change-log fields (created_by, ticket, attached_to, extra_events)
Then the whole dataset is regenerated from the shared state, so all tables stay in sync.

$ impact of an item = cost(world with items) − cost(same world without items), over the item's
resources and window. Exact, because every resource has its own random stream.

Inputs : data/config/world_calibrated.yaml, data/config/scenarios.yaml, data/config/catalog.csv
Outputs: data/synthetic/{easy,medium,hard}/r{1..5}/*.parquet       (what the agent may read)
         data/labels/ground_truth.csv                              (answer key — evaluation only)
         reports/figures/step5_medium_r1.png

Run:  uv run python src/data_gen/build_datasets.py
"""

import contextlib
import copy
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from generate_billing import build_billing, build_state
from generate_side_tables import build_side_tables, checks as base_checks

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
LAB = ROOT / "data/labels"
FIG = ROOT / "reports/figures"
D = pd.Timedelta(days=1)


def username(full_name: str) -> str:
    return full_name.lower().replace(" ", ".")


def pick_dates(rng, sc, start, end, n) -> list:
    lo = start + sc["warmup_days"] * D
    hi = end - sc["min_visible_days"] * D
    cand = list(pd.date_range(lo, hi))
    chosen = []
    for _ in range(n):
        if not cand:  # too crowded → allow closer dates
            cand = [c for c in pd.date_range(lo, hi) if c not in chosen]
        d = cand[int(rng.integers(len(cand)))]
        chosen.append(d)
        cand = [c for c in cand if abs((c - d).days) >= sc["min_gap_days"]]
    return chosen


PREFIX = {"storefront-dev": "sf", "ml-dev": "ml", "data-prod": "data", "analytics-prod": "bi",
          "payments-prod": "pay", "storefront-prod": "sf"}


def allocate(sc: dict) -> dict:
    """Spread the type totals over replicas → {replica: [types]}. Counts per replica vary.
    Guarantees: showcase replica has every type; every type appears in ≥ 2 test replicas."""
    rng = np.random.default_rng([sc["seed"], 0])
    reps = list(range(1, sc["replicas"] + 1))
    show = sc["showcase"]["replica"]
    alloc = {r: [] for r in reps}
    left = dict(sc["totals"])
    for t in left:  # showcase gets every type once
        alloc[show].append(t)
        left[t] -= 1
    cap = sc["max_per_replica"]
    for t in left:  # every type ≥ 2× in the final-test replicas (2 different replicas)
        for r in rng.choice(sc["test_replicas"], size=2, replace=False):
            alloc[int(r)].append(t)
            left[t] -= 1
    for t, n in left.items():
        for _ in range(n):
            ok = [r for r in reps if alloc[r].count(t) < cap.get(t, 99)
                  and len(alloc[r]) < sc["max_items_per_replica"]]
            alloc[int(rng.choice(ok))].append(t)
    return alloc


def plan_replica(r: int, types: list, sc: dict, world: dict) -> list:
    """Difficulty-independent plan for one replica: which items, where, when."""
    rng = np.random.default_rng([sc["seed"], r])
    start, end = pd.Timestamp(world["period"]["start"]), pd.Timestamp(world["period"]["end"])
    types = list(rng.permutation(types))
    dates = pick_dates(rng, sc, start, end, len(types))
    suffix = iter(rng.choice(np.arange(10, 99), size=len(types), replace=False))
    ticket = iter(rng.choice(np.arange(1000, 9999), size=len(types), replace=False))
    pools = {"transfer_spike": list(rng.permutation(sc["spike_resources"])),
             "transfer_drift": list(rng.permutation(sc["drift_resources"]))}
    items = []
    for i, (t, d) in enumerate(zip(types, dates), start=1):
        it = {"key": f"I{i:02d}", "type": t, "date": d}
        if t == "idle_gpu":
            it.update(name=f"ml-train-{next(suffix)}", sku=str(rng.choice(["ec2_g5_12xlarge", "ec2_g5_4xlarge"])))
        elif t in pools:
            it["name"] = pools[t].pop()
        elif t == "untagged_resource":
            acct = str(rng.choice(sc["untagged_accounts"]))
            it.update(account=acct, name=f"{PREFIX[acct]}-perf-test-{next(suffix)}")
        elif t == "sp_expired":
            it["name"] = world["commitments"][0]["id"]
            if r == sc["showcase"]["replica"]:
                it["date"] = pd.Timestamp(sc["showcase"]["sp_expiry"])
        elif t == "orphaned_disk":
            acct = str(rng.choice(sc["orphan_accounts"]))
            it.update(account=acct, name=f"{PREFIX[acct]}-migration-{next(suffix)}")
        elif t == "decoy_planned_training":
            it.update(name=f"ml-train-{next(suffix)}", ticket=f"CHG-{next(ticket)}")
        elif t == "decoy_marketing_campaign":
            it.update(name="sf-egress", ticket=f"CHG-{next(ticket)}")
        elif t == "decoy_feature_launch":
            acct = str(rng.choice(sc["launch_accounts"]))
            name = "pay-fraud" if acct == "payments-prod" else "sf-search"
            it.update(account=acct, name=f"{name}-{next(suffix)}", ticket=f"CHG-{next(ticket)}")
        items.append(it)
    return sorted(items, key=lambda x: x["date"])


def apply_difficulty(world: dict, diff: dict) -> dict:
    w = copy.deepcopy(world)
    w["noise_scale"] = diff["noise_scale"]
    w["utilization"]["gpu_inference_median"] = diff["gpu_inference_median"]
    return w


def plant(world: dict, items: list, diff: dict, sc: dict) -> tuple[dict, list]:
    """Return (world with items, label rows without $ impact)."""
    w = copy.deepcopy(world)
    end = pd.Timestamp(w["period"]["end"])
    res = {r["id"]: r for r in w["resources"]}
    owner = {t: username(v["owner"]) for t, v in w["teams"].items()}
    team_of_acct = {a: v["team"] for a, v in w["accounts"].items()}
    labels = []

    def label(it, resources, account, team, start, stop, notes, ticket=None):
        is_decoy = it["type"].startswith("decoy")
        a = sc["actions"]["decoy" if is_decoy else it["type"]]
        labels.append({
            "item": it["key"], "label": "decoy" if is_decoy else "anomaly", "type": it["type"],
            "primary_resource": resources[0], "resources": ";".join(resources), "account": account,
            "owner_team": team, "owner_email": w["teams"][team]["email"],
            "start_date": start.date(), "end_date": stop.date(),
            "expected_action": a["action"], "action_tier": a["tier"], "ticket": ticket, "notes": notes})

    for it in items:
        d, t = it["date"], it["type"]
        if t == "idle_gpu":
            train_end = d + (sc["training_days"] - 1) * D
            w["resources"].append({
                "id": it["name"], "account": "ml-dev", "sku": it["sku"], "base_qty": 24, "count": 1,
                "profile": "flat", "app": "training", "active_from": str(d.date()),
                "created_by": owner["ml"], "gpu_level": sc["gpu_idle_pct"], "cpu_level": 0.5,
                "util_windows": [{"from": str(d.date()), "to": str(train_end.date()),
                                  "gpu_pct": sc["gpu_training_pct"], "cpu_pct": 30}]})
            label(it, [it["name"]], "ml-dev", "ml", train_end + D, end,
                  f"training {sc['training_days']} days, then left running idle")
        elif t == "transfer_spike":
            stop = d + (sc["spike_days"] - 1) * D
            res[it["name"]].setdefault("qty_windows", []).append(
                {"from": str(d.date()), "to": str(stop.date()), "factor": diff["spike_factor"]})
            acct = res[it["name"]]["account"]
            label(it, [it["name"]], acct, team_of_acct[acct], d, stop,
                  f"replication traffic ×{diff['spike_factor']} for {sc['spike_days']} days")
        elif t == "transfer_drift":
            stop = d + sc["drift_days"] * D
            res[it["name"]].setdefault("qty_windows", []).append(
                {"from": str(d.date()), "to": str(stop.date()), "ramp_to": diff["drift_to"]})
            acct = res[it["name"]]["account"]
            label(it, [it["name"]], acct, team_of_acct[acct], d, end,
                  f"NAT traffic ramps to ×{diff['drift_to']} over {sc['drift_days']} days, then stays")
        elif t == "untagged_resource":
            u, acct = diff["untagged"], it["account"]
            team = team_of_acct[acct]
            w["resources"].append({
                "id": it["name"], "account": acct, "sku": u["sku"], "base_qty": 24,
                "count": u["count"], "profile": "flat", "tags": False, "active_from": str(d.date()),
                "created_by": owner[team]})
            label(it, [it["name"]], acct, team, d, end,
                  f"{u['count']}× {u['sku']} launched without tags")
        elif t == "sp_expired":
            c = next(c for c in w["commitments"] if c["id"] == it["name"])
            c["end"] = str(d.date())
            label(it, c["covers"], "multiple", c["managed_by"], d, end,
                  f"Savings Plan {c['id']} ended, not renewed: same usage, on-demand price")
        elif t == "orphaned_disk":
            o, acct = sc["orphan"], it["account"]
            team = team_of_acct[acct]
            created = d - o["server_days"] * D
            disk = f"{it['name']}-vol"
            w["resources"] += [
                {"id": it["name"], "account": acct, "sku": o["server_sku"], "base_qty": 24, "count": 1,
                 "profile": "flat", "app": "migration", "active_from": str(created.date()),
                 "active_to": str((d - D).date()), "created_by": owner[team]},
                {"id": disk, "account": acct, "sku": "ebs_gp3", "base_qty": o["disk_gb"],
                 "profile": "flat", "app": "migration", "active_from": str(created.date()),
                 "attached_to": it["name"], "created_by": owner[team]}]
            label(it, [disk], acct, team, d, end,
                  f"server {it['name']} deleted, its {o['disk_gb']} GB disk still billed")
        elif t == "decoy_planned_training":
            stop = d + (sc["training_days"] - 1) * D
            w["resources"].append({
                "id": it["name"], "account": "ml-dev", "sku": "ec2_g5_12xlarge", "base_qty": 24, "count": 1,
                "profile": "flat", "app": "training", "active_from": str(d.date()), "active_to": str(stop.date()),
                "created_by": owner["ml"], "gpu_level": sc["gpu_training_pct"], "cpu_level": 30,
                "ticket": f"{it['ticket']}: approved training run, {sc['training_days']} days"})
            label(it, [it["name"]], "ml-dev", "ml", d, stop, "approved training run, shut down on time",
                  it["ticket"])
        elif t == "decoy_marketing_campaign":
            c = sc["campaign"]
            stop = d + (c["days"] - 1) * D
            res[it["name"]].setdefault("qty_windows", []).append(
                {"from": str(d.date()), "to": str(stop.date()), "factor": c["factor"]})
            w.setdefault("extra_events", []).append({
                "time": str(d - 2 * D + pd.Timedelta(hours=9, minutes=30)), "event_name": "ChangeApproved",
                "actor": owner["storefront"], "resource": it["name"], "ticket": it["ticket"],
                "details": f"{it['ticket']}: marketing campaign, expect ~{c['factor']}× web traffic for {c['days']} days"})
            label(it, [it["name"]], "storefront-prod", "storefront", d, stop,
                  "approved marketing campaign", it["ticket"])
        elif t == "decoy_feature_launch":
            f, acct = sc["feature_launch"], it["account"]
            team = team_of_acct[acct]
            app = it["name"].rsplit("-", 1)[0]
            w["resources"].append({
                "id": it["name"], "account": acct, "sku": f["sku"], "base_qty": 24,
                "count": f["count"], "profile": "flat", "app": app, "active_from": str(d.date()),
                "created_by": owner[team],
                "ticket": f"{it['ticket']}: launch new service {app} ({f['count']}× {f['sku']})"})
            label(it, [it["name"]], acct, team, d, end, "approved new service launch",
                  it["ticket"])
    return w, labels


def add_impact(labels: list, state: pd.DataFrame, clean: pd.DataFrame) -> None:
    cost = state.assign(cost=state.quantity * state.unit_price).set_index(["ResourceName", "date"]).cost
    base = clean.assign(cost=clean.quantity * clean.unit_price).set_index(["ResourceName", "date"]).cost
    for lb in labels:
        days = pd.date_range(lb["start_date"], lb["end_date"])
        idx = pd.MultiIndex.from_product([lb["resources"].split(";"), days], names=["ResourceName", "date"])
        extra = cost.reindex(idx, fill_value=0).sum() - base.reindex(idx, fill_value=0).sum()
        lb["impact_usd"] = round(float(extra), 2)
        lb["avg_daily_impact_usd"] = round(float(extra) / len(days), 2)
        lb["days"] = len(days)


def label_checks(labels, bill, met, cl, state) -> list:
    fails = []
    tickets = set(cl.ticket.dropna().str.split(":").str[0])
    for lb in labels:
        if lb["impact_usd"] <= 0:
            fails.append(f"{lb['item']}: impact not > 0")
        if lb["label"] == "decoy" and lb["ticket"] not in tickets:
            fails.append(f"{lb['item']}: decoy ticket missing in change log")
        if lb["label"] == "anomaly":
            evts = cl[cl.ResourceName.isin(lb["resources"].split(";"))]
            if evts.ticket.notna().any():
                fails.append(f"{lb['item']}: anomaly has a ticket")
        if lb["type"] == "untagged_resource" and bill[bill.ResourceName == lb["primary_resource"]].Tags.notna().any():
            fails.append(f"{lb['item']}: untagged resource has tags")
        if lb["type"] == "idle_gpu":
            g = met[(met.ResourceName == lb["primary_resource"]) & (met.date >= pd.Timestamp(lb["start_date"]))]
            if g.empty or g.gpu_pct.max() > 5:
                fails.append(f"{lb['item']}: idle GPU not idle")
        if lb["type"] == "orphaned_disk":
            disk_last = bill[bill.ResourceName == lb["primary_resource"]].ChargePeriodStart.max()
            if disk_last.date() != lb["end_date"]:
                fails.append(f"{lb['item']}: orphaned disk not billed to the end")
        if lb["type"] == "sp_expired":
            after = bill[bill.ResourceName.isin(lb["resources"].split(";"))
                         & (bill.ChargePeriodStart >= pd.Timestamp(lb["start_date"]))]
            if (after.PricingCategory != "Standard").any():
                fails.append(f"{lb['item']}: committed price after expiry")
    return fails


def chart(path, labels, state, clean, met):
    rows = (len(labels) + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(14, 3.2 * rows))
    for ax, lb in zip(axs.flat, labels):
        names = lb["resources"].split(";")
        s = state[state.ResourceName.isin(names)].assign(c=lambda x: x.quantity * x.unit_price)
        c = clean[clean.ResourceName.isin(names)].assign(c=lambda x: x.quantity * x.unit_price)
        s.groupby("date").c.sum().plot(ax=ax, lw=1.2, label="with item")
        if not c.empty:
            c.groupby("date").c.sum().plot(ax=ax, lw=1, ls="--", color="grey", label="without item")
        ax.axvspan(pd.Timestamp(lb["start_date"]), pd.Timestamp(lb["end_date"]),
                   color="red" if lb["label"] == "anomaly" else "green", alpha=0.12)
        ax.set_title(f"{lb['item']} {lb['type']} — {lb['primary_resource']} "
                     f"(+${lb['avg_daily_impact_usd']}/day)", fontsize=9)
        ax.set_ylabel("$ / day")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("")
        if lb["type"] == "idle_gpu":
            g = met[met.ResourceName == lb["primary_resource"]].set_index("date").gpu_pct
            ax2 = ax.twinx()
            ax2.plot(g.index, g.values, color="black", lw=0.8, label="GPU %")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("GPU %")
    axs.flat[0].legend(fontsize=8, loc="upper left")
    for ax in axs.flat[len(labels):]:
        ax.axis("off")
    fig.suptitle("Step 5 — medium / replica 1: planted items (red = anomaly window, green = decoy window)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def main() -> None:
    world = yaml.safe_load((ROOT / "data/config/world_calibrated.yaml").read_text())
    sc = yaml.safe_load((ROOT / "data/config/scenarios.yaml").read_text())
    cat = pd.read_csv(ROOT / "data/config/catalog.csv").set_index("sku_key")
    all_labels, all_ok = [], True

    alloc = allocate(sc)
    for r in range(1, sc["replicas"] + 1):
        items = plan_replica(r, alloc[r], sc, world)
        split = "dev" if r in sc["dev_replicas"] else "test"
        for dname, diff in sc["difficulties"].items():
            base_world = apply_difficulty(world, diff)
            w, labels = plant(base_world, items, diff, sc)
            clean = build_state(base_world, cat)
            state = build_state(w, cat)
            bill = build_billing(state, w, cat)
            side = build_side_tables(state, w, cat)
            add_impact(labels, state, clean)

            out = SYN / dname / f"r{r}"
            out.mkdir(parents=True, exist_ok=True)
            state.to_parquet(out / "resource_days.parquet", index=False)
            bill.to_parquet(out / "billing.parquet", index=False)
            for name, df in side.items():
                df.to_parquet(out / f"{name}.parquet", index=False)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ok = base_checks(state, bill, side["usage_metrics"], side["change_log"],
                                 side["owners"], side["commitments"], w)
            fails = [ln.strip() for ln in buf.getvalue().splitlines() if "FAIL" in ln]
            fails += label_checks(labels, bill, side["usage_metrics"], side["change_log"], state)
            all_ok &= ok and not fails
            n_a = sum(lb["label"] == "anomaly" for lb in labels)
            print(f"{dname:6} r{r} ({split:4}) items={len(labels):2} ({n_a} anomalies) rows={len(bill):5,}  checks: "
                  f"{'PASS' if ok and not fails else 'FAIL ' + '; '.join(fails)}")

            for lb in labels:
                lb.update({"case_id": f"{dname}-r{r}-{lb['item']}", "difficulty": dname, "replica": r,
                           "split": split})
            all_labels += labels
            if dname == "medium" and r == sc["showcase"]["replica"]:
                FIG.mkdir(parents=True, exist_ok=True)
                chart(FIG / "step5_medium_r1.png", labels, state, clean, side["usage_metrics"])

    gt = pd.DataFrame(all_labels)
    cols = ["case_id", "difficulty", "replica", "split", "item", "label", "type", "primary_resource",
            "resources", "account", "owner_team", "owner_email", "start_date", "end_date", "days",
            "impact_usd", "avg_daily_impact_usd", "expected_action", "action_tier", "ticket", "notes"]
    LAB.mkdir(parents=True, exist_ok=True)
    gt[cols].to_csv(LAB / "ground_truth.csv", index=False)

    daily_total = build_billing(build_state(world, cat), world, cat).groupby("ChargePeriodStart").EffectiveCost.sum().mean()
    print(f"\nAnswer key: {len(gt)} rows ({(gt.label == 'anomaly').sum()} anomalies, "
          f"{(gt.label == 'decoy').sum()} decoys) → data/labels/ground_truth.csv")
    print(f"\nAverage extra cost per day while active, $ (company normal ≈ ${daily_total:,.0f}/day):")
    print(gt.pivot_table(index="type", columns="difficulty", values="avg_daily_impact_usd", aggfunc="mean")
          [["easy", "medium", "hard"]].round(1).to_string())
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} · chart: reports/figures/step5_medium_r1.png")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
