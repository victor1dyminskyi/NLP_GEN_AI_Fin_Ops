"""Agent tools — read-only access to ONE dataset folder. Every number the agent reports comes from here.

The hard SQL patterns from the model benchmark (owner of an account, first appearance, attached server)
are ready-made tools, so the model does not have to write tricky SQL for them.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

TABLES = ["billing", "usage_metrics", "change_log", "owners", "commitments"]
MAX_ROWS = 25


def _fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    more = f"\n… {len(df) - MAX_ROWS} more rows" if len(df) > MAX_ROWS else ""
    return df.head(MAX_ROWS).to_csv(index=False, float_format="%.4g") + more


class Tools:
    def __init__(self, dataset_dir):
        self.dir = Path(dataset_dir)
        self.con = duckdb.connect()
        for t in TABLES:
            self.con.execute(f"CREATE VIEW {t} AS SELECT * FROM '{self.dir / (t + '.parquet')}'")

    def q(self, sql: str, params=None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    # ---------- tools exposed to the LLM ----------
    def run_sql(self, query: str) -> str:
        """Read-only SQL (SELECT/WITH only) on billing, usage_metrics, change_log, owners, commitments."""
        if not query.strip().lower().startswith(("select", "with")):
            return "ERROR: only SELECT / WITH queries are allowed."
        try:
            return _fmt(self.q(query))
        except Exception as e:  # noqa: BLE001 — return the error so the model can fix its query
            return f"ERROR: {str(e).splitlines()[0][:300]}"

    def resource_info(self, resource: str) -> str:
        """Account, owning team, service, price line, tags, first/last billed day of a resource."""
        df = self.q("""
            SELECT ResourceName, ResourceId, SubAccountName AS account, ServiceName,
                   any_value(ChargeDescription) AS price_line, any_value(Tags) AS tags,
                   strftime(MIN(ChargePeriodStart), '%Y-%m-%d') AS first_billed,
                   strftime(MAX(ChargePeriodStart), '%Y-%m-%d') AS last_billed,
                   ROUND(AVG(EffectiveCost), 2) AS avg_cost_per_day
            FROM billing WHERE ResourceName = ? GROUP BY ALL""", [resource])
        if df.empty:
            return f"No resource named {resource!r}."
        r = df.iloc[0].to_dict()
        team = self.q("SELECT team FROM owners WHERE ',' || replace(accounts, ' ', '') || ',' LIKE ?",
                      [f"%,{r['account']},%"])
        r["account_team"] = team.team.iloc[0] if len(team) else None
        r["tags"] = r["tags"] if isinstance(r["tags"], str) else "NONE (untagged)"
        return json.dumps(r, default=str)

    def cost_history(self, resource: str, around_date: str, days_before: int = 21, days_after: int = 14) -> str:
        """Daily cost, quantity, price per unit and pricing category around a date, plus before/after averages."""
        df = self.q(f"""
            SELECT ChargePeriodStart::DATE AS day, ROUND(EffectiveCost, 2) AS cost,
                   ROUND(PricingQuantity, 2) AS quantity, PricingUnit AS unit,
                   ROUND(ContractedUnitPrice, 4) AS unit_price, PricingCategory, CommitmentDiscountId
            FROM billing WHERE ResourceName = ?
              AND ChargePeriodStart BETWEEN DATE '{around_date}' - INTERVAL {int(days_before)} DAY
                                        AND DATE '{around_date}' + INTERVAL {int(days_after)} DAY
            ORDER BY day""", [resource])
        if df.empty:
            return "(no billing rows in that window)"
        d = pd.Timestamp(around_date)
        before, after = df[df.day < d], df[df.day >= d]
        def pct(a, b):
            return None if a in (None, 0) or b is None or a != a or b != b else round(100 * (b - a) / a, 1)
        summary = {
            "avg_cost_before": round(before.cost.mean(), 2) if len(before) else None,
            "avg_cost_after": round(after.cost.mean(), 2) if len(after) else None,
            "avg_quantity_before": round(before.quantity.mean(), 2) if len(before) else None,
            "avg_quantity_after": round(after.quantity.mean(), 2) if len(after) else None,
            "unit_price_before": before.unit_price.median() if len(before) else None,
            "unit_price_after": after.unit_price.median() if len(after) else None,
            "pricing_before": before.PricingCategory.mode().iloc[0] if len(before) else None,
            "pricing_after": after.PricingCategory.mode().iloc[0] if len(after) else None,
        }
        recent = df[df.day >= d - pd.Timedelta(days=3)].assign(day=lambda x: x.day.dt.strftime("%Y-%m-%d"))
        change = (f"CHANGE from {around_date}: cost {pct(summary['avg_cost_before'], summary['avg_cost_after'])}%, "
                  f"quantity {pct(summary['avg_quantity_before'], summary['avg_quantity_after'])}%, "
                  f"price per unit {pct(summary['unit_price_before'], summary['unit_price_after'])}%; "
                  f"pricing {summary['pricing_before']} → {summary['pricing_after']}")
        q_after = after.quantity.reset_index(drop=True)
        if len(before) and len(q_after) >= 3:
            new_level = q_after.tail(5).median()
            base = summary["avg_quantity_before"]
            reached = next((i for i, v in enumerate(q_after) if abs(v - base) >= 0.8 * abs(new_level - base)), None)
            if base and abs(new_level - base) / base >= 0.1 and reached is not None:
                change += f"; quantity reached its new level after {reached + 1} day(s)"
        return change + "\nSUMMARY " + json.dumps(summary, default=str) + "\nDAILY\n" + _fmt(recent)

    def usage(self, resource: str, around_date: str) -> str:
        """Average CPU / GPU % busy before and after a date (servers only)."""
        df = self.q(f"""
            SELECT CASE WHEN date < DATE '{around_date}' THEN 'before' ELSE 'after' END AS period,
                   COUNT(*) AS days, ROUND(AVG(cpu_pct), 1) AS avg_cpu_pct, ROUND(AVG(gpu_pct), 1) AS avg_gpu_pct,
                   ROUND(MAX(gpu_pct), 1) AS max_gpu_pct, ROUND(AVG(instances_avg), 1) AS avg_instances
            FROM usage_metrics WHERE ResourceName = ? GROUP BY 1 ORDER BY 1 DESC""", [resource])
        if df.empty:
            return "(no usage metrics — not a server, or never running)"
        b, a = (df[df.period == p].iloc[0] if (df.period == p).any() else None for p in ("before", "after"))
        lines = []
        for col, name in [("avg_gpu_pct", "GPU"), ("avg_cpu_pct", "CPU")]:
            if a is not None and a[col] == a[col]:
                lines.append(f"{name} busy: {b[col] if b is not None else 'n/a'}% before → {a[col]}% after "
                             f"({int(a.days)} days after)")
        return "CHANGE: " + "; ".join(lines) + "\n" + _fmt(df)

    def change_events(self, resource: str) -> str:
        """All change-log events for a resource (who, what, when, ticket, details), scheduler events summarised."""
        df = self.q("""SELECT event_time, event_name, actor, ticket, details FROM change_log
                       WHERE ResourceName = ? ORDER BY event_time""", [resource])
        sched = df.event_name.isin(["StopInstances", "StartInstances"])
        out = df[~sched]
        note = f"\n(+ {int(sched.sum())} weekly scheduler stop/start events)" if sched.any() else ""
        tickets = out.ticket.dropna().unique()
        head = "TICKETS: " + (" | ".join(tickets) if len(tickets) else "none") + "\n"
        return head + (_fmt(out) if len(out) else "(no create/delete/ticket events)") + note

    def attached_server(self, volume: str) -> str:
        """Which server a disk was attached to, and whether that server was terminated (and when)."""
        att = self.q("""SELECT details FROM change_log WHERE ResourceName = ? AND event_name = 'AttachVolume'""",
                     [volume])
        if att.empty:
            return "No AttachVolume event for this resource."
        server = att.details.iloc[0].replace("attached to ", "")
        term = self.q("""SELECT event_time, actor FROM change_log
                         WHERE ResourceName = ? AND event_name = 'TerminateInstances'""", [server])
        last = self.q("SELECT strftime(MAX(ChargePeriodStart), '%Y-%m-%d') d FROM billing WHERE ResourceName = ?",
                      [volume]).d.iloc[0]
        return json.dumps({"server": server,
                           "server_terminated_at": str(term.event_time.iloc[0]) if len(term) else None,
                           "terminated_by": term.actor.iloc[0] if len(term) else None,
                           "volume_last_billed_day": str(last)})

    def get_owner(self, account: str) -> str:
        """Team that owns an AWS account (accounts are a comma-separated list in owners)."""
        df = self.q("""SELECT team, owner, email, slack FROM owners
                       WHERE ',' || replace(accounts, ' ', '') || ',' LIKE ?""", [f"%,{account},%"])
        return _fmt(df) if len(df) else f"No team owns account {account!r}."

    def commitment_status(self, resource: str) -> str:
        """Savings Plans that cover a resource: discount, start, end, and which team manages renewals."""
        df = self.q("""SELECT commitment_id, type, discount, strftime(start, '%Y-%m-%d') AS start,
                              strftime("end", '%Y-%m-%d') AS "end", managed_by
                       FROM commitments WHERE ',' || replace(covers, ' ', '') || ',' LIKE ?""", [f"%,{resource},%"])
        return _fmt(df) if len(df) else "No Savings Plan covers this resource."


TOOL_SPECS = """
- run_sql(query): read-only SQL on billing, usage_metrics, change_log, owners, commitments. Use only if no other tool fits.
- resource_info(resource): account, owning team, service, price line, tags, first/last billed day, avg cost/day.
- cost_history(resource, around_date): daily cost/quantity/price per unit/pricing category around a date + before/after summary.
- usage(resource, around_date): average CPU % and GPU % busy before vs after a date (servers only).
- change_events(resource): who created/deleted it, when, change tickets, details.
- attached_server(volume): which server a disk was attached to and whether that server was terminated.
- get_owner(account): team, owner, email, slack for an AWS account.
- commitment_status(resource): Savings Plans covering the resource (discount, start, end, managed_by).
""".strip()
