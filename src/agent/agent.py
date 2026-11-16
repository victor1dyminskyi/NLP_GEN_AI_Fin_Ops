"""Component 2 + 3 — the investigating agent and the action gate.

Loop (one alert):
  alert → LLM decides next step (JSON) → we run the tool → result back to the LLM → … → final incident form
Protocol is plain JSON (works with any model / provider, no special tool-calling support needed):
  {"thought": "...", "tool": "<name>", "args": {...}}      ← ask for evidence
  {"thought": "...", "final": {...incident form...}}      ← done
Invalid forms are sent back with the reason. The same error 3× in a row stops the loop (escalated).
Every step is recorded (the trace shown in the UI).
Versions: v1 free agent · v2 + code-forced evidence checklist (worse: 33% vs 67%) ·
v3 = free agent again (the LLM decides what to check) + resource_kind hint in the alert + tools that state
changes plainly + owner_message written by the LLM.

Action gate (plain code, NOT the LLM):
  needs_approval  = action ∈ {stop_instance, recommend_sp_renewal, snapshot_then_delete_volume}
  escalated       = confidence < ESCALATE_BELOW, or type "other" → a human reviews it
  status          = no_action | auto_executed (simulated) | awaiting_approval | escalated_to_human
"""

import json
import os
import re
import time
from pathlib import Path

from llm import chat
from tools import TOOL_SPECS, Tools

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.getenv("LLM_MODEL", "finops-agent")  # qwen2.5-coder:7b + 16k context (ollama/Modelfile)
MAX_STEPS = 10
ESCALATE_BELOW = 0.6

TYPES = ["idle_gpu", "transfer_spike", "transfer_drift", "untagged_resource", "sp_expired", "orphaned_disk",
         "planned_change", "no_issue", "other"]
ACTIONS = ["stop_instance", "notify_owner_open_ticket", "tag_and_notify_owner", "recommend_sp_renewal",
           "snapshot_then_delete_volume", "no_action"]
NEEDS_APPROVAL = {"stop_instance", "recommend_sp_renewal", "snapshot_then_delete_volume"}
TOOL_NAMES = ["run_sql", "resource_info", "cost_history", "usage", "change_events", "attached_server",
              "get_owner", "commitment_status"]

SYSTEM = """You are a FinOps investigator. A detector raised an alert about one cloud resource.
Investigate it with the tools, then fill the incident form. Be factual and brief.

DATA DICTIONARY
{dictionary}

PLAYBOOK
{playbook}

TOOLS
{tools}

PROTOCOL — reply with exactly ONE JSON object per turn:
  {{"thought": "<one short sentence>", "tool": "<tool name>", "args": {{...}}}}
or, when you have enough evidence:
  {{"thought": "<one short sentence>", "final": {{
     "type": one of {types},
     "owner_team": one of {teams},
     "action": one of {actions},
     "start_date": "YYYY-MM-DD" (when the problem started),
     "confidence": number 0–1,
     "explanation": "2–4 sentences; every number must come from a tool result",
     "evidence": ["short fact copied from a tool result", "..."],
     "owner_message": "short Slack message to the owning team: what happened, the cost, what you propose
                       (empty string if action is no_action)"}}}}
YOU decide what to investigate. The alert says what kind of resource it is (resource_kind) — use that to
choose useful tools (e.g. servers: is it busy?; disks: is its server still there?; price changes: did a
Savings Plan end?). Always look at change_events for tickets and who created it. Compare what a ticket says
with what actually happened (duration, size). Then decide with the PLAYBOOK.
RULES: never invent numbers; actions are NOT tools; at least 2 tool calls; at most {max_steps} tool calls."""


def gate(form: dict) -> dict:
    action, conf = form["action"], float(form.get("confidence", 0))
    needs = action in NEEDS_APPROVAL
    if conf < ESCALATE_BELOW or form["type"] == "other":  # unsure or unclassified → a human decides
        status = "escalated_to_human"
    elif action == "no_action":
        status = "no_action"
    elif needs:
        status = "awaiting_approval"
    else:
        status = "auto_executed (simulated)"
    return {"needs_approval": needs, "escalated": status == "escalated_to_human", "status": status}


def validate(form, teams) -> str | None:
    if not isinstance(form, dict):
        return "final must be an object"
    for key, allowed in [("type", TYPES), ("owner_team", teams), ("action", ACTIONS)]:
        if form.get(key) not in allowed:
            return f"{key}={form.get(key)!r} is not one of {allowed}"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(form.get("start_date", ""))):
        return "start_date must be YYYY-MM-DD"
    try:
        c = float(form.get("confidence"))
        assert 0 <= c <= 1
    except Exception:  # noqa: BLE001
        return "confidence must be a number between 0 and 1"
    if form["action"] != "no_action" and not str(form.get("owner_message", "")).strip():
        return "owner_message must not be empty when an action is proposed"
    return None


GPU_FAMILIES = ("g4dn.", "g5.", "g6.", "p3.", "p4d.", "p5.")


def resource_kind(resource: str, tools: "Tools") -> str:
    """Plain label from the price line (agent-visible billing data). A hint, not a conclusion."""
    d = tools.q("SELECT any_value(ChargeDescription) d FROM billing WHERE ResourceName = ?", [resource]).d.iloc[0] or ""
    if "Instance Hour" in d:
        return "gpu_server" if any(g in d for g in GPU_FAMILIES) else "server"
    if "RDS" in d:
        return "database"
    if "snapshot" in d.lower():
        return "disk_snapshot"
    if "GB-month" in d and "storage" in d.lower() and "provisioned" in d.lower():
        return "disk"
    if "storage used" in d:
        return "object_storage"
    if "transfer" in d.lower() or "Data Processed" in d or "log data ingested" in d:
        return "data_transfer"
    if "LoadBalancer" in d or "NAT Gateway Hour" in d:
        return "network_device"
    return "other"


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def investigate(alert: dict, dataset_dir, model: str = MODEL, on_step=None) -> dict:
    """Run the agent on one alert. Returns the incident record (form + gate + full trace)."""
    tools = Tools(dataset_dir)
    teams = tools.q("SELECT team FROM owners ORDER BY team").team.tolist()
    system = SYSTEM.format(
        dictionary=(ROOT / "data/config/data_dictionary.md").read_text(),
        playbook=(ROOT / "data/config/playbooks.md").read_text(),
        tools=TOOL_SPECS, types=TYPES, teams=teams, actions=ACTIONS, max_steps=MAX_STEPS)
    alert = {**alert, "resource_kind": resource_kind(alert["resource"], tools)}
    alert_text = json.dumps({k: v for k, v in alert.items() if v is not None and v == v}, default=str)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"ALERT: {alert_text}\nStart the investigation."}]
    trace, t0, form, tools_used, fixes = [], time.time(), None, 0, 0
    called, last_error, same_error = set(), None, 0

    for step in range(MAX_STEPS + 3):
        text, secs = chat(model, msgs, think=False, response_format={"type": "json_object"})
        msg = parse(text)
        entry = {"step": step + 1, "seconds": round(secs, 1), "raw": text[:2000]}
        if msg is None:
            entry["error"] = "not valid JSON"
            reply = "Your reply was not valid JSON. Reply with exactly one JSON object."
        elif "final" in msg:
            entry["thought"] = msg.get("thought")
            problem = validate(msg["final"], teams)
            if problem is None and (tools_used < 2 or "change_events" not in called):
                problem = "call at least 2 tools, including change_events, before the final answer"
            if problem is None:
                form = msg["final"]
                entry["final"] = form
                trace.append(entry)
                if on_step:
                    on_step(entry)
                break
            entry["error"] = problem
            fixes += 1
            reply = f"Invalid final answer: {problem}. Fix it and reply again with JSON."
        elif msg.get("tool") in TOOL_NAMES and tools_used < MAX_STEPS:
            entry.update(thought=msg.get("thought"), tool=msg["tool"], args=msg.get("args", {}))
            try:
                result = getattr(tools, msg["tool"])(**(msg.get("args") or {}))
            except TypeError as e:
                result = f"ERROR: wrong arguments: {e}"
            entry["result"] = result[:3000]
            tools_used += 1
            called.add(msg["tool"])
            reply = f"TOOL RESULT ({msg['tool']}):\n{result[:3000]}"
        else:
            bad = msg.get("tool")
            if tools_used >= MAX_STEPS:
                entry["error"], reply = "tool budget used up", "No more tool calls. Give your final answer now."
            elif bad in ACTIONS:
                entry["error"] = f"used action {bad!r} as a tool"
                reply = (f"'{bad}' is an ACTION, not a tool. Actions go inside the final answer: "
                         f'{{"thought": "...", "final": {{..., "action": "{bad}", ...}}}}. Tools are: {TOOL_NAMES}.')
            else:
                entry["error"] = f"unknown tool {bad!r}"
                reply = f"Unknown tool {bad!r}. Tools are: {TOOL_NAMES}. To finish, reply with a JSON containing \"final\"."
        trace.append(entry)
        if on_step:
            on_step(entry)
        same_error = same_error + 1 if entry.get("error") and entry.get("error") == last_error else 0
        last_error = entry.get("error")
        if same_error >= 2:  # the same mistake 3 times in a row → stop, a human takes over
            break
        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": reply}]

    if form is None:  # gave up → a human must look at it
        form = {"type": "other", "owner_team": None, "action": "no_action", "start_date": str(alert["start_date"])[:10],
                "confidence": 0.0, "explanation": "Agent could not reach a valid conclusion.", "evidence": []}
    return {"alert": {k: (str(v) if k == "start_date" else v) for k, v in alert.items()},
            "resource": alert["resource"], "form": form, "gate": gate(form),
            "model": model, "llm_calls": sum(1 for e in trace), "tool_calls": tools_used, "form_fixes": fixes,
            "seconds": round(time.time() - t0, 1), "trace": trace}
