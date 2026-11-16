"""FinOps Anomaly Agent — app + presentation. Pages tell one story: problem → data → system → demo → results.

Run:  uv run streamlit run app/app.py
Needs Ollama running locally with the `finops-agent` model (see README) for the live pages.
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
for sub in ["src/agent", "src/eval", "src/data_gen"]:
    sys.path.insert(0, str(ROOT / sub))

from agent import ESCALATE_BELOW, NEEDS_APPROVAL, investigate  # noqa: E402
from detector import detect, load_params  # noqa: E402

SYN, REP, FIG, CFG = ROOT / "data/synthetic", ROOT / "reports", ROOT / "reports/figures", ROOT / "data/config"
SC = yaml.safe_load((CFG / "scenarios.yaml").read_text())
DEV, TEST = SC["dev_replicas"], SC["test_replicas"]

st.set_page_config(page_title="FinOps Anomaly Agent", page_icon="💸", layout="wide")


# ================================================================ helpers
def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def img(name: str, caption: str = ""):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


def pct_table(rows, cols):
    return pd.DataFrame(rows, columns=cols)


def next_step(text: str):
    st.divider()
    st.markdown(f"➡️ **Next:** {text}")


def dataset_picker(key: str, default=("medium", 1)) -> Path:
    c1, c2 = st.columns(2)
    diff = c1.selectbox("Difficulty — how hard the anomalies are to see", ["easy", "medium", "hard"],
                        index=["easy", "medium", "hard"].index(default[0]), key=f"{key}_d")
    rep = c2.selectbox("Scenario — which anomalies happen, and when", list(range(1, SC["replicas"] + 1)), index=default[1] - 1,
                       key=f"{key}_r",
                       format_func=lambda r: f"r{r} — {'practice' if r in DEV else 'sealed final exam'}")
    st.caption("Same scenario = the same anomalies on the same dates at every difficulty; only how visible they are changes "
               "(noise, spike size, how busy normal GPUs are). Scenarios 1–2 = practice, 3–5 = sealed final exam.")
    if rep in TEST:
        st.info("Final-exam data: live runs here are demonstrations only — the official scores were computed once "
                "and are on the Results page.")
    return SYN / diff / f"r{rep}"


@st.cache_data(show_spinner=False)
def cached_detect(path: str) -> pd.DataFrame:
    return detect(Path(path))


@st.cache_data(show_spinner=False)
def planted_series(difficulty: str, replica: int):
    """Daily cost of each planted item's resources WITH the item and WITHOUT it (same random noise)."""
    from build_datasets import apply_difficulty, plan_replica, allocate, plant  # noqa: F401
    from generate_billing import build_state
    world = yaml.safe_load((CFG / "world_calibrated.yaml").read_text())
    cat = pd.read_csv(CFG / "catalog.csv").set_index("sku_key")
    clean = build_state(apply_difficulty(world, SC["difficulties"][difficulty]), cat)
    state = pd.read_parquet(SYN / difficulty / f"r{replica}" / "resource_days.parquet")
    labels = pd.read_csv(ROOT / "data/labels/ground_truth.csv")  # presentation only — the agent never reads this
    labels = labels[(labels.difficulty == difficulty) & (labels.replica == replica)]
    for df in (state, clean):
        df["cost"] = df.quantity * df.unit_price
    return state, clean, labels


def show_step(e: dict):
    if e.get("tool"):
        st.markdown(f"**Step {e['step']} · tool `{e['tool']}`** `{json.dumps(e.get('args', {}))}` · {e['seconds']} s")
        if e.get("thought"):
            st.caption(f"💭 {e['thought']}")
        st.code(e.get("result", "")[:1500], language="text")
    elif e.get("final"):
        st.markdown(f"**Step {e['step']} · final answer** · {e['seconds']} s")
        if e.get("thought"):
            st.caption(f"💭 {e['thought']}")
    else:
        st.markdown(f"**Step {e['step']} · ⚠️ {e.get('error')}** · {e['seconds']} s")


STATUS_ICON = {"no_action": "⚪", "auto_executed (simulated)": "🟢", "awaiting_approval": "🟠",
               "escalated_to_human": "🔴"}


def incident_card(inc: dict, key: str):
    f, g = inc["form"], inc["gate"]
    st.subheader(f"{STATUS_ICON.get(g['status'], '')} {inc['resource']} — {f['type']}")
    c = st.columns(4)
    c[0].metric("Owner team", f.get("owner_team") or "—")
    c[1].metric("Proposed action", f["action"])
    c[2].metric("Agent's confidence", f"{float(f['confidence']):.2f}")
    c[3].metric("Status (set by code)", g["status"])
    st.markdown(f"**Explanation.** {f.get('explanation', '')}")
    if f.get("evidence"):
        st.markdown("**Evidence (copied from tool results):**\n" + "\n".join(f"- {x}" for x in f["evidence"]))
    if f.get("owner_message"):
        st.markdown("**Message to the cost owner (draft written by the LLM):**")
        st.info(f["owner_message"])
    st.caption(f"{inc['tool_calls']} tool calls · {inc['llm_calls']} LLM calls · {inc['seconds']} s · model {inc['model']}")
    if g["status"] == "awaiting_approval":
        decided = st.session_state.setdefault("decisions", {}).get(key)
        if decided:
            st.success(f"Human decision: **{decided}** — "
                       f"{'action executed (simulated)' if decided == 'approved' else 'no action taken'}")
        else:
            st.warning(f"Risky action **{f['action']}** waits for a human. The agent cannot approve it itself.")
            b1, b2 = st.columns(2)
            if b1.button("✅ Approve", key=f"ap_{key}"):
                st.session_state["decisions"][key] = "approved"
                st.rerun()
            if b2.button("❌ Reject", key=f"rj_{key}"):
                st.session_state["decisions"][key] = "rejected"
                st.rerun()
    elif g["status"] == "escalated_to_human":
        st.error("Escalated: the agent was unsure (confidence below 0.6, or type 'other'). A human investigates.")


# ================================================================ 1. Problem
def page_problem():
    st.title("💸 FinOps Anomaly Agent")
    st.markdown("#### An AI assistant that watches a company's cloud bill, finds unexpected costs, works out why they "
                "happened and who is responsible, and prepares the fix for a human to approve.")
    st.subheader("The problem")
    c1, c2, c3 = st.columns(3)
    c1.error("**Found late.** An idle GPU server costs ~$136 per day. Without monitoring it is often noticed only "
             "when the monthly invoice arrives.")
    c2.warning("**Slow to investigate.** A FinOps analyst digs by hand: which service → which account → which "
               "resource → who created it → is it planned?")
    c3.info("**AI tools exist but are not trusted.** In a July 2026 study of 28 FinOps practitioners (MSc "
            "dissertation, BSBI / UCA; summarised by FinOps Weekly, 28 Jul 2026), lack of trust in automated outputs "
            "was the most-cited barrier (38%). No vendor publishes how accurate its AI is.")
    st.subheader("What already exists")
    st.table(pd.DataFrame([
        ["AWS Cost Anomaly Detection + Amazon Q (Jun 2026)", "finds anomalies, explains the cause", "reports only — no fixes; no published accuracy"],
        ["AWS FinOps Agent (preview, Jun 2026)", "finds cause and owner, posts to Jira/Slack", "reports only — cannot stop, delete or change any server, database or network resource; a person must fix it"],
        ["Finout Agents (early access, Jun 2026)", "detector → investigator → orchestrator; reversible fixes run automatically, destructive ones wait for approval",
         "only an illustrative example — no published accuracy"],
    ], columns=["Product", "Does", "Missing"]))
    st.subheader("What our assistant does — and what it does not")
    c1, c2 = st.columns(2)
    c1.success("""**Does**
1. Scans every resource's daily cost and flags unusual changes
2. Investigates each flag: checks usage, tickets, prices, who created it
3. Names the cause, the responsible team and the fix
4. Drafts the message to that team
5. Low-risk fixes (notify, add tags) run automatically; risky ones (stop a server, buy a commitment) wait for a human
6. **Measures how often it is right** — the part vendors don't publish""")
    c2.error("""**Does not**
- Change anything in a real AWS account — every action is **simulated** in this prototype
- Optimise the whole account (rightsizing etc.) — only investigates anomalies
- Replace the human for risky decisions""")
    st.subheader("Requirements we chose for the scope (what the system must do)")
    st.table(pd.DataFrame([
        ["Find anomalies in a daily cloud bill (FOCUS format)", "Detector catch rate / false alarms"],
        ["Explain each one: cause, owner, action", "Agent accuracy vs answer key"],
        ["Never take a risky action without a human", "0 risky actions without approval (hard rule)"],
        ["Ignore planned, approved cost increases", "Decoys dismissed"],
        ["Every number must come from the data, not from the AI", "Tools compute numbers; AI only quotes them"],
        ["Free to run, no API key, runs on a laptop", "Local open model (Ollama)"],
        ["Same input → same answer", "Fixed seed; reproducibility check"],
    ], columns=["Requirement", "How we check it"]))
    st.subheader("What you need to run it")
    st.table(pd.DataFrame([
        ["Computer", "Mac with Apple Silicon, 16 GB RAM (tested on M2 Pro, 16 GB). No separate graphics card — the "
                     "model runs on the Mac's built-in GPU"],
        ["Disk", "~5 GB for the agent model (qwen2.5-coder 7B); +10 GB to also try qwen3 and llama3.1"],
        ["Software", "Ollama (runs the AI model locally), Python 3.11+ with uv, a web browser"],
        ["Internet / accounts", "only to download once — no API key, no cloud account, $0"],
        ["Speed", "detector: ~1 s per dataset · agent: ~15 s per alert · one dataset (≈ 12 alerts): ~3 min"],
    ], columns=["Requirement", "Detail"]))
    next_step("we need data — a company's cloud bill over several months, with anomalies whose answers we know.")


# ================================================================ 2. Data
def page_data():
    st.title("📂 Data — what we needed and what exists")
    st.markdown("""
To test the assistant we need **several months of one company's daily cloud bill, with anomalies whose correct answer
we know**. We searched for public data first.
""")
    st.subheader("1 · What we found")
    st.table(pd.DataFrame([
        ["FOCUS Sample Data", "FinOps Foundation (GitHub)", "Sep 2024 billing month, published Dec 2024",
         "Real, anonymised bill of one company, 94k AWS rows, 50 services", "✅ real prices · ✅ real spending mix · ❌ only 1 month, no anomalies"],
        ["Azure VM trace V2", "Microsoft (GitHub)", "2019, 30 days",
         "Average CPU load of 2.7 million real virtual servers", "✅ how busy normal servers are"],
        ["3 research papers on GPU clusters", "ATC'23, NSDI'24, OSDI'26", "2023–2026",
         "Measured how busy GPUs serving AI models are", "✅ how busy normal GPUs are"],
        ["Databend AWS billing sample", "Databend", "Aug 2024",
         "Another 1-month AWS bill in AWS's own detailed format (CUR)", "❌ checked, not used — same gap (1 month), so it added nothing"],
        ["Kaggle / Hugging Face", "—", "—", "searched for labelled cloud-cost anomaly data", "❌ none exists"],
    ], columns=["Source", "From", "Date", "What it is", "What it gave us"]))
    st.markdown("**FOCUS** = the FinOps Foundation's standard format for detailed cloud bills: one row per resource "
                "per period, the same columns for AWS, Azure, Google and Oracle.")

    st.subheader("2 · The gap")
    st.markdown("No public dataset has **months of history + anomalies with known answers + usage and change logs**. "
                "So we build a realistic company ourselves — from the **real ingredients** above.")

    st.subheader("3 · What each real source gave us")
    st.markdown("**From the FOCUS bill — prices and the spending mix.** We copied 21 real prices (e.g. \"$0.34 per hour "
                "for a c5.2xlarge server\") and measured how the real company's bill is split:")
    st.table(pct_table([["Servers (CPU)", "63.7%"], ["Servers (GPU)", "23.0%"], ["Disks + backups", "4.8%"],
                        ["Data transfer", "3.0%"], ["Databases", "2.2%"], ["Network gateways", "2.1%"],
                        ["Logs", "1.1%"], ["File storage (S3)", "0.1%"]], ["Group", "Share of the real bill"]))
    st.markdown("""
**From the Azure trace — how busy normal servers are.** For each of 2.7 million servers Microsoft recorded its average
CPU load. We kept the 237,558 servers that ran at least 7 days (like the always-on servers of our company) and looked
at the spread of their average load:
""")
    st.table(pd.DataFrame([["Web / API servers", "3.7%", "6.4%", "14.5%"],
                           ["Batch servers", "2.8%", "5.7%", "10.1%"]],
                          columns=["Server type", "25% of servers are below", "half are below (median)", "75% are below"]))
    st.markdown("""
→ **Real servers are mostly idle even when everything is normal** (half below ~6% CPU). Each server group of our
company gets a load drawn from exactly this real spread.

**From the GPU papers — how busy normal AI-serving GPUs are.** "Inference" = a trained AI model answering requests
(for our shop: choosing recommended products for each visitor). The studies disagree:
""")
    st.table(pd.DataFrame([["ATC'23 study", "25%"], ["Alibaba PAI (NSDI'24)", "20%"], ["Alibaba ASI (OSDI'26)", "6%"]],
                          columns=["Study", "Typical GPU busy %"]))
    st.markdown("""
This matters because an **idle** (forgotten) GPU shows ~0%. If normal GPUs run at 25%, idle stands out; at 6% it
almost doesn't. Since we cannot know which is "right", we test all three (see next page: *easy / medium / hard*).
""")
    next_step("how we turned these ingredients into a company.")


# ================================================================ 3. Synthetic company
def page_synthetic():
    st.title("🏭 Building the company: DemoShop")
    st.markdown("An online shop with an AI recommendation engine: **6 teams, 8 AWS accounts, 38 resources**, "
                "6 months (Mar–Aug 2026) of daily bills.")

    st.subheader("Step 1 · Real prices")
    st.markdown("Every price comes from a real row of the FOCUS bill (21 price lines). No price is invented.")

    st.subheader("Step 2 · First draft of the company, then calibration")
    st.markdown("""
I designed the company: which teams, which servers, roughly how much traffic. Those first guesses were wrong —
storage and traffic were **4× too large** compared with the real bill. So we **calibrated**: we kept the real prices
and changed only the *amounts* (number of servers, GB) until each group's share of the bill matched the real company.

```
amount for a group = real share of that group × total bill ÷ real price
```
**Where does the total bill come from?** Some things can't be fractional — you can't have half a database. The
databases + network gateways cost ≈ $860/month, and in the real bill those groups are 4.3% of the total,
so the total = $860 ÷ 0.043 ≈ **$20,300/month**. Every other group then gets its real share of that total.
""")
    st.table(pct_table([
        ["Servers (CPU)", "63.7%", "30.3%", "64.2%"], ["Servers (GPU)", "23.0%", "17.9%", "22.8%"],
        ["Disks + backups", "4.8%", "5.9%", "4.6%"], ["Data transfer", "3.0%", "17.6%", "3.0%"],
        ["Databases", "2.2%", "5.8%", "3.7%"], ["Network gateways", "2.1%", "0.7%", "0.5%"],
        ["Logs", "1.1%", "3.9%", "1.1%"], ["File storage (S3)", "0.1%", "17.9%", "0.1%"],
    ], ["Group", "Real company (target)", "DemoShop — my first guesses", "DemoShop — after changing the amounts"]))
    st.markdown("""
**Why 64.2% and not exactly 63.7%?** 63.7% is the *plan*; 64.2% is *reality after rounding to whole servers*.
""")
    st.table(pd.DataFrame([
        ["Plan for CPU servers", "63.7% × $20,271", "$12,907 / month"],
        ["…which means, e.g., cache servers", "formula says 6.51", "we can't buy half a server → 7"],
        ["Reality after rounding all groups", "$13,373 of $20,828", "= 64.2%"],
    ], columns=["", "Calculation", "Result"]))
    st.caption("Same reason for the other small differences: databases and gateways are whole machines we did not "
               "change. Every group ends within ~1.5 percentage points of the real company.")

    st.subheader("Step 3 · Six months of daily bills")
    st.code("""quantity(day) = base × weekday pattern × growth × noise
cost(day)     = quantity(day) × real price""", language="text")
    st.table(pd.DataFrame([
        ["base", "the resource's normal daily amount after calibration", "20 web servers × 24 h = 480 hours/day"],
        ["weekday pattern", "dev servers mostly off at weekends; shop traffic higher at weekends", "weekend × 0.35"],
        ["growth", "stored data grows slowly", "+4% per month"],
        ["noise", "random day-to-day wiggle", "traffic ±15%, servers ±1–5%"],
    ], columns=["Part", "Meaning", "Example"]))
    img("step3_daily_cost.png", "The normal company: ~$680/day, small weekend dips")

    st.subheader("Step 4 · Four more tables, all derived from the same daily state")
    st.table(pd.DataFrame([
        ["Bill (FOCUS)", "cost, hours/GB, price per unit, tags, Savings Plan per resource per day", "what we paid"],
        ["Usage metrics", "average CPU % / GPU % per server per day (only on days it runs)", "is it busy or idle?"],
        ["Change log", "who created / deleted what, when, with which ticket", "who did it, was it planned?"],
        ["Owners", "team → owner, email, Slack, accounts", "who is responsible?"],
        ["Savings Plans", "discount commitments: discount, start, end", "did a discount expire?"],
    ], columns=["Table", "Contains", "Answers"]))
    st.caption("Because every table is computed from the same daily state, they always agree: a stopped server has "
               "no cost, no usage row, and a 'stopped' event.")

    st.subheader("Step 5 · Planting anomalies — and decoys")
    st.table(pd.DataFrame([
        ["Idle GPU", "a GPU server finishes training but is never switched off", "new server; GPU 85% → 0.5%"],
        ["Data-transfer spike", "a misconfigured job sends far more data for a week", "GB per day × 1.5–10"],
        ["Data-transfer drift", "traffic creeps up over two weeks and stays", "GB per day rising"],
        ["Untagged resource", "servers launched without the team label", "new server, empty tags"],
        ["Expired Savings Plan", "a 30% discount ends, nobody renews it", "same hours, price +43%"],
        ["Orphaned disk", "a server is deleted, its disk is forgotten", "disk still billed"],
        ["**Decoy** (not an anomaly)", "a planned, approved increase (training run, campaign, launch)", "change with a ticket"],
    ], columns=["Type", "Story", "What changes in the data"]))
    st.markdown("""
**How do we know the cost without the anomaly?** Because we created it: we generate the same company twice — once with
the planted item and once without, with exactly the same random noise. The difference is the exact extra cost:
`extra $ = cost(with item) − cost(without item)`.
""")
    try:
        state, clean, labels = planted_series("medium", 1)
        pick = st.selectbox("See one planted item (practice data, medium difficulty)", labels.case_id,
                            format_func=lambda c: f"{labels.set_index('case_id').loc[c, 'type']} — "
                                                  f"{labels.set_index('case_id').loc[c, 'primary_resource']}")
        lb = labels.set_index("case_id").loc[pick]
        res = lb.resources.split(";")
        a = state[state.ResourceName.isin(res)].groupby("date").cost.sum().rename("with the item")
        b = clean[clean.ResourceName.isin(res)].groupby("date").cost.sum().rename("without the item")
        df = pd.concat([a, b], axis=1).fillna(0).reset_index().melt("date", var_name="series", value_name="$ per day")
        line = alt.Chart(df).mark_line().encode(x="date:T", y="$ per day:Q", color="series:N",
                                                strokeDash=alt.condition(alt.datum.series == "without the item",
                                                                         alt.value([5, 4]), alt.value([1, 0])))
        band = alt.Chart(pd.DataFrame({"s": [pd.Timestamp(lb.start_date)], "e": [pd.Timestamp(lb.end_date)]})).mark_rect(
            opacity=0.15, color="red" if lb.label == "anomaly" else "green").encode(x="s:T", x2="e:T")
        st.altair_chart(band + line, width="stretch")
        st.caption(f"{'Red' if lb.label == 'anomaly' else 'Green'} band = when the {lb.label} is active · "
                   f"extra ≈ ${lb.avg_daily_impact_usd}/day · {lb.notes}")
    except Exception as e:  # noqa: BLE001
        st.caption(f"(chart unavailable: {e})")

    st.subheader("Step 6 · Easy, medium, hard")
    st.markdown("Some numbers can't be measured from real data (how noisy, how big an anomaly is, how busy GPUs are). "
                "Instead of guessing one value, we build the company at three difficulty levels:")
    st.table(pd.DataFrame([
        ["Normal GPU load", "25%", "20%", "6%", "idle (0.5%) hides better when normal is also low"],
        ["Day-to-day noise", "×0.33", "×1", "×2", "real changes drown in noise"],
        ["Data-transfer spike", "×10", "×3", "×1.5", "smaller spikes are harder to see"],
        ["Data-transfer drift reaches", "×3", "×2", "×1.3", "slower, smaller drift"],
        ["Untagged server", "~$33/day", "~$33/day", "~$1/day", "a cheap one is easy to miss"],
    ], columns=["Setting", "Easy", "Medium", "Hard", "Why it gets harder"]))

    st.subheader("Step 7 · Practice data and a sealed final exam")
    st.markdown("""
For each difficulty we generated **5 different half-years** of the same company (different dates, resources and
numbers of anomalies — 7 to 12 each, so nobody can rely on a fixed count). That's 15 datasets, 147 planted items.

- **2 half-years = practice.** We built and tuned the assistant on them, looking at its mistakes.
- **3 half-years = sealed final exam.** Never looked at while building; run once at the end. That way the final
  score shows how the assistant does on data it has never seen — not on data it was tuned for.

The answer key is stored separately; automatic tests check that the assistant's code never reads it.
""")
    next_step("how the assistant works.")


# ================================================================ 4. System
def page_system():
    st.title("🧩 How the assistant works")
    svg = (ROOT / "app/assets/how_it_works.svg").read_text()
    # our own static file (not user/LLM content) → safe to embed; iframe keeps the SVG animation running
    st.iframe(f'<div style="width:100%;margin:0">{svg.replace("<svg ", "<svg style=\'width:100%;height:auto\' ", 1)}</div>',
              height=640)
    st.caption("The red dot is one anomaly travelling through the system. Example from the practice data (medium/r1).")

    st.subheader("① Data: 5 tables")
    st.markdown("Bill, usage metrics, change log, owners, Savings Plans — built as shown on the previous page. "
                "The **alerts** are not a sixth stored table: they are produced fresh by the detector on each run.")

    st.subheader("② Detector — plain statistics, no AI")
    st.markdown("""
Why a detector first? The bill has ~7,500 rows. A small AI model cannot read them all, and statistics can check every
resource every day in a second. The detector decides **where to look**; the AI decides **what it means**.

**Rule (a) — unusual cost.** For each resource, compare today with its own last 28 days:
""")
    st.code("""normal  = median of the last 28 days            (a rolling median)
spread  = 1.4826 × median distance from normal   (a rolling "robust standard deviation")
score   = (today − normal) ÷ spread
alert if score > 3 for 2 days in a row AND today − normal ≥ $0.50""", language="text")
    st.markdown("""
- **Median, not average:** one crazy day does not distort "normal". **28 days** = 4 full weeks, so weekends are normal.
- **1.4826** turns the median distance into the same scale as a standard deviation (for bell-shaped data).
- **3, 2 days, $0.50** were *tuned* on the practice data (60 combinations tried), then frozen.
- *Example:* traffic normally $10/day, typical wiggle $1.20 → spread $1.78. Today $23 → score 7.3 → unusual.

**Rule (b) — price change:** price per unit (cost ÷ hours or GB) ≥ 5% above its 28-day median → *expired discount*.
**Rule (c) — new resource:** first appears after the first 28 days and costs ≥ $0.50/day.
**Rule (d) — orphaned disk:** a disk whose server was deleted (change log) is still billed — its cost doesn't
change, so rules (a)–(c) can't see it.
""")

    st.subheader("③ Agent — a local AI model that investigates each alert")
    st.markdown("""
It receives the alert (resource, date, normal vs current $/day, kind of resource) and **decides itself** which tools to
call, one at a time, reading each result before choosing the next. Its tool pool — all **read-only**, all on the 5 tables:
""")
    st.table(pd.DataFrame([
        ["resource_info", "account, team, service, price line, tags, first/last billed day"],
        ["cost_history", "daily cost, quantity, price per unit around the alert date + plain summary of the change"],
        ["usage", "CPU % / GPU % busy before vs after the date"],
        ["change_events", "who created/deleted it, when, which ticket"],
        ["attached_server", "for a disk: which server it belonged to, was that server deleted"],
        ["get_owner", "team, owner, email, Slack for an account"],
        ["commitment_status", "Savings Plans covering the resource, their end date"],
        ["run_sql", "a free SQL query (read-only) — the only place the AI writes code, for questions no tool answers"],
    ], columns=["Tool", "Returns"]))
    st.markdown("""
**What code does the AI write?** Only read-only SQL queries (`run_sql`), when no ready-made tool fits. It never writes
code that changes anything — the fixes are proposed in the form and executed (simulated) only by the gate.

**Its final answer is a fixed form** (choices from lists, so it can be scored): type of anomaly · owner team · action ·
start date · confidence 0–1 · explanation · evidence · draft message to the owner.

**How is confidence built?** The model rates itself (0–1) following the playbook: 0.9+ = evidence clearly matches,
below 0.6 = unsure. **We found this self-rating unreliable** — wrong answers often came with 0.9–1.0. That is why the
decision about acting is made by code, not by the model's confidence alone.
""")

    st.subheader("What the AI agent is given — its context, and how its behaviour is controlled")
    st.table(pd.DataFrame([
        ["Instructions (every alert)", "role · data dictionary (meaning of each column) · FinOps playbook (signs of each anomaly type, matching "
                                       "action, who the owner is, how to rate confidence) · list of the 8 tools · answer format"],
        ["The case", "the alert: resource, kind of resource (server / GPU server / disk / data transfer …), date, normal vs current $/day"],
        ["Evidence", "only what it asks for: each tool result is added to the conversation, step by step"],
        ["Never given", "the answer key, the type of the planted item, or any dataset-specific hint (checked by automatic tests)"],
    ], columns=["Context", "Content"]))
    st.table(pd.DataFrame([
        ["Fixed answer format", "JSON with choices from lists → invalid answers are sent back with the reason"],
        ["Must look before judging", "at least 2 tools, always the change log (tickets) before the final answer"],
        ["Repeated mistakes", "the same error 3× in a row → stop, a human takes over"],
        ["Deterministic", "temperature 0, fixed seed → the same alert gives the same answer"],
        ["Cannot act", "it only fills the form; the gate (code) decides what happens"],
    ], columns=["Behaviour control", "How"]))
    with st.expander("Show the agent's full instructions (exactly what the model reads)"):
        from agent import ACTIONS, MAX_STEPS, SYSTEM, TYPES
        from tools import TOOL_SPECS
        st.code(SYSTEM.format(dictionary=(CFG / "data_dictionary.md").read_text(),
                              playbook=(CFG / "playbooks.md").read_text(), tools=TOOL_SPECS, types=TYPES,
                              teams=["analytics", "data", "ml", "payments", "platform", "storefront"],
                              actions=ACTIONS, max_steps=MAX_STEPS), language="text")

    st.subheader("④ Action gate — when it acts alone and when it asks a human (plain code)")
    st.table(pd.DataFrame([
        [f"confidence below {ESCALATE_BELOW}, or type 'other'", "🔴 escalated to a human"],
        ["action = no action (planned change / no issue)", "⚪ nothing happens"],
        [f"risky action: {', '.join(sorted(NEEDS_APPROVAL))}", "🟠 waits for Approve / Reject"],
        ["low-risk action: notify owner, add tags", "🟢 executed automatically (simulated)"],
    ], columns=["Condition", "Result"]))

    st.subheader("One alert through the whole system (expired Savings Plan)")
    st.markdown("""
1. **Detector:** web servers' price per hour $0.238 → $0.340 (+43%) on 1 July, hours unchanged → price alert.
2. **Agent:** `resource_info` (server, team storefront) → `cost_history` (price +42.9%, hours +1.2%, discounted →
   on-demand) → `commitment_status` (plan ended 1 July, managed by platform) → `change_events` (no ticket).
3. **Form:** expired Savings Plan · owner platform · recommend renewal · confidence 0.9 · message to #team-platform.
4. **Gate:** renewing costs money → waits for a human → Approve.
""")
    next_step("see it run live.")


# ================================================================ 5. Demo
def page_demo():
    st.title("▶️ Live demo")
    st.markdown("Pick a dataset → the detector runs → pick an alert → watch the agent investigate step by step "
                "(this step list is the trace of every LLM decision and tool call).")
    ds = dataset_picker("demo")
    with st.spinner("Detector scanning the bill…"):
        alerts = cached_detect(str(ds))
    p = load_params()
    st.markdown(f"**Detector:** {len(alerts)} alerts in `{ds.parent.name}/{ds.name}` "
                f"(score > {p['z']} for {p['k']} days, ≥ ${p['min_usd']}/day)")
    st.dataframe(alerts[["alert_id", "resource", "account", "rule", "start_date", "normal_usd_day", "current_usd_day",
                         "extra_usd_day"]], hide_index=True, width="stretch")
    a_id = st.selectbox("Pick an alert to investigate", alerts.alert_id, format_func=lambda a: f"{a} — "
                        f"{alerts.set_index('alert_id').loc[a, 'resource']} ({alerts.set_index('alert_id').loc[a, 'rule']})")
    saved = REP / "runs" / ("final/" if ds.name in [f"r{r}" for r in TEST] else "") / f"{ds.parent.name}_{ds.name}" / f"{a_id}.json"
    c1, c2 = st.columns(2)
    run_live = c1.button("🔎 Investigate live (LLM on this computer, ~15 s)", type="primary")
    load_saved = c2.button("📄 Show saved investigation", disabled=not saved.exists())
    key = f"{ds.parent.name}_{ds.name}_{a_id}"
    if run_live:
        alert = alerts.set_index("alert_id").loc[a_id].to_dict() | {"alert_id": a_id}
        box = st.status("Agent investigating…", expanded=True)

        def on_step(e):
            with box:
                show_step(e)
        try:
            inc = investigate(alert, ds, on_step=on_step)
            box.update(label=f"Done in {inc['seconds']} s", state="complete", expanded=False)
            st.session_state[f"inc_{key}"] = inc
        except Exception as e:  # noqa: BLE001
            box.update(label="Failed", state="error")
            st.error(f"Could not reach the local model. Is Ollama running with `finops-agent`? ({e})")
    elif load_saved:
        st.session_state[f"inc_{key}"] = load_json(saved)
    inc = st.session_state.get(f"inc_{key}")
    if inc:
        incident_card(inc, key)
        with st.expander("Full trace (every step the agent took)"):
            for e in inc["trace"]:
                show_step(e)


# ================================================================ 6. Model choice
def page_model():
    st.title("🧠 Choosing the AI model")
    st.markdown("""
**Why the choice depends on SQL.** The agent learns facts by querying the 5 tables. If it queries wrongly, it reasons
from wrong facts. So we tested which model writes correct queries on *our* tables. Constraints: free, runs on a
16 GB laptop, no API key.
""")
    st.table(pd.DataFrame([
        ["qwen3:8b", "Alibaba (Qwen team)", "8 B", "newest general model; can 'think' before answering"],
        ["qwen2.5-coder:7b", "Alibaba (Qwen team)", "7 B", "specialised in code and SQL"],
        ["llama3.1:8b", "Meta", "8 B", "widely used model from a different company — a neutral baseline"],
        ["(gemma, Google)", "Google", "—", "also fits — not tested for lack of time (limitation)"],
    ], columns=["Model", "Made by", "Size (parameters)", "Why a candidate"]))
    st.markdown("""
**The test:** 20 questions the agent will need (e.g. *"price per hour of sf-api-asg on 30 June and 1 July?"*), each
with a correct reference answer. A model scores a point if its query returns exactly the same rows. One retry is
allowed after an error message.
""")
    st.subheader("Two ways of answering")
    c1, c2 = st.columns(2)
    c1.markdown("""**Direct** — writes the query straight away.
```sql
SELECT ChargePeriodStart, ContractedUnitPrice
FROM billing WHERE ResourceName='sf-api-asg' AND ...
```""")
    c2.markdown("""**Step-by-step** — first reasons, then writes the query.
```text
Need: price per unit → billing.ContractedUnitPrice
Filter: resource name + two dates (the DAY column)
```
```sql
SELECT ...
```""")
    st.caption("The plan is written in the same answer, before the query. A model writes word by word and sees everything it "
               "already wrote — so the plan works like notes on scratch paper while it writes the SQL. qwen3 has a built-in "
               "'thinking' mode for this; for the others we asked for the plan in the instructions.")
    r1, r2 = load_json(REP / "eval/llm_sql_benchmark_round1.json"), load_json(REP / "eval/llm_sql_benchmark_round2.json")
    rename = {"model": "Model", "valid_%": "SQL runs without error", "first_try_%": "Correct 1st try",
              "with_retry_%": "Correct after retry", "median_s": "Seconds per question"}

    def table(r):
        df = pd.DataFrame(r)[list(rename)].rename(columns=rename)
        for c in ["SQL runs without error", "Correct 1st try", "Correct after retry"]:
            df[c] = df[c].astype(str) + "%"
        df["Model"] = df["Model"].str.replace("@direct", " · direct").str.replace("@cot", " · step-by-step") \
            .str.replace("@think", " · built-in thinking")
        return df
    if r1:
        st.subheader("Round 1 — the model saw only table and column names")
        st.dataframe(table(r1), hide_index=True)
        st.caption("Most mistakes: using the invoice-MONTH date column instead of the usage-DAY column — nothing told "
                   "the models which is which.")
    if r2:
        st.subheader("Round 2 — plus a one-line meaning of each important column (data dictionary)")
        c1, c2 = st.columns(2)
        c1.markdown("**Round 1 — what the model saw**")
        c1.code("billing(\n  ChargePeriodStart  TIMESTAMP,\n  BillingPeriodStart TIMESTAMP,\n  EffectiveCost      DOUBLE,\n  ...)",
                language="text")
        c2.markdown("**Round 2 — what was added**")
        c2.code("ChargePeriodStart: the DAY of usage.\n  Use this for any question about dates.\n"
                "BillingPeriodStart: first day of the invoice MONTH.\n  Do NOT use for daily questions.\n"
                "EffectiveCost: cost in USD of that resource\n  on that day. 'cost' always means EffectiveCost.", language="text")
        st.caption("Two date columns with similar names — without a description, most models picked the wrong one.")
        with st.expander("Full data dictionary (one line per important column)"):
            st.code((CFG / "data_dictionary.md").read_text(), language="text")
        st.dataframe(table(r2), hide_index=True)
        st.success("**Chosen: qwen2.5-coder:7b, direct** — tied for the best accuracy (80%) and the fastest (2 s vs "
                   "21 s per answer). Lessons: **describing the data doubled accuracy**; step-by-step reasoning did not help here.")
        with st.expander("Which question each model got right (✓)"):
            qs = [d["q"] for d in r2[0]["details"]]
            st.dataframe(pd.DataFrame({m["model"]: ["✓" if d["ok"] else "✗" for d in m["details"]] for m in r2},
                                      index=[f"Q{i + 1}: {q[:60]}" for i, q in enumerate(qs)]), width="stretch")

    st.subheader("Try it yourself")
    from bench_llm_sql import DICTIONARY, MODE_RULE, SYSTEM, connect, extract_sql, rows_key, schema_text
    from llm import chat
    bench = yaml.safe_load((CFG / "sql_benchmark.yaml").read_text())
    ds = dataset_picker("bench")
    con = connect(ds)
    choice = st.selectbox("Question", ["✍️ my own question"] + [q["q"] for q in bench["questions"]])
    q = st.text_input("Your question") if choice.startswith("✍️") else choice
    gold = next((x["gold"] for x in bench["questions"] if x["q"] == q), None)
    variants = ["qwen2.5-coder:7b@direct", "qwen3:8b@direct", "llama3.1:8b@direct", "qwen3:8b@think"]
    picked = st.multiselect("Models", variants, default=variants[:1])
    if st.button("Run", type="primary") and q and picked:
        system = SYSTEM.format(schema=schema_text(con), dictionary=DICTIONARY, mode_rule=MODE_RULE["direct"])
        gold_rows = con.execute(gold).df() if gold else None
        for col, v in zip(st.columns(len(picked)), picked):
            model, mode = v.split("@")
            with col:
                st.markdown(f"**{v}**")
                try:
                    text, secs = chat(model, [{"role": "system", "content": system}, {"role": "user", "content": q}],
                                      think=(mode == "think"))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Model not available: {e}")
                    continue
                sql = extract_sql(text)
                st.code(sql, language="sql")
                try:
                    got = con.execute(sql).df()
                    st.dataframe(got.head(20), hide_index=True)
                    if gold is not None:
                        ok = rows_key(got.itertuples(index=False)) == rows_key(gold_rows.itertuples(index=False))
                        st.markdown(("✅ correct" if ok else "❌ wrong rows") + f" · {secs:.1f} s")
                except Exception as e:  # noqa: BLE001
                    st.error(f"SQL error: {e}")
        if gold is not None:
            st.markdown("**Reference answer**")
            st.code(gold, language="sql")
            st.dataframe(gold_rows.head(20), hide_index=True)


# ================================================================ 7. Results
def page_eval():
    st.title("📊 Results — how far can it be trusted?")
    st.markdown("Each part is measured on its own. Practice data was used to build and tune; the sealed final exam "
                "was run once at the end.")
    st.subheader("① Detector (practice data)")
    d = load_json(REP / "eval/detector_dev.json")
    if d:
        df = pd.DataFrame(d["by_difficulty"]).rename(columns={"difficulty": "Difficulty", "anomalies": "Anomalies",
                                                              "caught": "Caught", "alerts": "Alerts",
                                                              "false_alarms": "False alarms", "catch_rate_%": "Catch rate %",
                                                              "precision_%": "Alerts that were real %"})
        st.dataframe(df, hide_index=True)
        st.caption("Missed: only the hardest data-transfer spikes/drifts ($0.3–1.3/day inside ±30% noise). False alarms: "
                   "new migration servers without a ticket — the agent dismisses them.")
    st.subheader("② Model choice")
    st.markdown("qwen2.5-coder:7b — 80% of SQL questions correct, 2 s per answer (details on the model page).")
    st.subheader("③ Agent + gate — three versions on practice data")
    st.markdown("**All versions use the same detector.** Only the AI agent changed — how it investigates each alert:")
    st.code("data → detector (same in v1, v2, v3) → alerts → AI agent (this is what changed) → report", language="text")
    rows = []
    for tag in ["v1", "v2", "v3"]:
        r = load_json(REP / f"eval/agent_dev_{tag}.json")
        for ds, v in (r or {}).items():
            s = v["summary"]
            rows.append({"Version": tag, "Dataset": Path(ds).name, "Right type %": s["type_acc_%"],
                         "Right owner %": s["owner_acc_%"], "Right action %": s["action_acc_%"],
                         "Fully right %": s["end_to_end_%"], "Decoys dismissed": s["decoys_dismissed"],
                         "Escalated": s["escalated"], "Risky w/o approval": s["risky_actions_without_approval"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("What the columns mean"):
            st.markdown("""
- **Right type / owner / action %** — of the alerts that were real anomalies, how often the agent named the right kind,
  the right team, the right fix (counted per **alert**).
- **Fully right %** — of all anomalies planted in the dataset, how many were caught by the detector **and** got type +
  owner + action right (counted per **anomaly**; an expired Savings Plan raises 3 alerts but is 1 anomaly — one fully
  correct alert is enough. That is why it can be higher than "right type").
- **Decoys dismissed** — of the planned changes (decoys), how many the agent correctly called "planned, no action".
  The totals differ because each dataset has a different number of planted items (by design).
- **Escalated** — alerts sent to a human by the gate (confidence below 0.6, or type "other").
- **Risky w/o approval** — risky actions (stop, delete, renew) executed without a human. Must always be 0.
""")
        st.markdown("""
- **v1** — the AI chooses its checks freely → 67% fully right. Mistakes: never checked GPU usage, didn't recognise data transfer.
- **v2** — code forced it through a checklist → **worse (33%)**: more evidence confused the small model.
- **v3** — the AI chooses again, but tools state changes plainly and the alert says what kind of resource it is → best on average. Frozen.
- **Trust finding:** wrong answers often came with confidence 0.9–1.0 → the model's confidence alone cannot be trusted; the code gate is essential.
""")
    st.subheader("④ Final exam (sealed data, run once)")
    final = load_json(REP / "eval/final_test.json")
    if final:
        st.markdown(f"Run at **{final['run_at']}** · model `{final['model']}`")
        st.markdown("**The final exam in four questions** (19 anomalies and 10 decoys hidden in each difficulty):")
        ex = pd.DataFrame(final["detector"])
        rws = pd.DataFrame([dict(r, difficulty=a["difficulty"]) for a in final["agent"] for r in a["rows"]])
        risky = {"stop_instance", "recommend_sp_renewal", "snapshot_then_delete_volume"}
        simple = {}
        for dname in ["easy", "medium", "hard"]:
            x, r = ex[ex.difficulty == dname], rws[rws.difficulty == dname]
            n = int(x.anomalies.sum())
            an = r[r.kind == "anomaly"]
            harmless = r[r.kind != "anomaly"]
            ignored = int(((harmless.pred_action == "no_action") & (harmless.status != "escalated_to_human")).sum())
            simple[dname.capitalize()] = [
                f"{int(x.caught.sum())} of {n}",
                f"{an[an.all_ok.astype(bool)].case.nunique()} of {n}",
                f"{ignored} of {len(harmless)}",
                str(int(((r.pred_action.isin(risky)) & (r.status == "auto_executed (simulated)")).sum())),
            ]
        st.table(pd.DataFrame(simple, index=[
            "1 · Did it FIND the anomaly?",
            "2 · Did it fully UNDERSTAND it? (right cause + team + fix)",
            "3 · Did it leave HARMLESS changes alone? (planned changes + legitimate new servers)",
            "4 · Risky actions done WITHOUT a human",
        ]))
        st.markdown("""
- **Finds** almost everything on easy/medium data; on hard data it misses tiny changes ($0.3–1.3 a day hidden in noise).
- **Understands** about 2 of 3 anomalies completely; the rest are partly right or sent to a human.
- **False alarms:** it ignored *all* planned changes, but sometimes calls a harmless new server a problem.
- **Safety: 0** — nothing risky ever happened without a human's approval.
""")
        with st.expander("Detailed tables (per difficulty, TP / FP / FN / TN, precision, recall, F1)"):
            order = {"easy": 0, "medium": 1, "hard": 2}
            dd = pd.DataFrame(final["detector_by_difficulty"]).sort_values("difficulty", key=lambda x: x.map(order))
            dd = dd.rename(columns={"difficulty": "Difficulty", "anomalies": "Anomalies planted", "caught": "Caught",
                                    "alerts": "Alerts raised", "false_alarms": "False alarms", "catch_rate_%": "Catch rate %",
                                    "precision_%": "Alerts on a planted item %"})
            st.markdown("**Detector**")
            st.dataframe(dd, hide_index=True)
            st.caption("Why more alerts than anomalies (easy): 40 alerts = 22 on real anomalies + 10 on decoys (planned "
                       "increases — real cost changes, the agent must dismiss them) + 8 false alarms. 22 alerts for 18 caught "
                       "anomalies because one anomaly can raise several alerts (an expired Savings Plan hits 3 server groups).")
            ad = pd.DataFrame(final["agent_by_difficulty"]).rename(columns={
                "difficulty": "Difficulty", "anomaly_alerts": "Alerts on anomalies", "type_acc_%": "Right type %",
                "owner_acc_%": "Right owner %", "action_acc_%": "Right action %", "tier_acc_%": "Right approval tier %",
                "decoys_dismissed": "Decoys dismissed", "false_alarms_dismissed": "False alarms dismissed",
                "end_to_end_%": "Fully right %", "escalated": "Escalated", "risky_without_approval": "Risky w/o approval"})
            st.markdown("**Agent + gate**")
            st.dataframe(ad, hide_index=True, width="stretch")

            st.markdown("**The same results as TP / FP / FN / TN**")
            det = pd.DataFrame(final["detector"])
            rows = pd.DataFrame([dict(r, difficulty=a["difficulty"]) for a in final["agent"] for r in a["rows"]])
            rows["flagged"] = (rows.pred_action != "no_action") | (rows.status == "escalated_to_human")
            rows["truth"] = rows.kind == "anomaly"
            d_rows, a_rows = [], []
            for dname in ["easy", "medium", "hard"]:
                x = det[det.difficulty == dname]
                tp, fn, fp = int(x.caught.sum()), int((x.anomalies - x.caught).sum()), int(x.false_alarms.sum())
                d_rows.append({"Difficulty": dname, "TP (caught)": tp, "FN (missed)": fn, "FP (false alarms)": fp,
                               "Recall": round(tp / (tp + fn), 2)})
                r = rows[rows.difficulty == dname]
                atp, afp = int((r.truth & r.flagged).sum()), int((~r.truth & r.flagged).sum())
                afn, atn = int((r.truth & ~r.flagged).sum()), int((~r.truth & ~r.flagged).sum())
                prec, rec = atp / (atp + afp), atp / (atp + afn)
                a_rows.append({"Difficulty": dname, "TP": atp, "FP": afp, "FN": afn, "TN": atn, "Precision": round(prec, 2),
                               "Recall": round(rec, 2), "F1": round(2 * prec * rec / (prec + rec), 2)})
            c1, c2 = st.columns([2, 3])
            c1.markdown("*Detector — \"is there an anomaly?\" (per planted anomaly)*")
            c1.dataframe(pd.DataFrame(d_rows), hide_index=True)
            c1.caption("No TN: it would be every normal resource on every normal day — thousands, meaningless.")
            c2.markdown("*Agent — \"is this alert a real problem?\" (per alert; problem = proposes an action or escalates)*")
            c2.dataframe(pd.DataFrame(a_rows), hide_index=True)
            c2.caption("Units differ on purpose: the detector table counts ANOMALIES (18 + 1 = 19), the agent table counts the "
                       "ALERTS it received (easy: TP 19 + FN 3 = 22 alerts on anomalies; FP 5 + TN 13 = 18 = 10 decoys + 8 false "
                       "alarms; 22 + 18 = 40). TN include all 30 decoys. Every FP comes from a detector false alarm (a legitimate new server without "
                       "a ticket) — the agent's main weakness. Type / owner / action correctness is in 'Fully right %' above.")
        with st.expander("Proof nothing was tuned on the exam data"):
            st.markdown("\n".join(f"- {m}" for m in final["audit"]))
        with st.expander("Fingerprint of the frozen code (SHA-256)"):
            st.table(pd.DataFrame(final["fingerprint"].items(), columns=["file", "sha256"]))
        rep = REP / "eval/reproducibility.log"
        if rep.exists():
            with st.expander("Reproducibility: re-running gives the same answer?"):
                st.code(rep.read_text(), language="text")
    else:
        st.info("The final exam is running / not run yet.")


# ================================================================ 8. Built with
def page_built():
    st.title("🛠️ How it was built with an AI coding assistant")
    st.markdown("""
The code was written with **Claude Code**. It was steered by files it reads automatically, not only by chat:

| File | When the assistant reads it | What it controls |
|---|---|---|
| `CLAUDE.md` | at the start of every session | the rules: CRISP-DM order, stop for approval after each step, every number needs a source, prices only from real data, the agent never sees the answer key, code style |
| `.claude/skills/add-data-step/SKILL.md` | when adding a data step | a 10-step recipe: justify numbers → script → checks → chart → update the report → stop |
| `.claude/skills/evaluate-component/SKILL.md` | when evaluating | metric vs target, error analysis, judge calibration |
| `tests/test_no_label_leak.py` | before every commit | automatic proof the agent can't see the answers |
| `docs/CRISP_DM.md` | always | the project log, which is also the report |

**Where the rules came from** — each one fixed a real problem during the project:

| Rule | What happened |
|---|---|
| Every number needs a stated source | the assistant guessed traffic amounts; calibration showed they were 4× too high |
| No unsourced statistics | it once quoted an unsourced "80% of the bill" claim — retracted |
| Stop for approval after each step | I wanted to control every step |
| Every approved decision goes into the report immediately | I had to remind it once |
| Vary the number of anomalies + leak tests | I asked whether the agent could "learn" how many answers there are |
""")
    st.subheader("How the assistant's context was controlled")
    st.markdown("An AI coding assistant only knows what is in its context window. We decided what goes in, when, and what stays out:")
    st.table(pd.DataFrame([
        ["Always in context", "`CLAUDE.md` — short project rules (~1 page), loaded automatically every session"],
        ["Only when needed", "skills (`add-data-step`, `evaluate-component`) — loaded only for that kind of task, so the context stays small"],
        ["Remembered between sessions", "memory notes (e.g. \"user prefers concise answers\", project progress) — so a new session doesn't start from zero"],
        ["Single source of truth", "`docs/CRISP_DM.md` — read before work, updated after every step; nothing agreed lives only in chat"],
        ["Kept OUT on purpose", "the answer key (`data/labels/`) — enforced by tests; large raw files — scripts summarise them instead"],
        ["Delegated", "the first Business Understanding draft was written by a separate sub-agent with its own brief, while we continued with data"],
    ], columns=["Context", "How"]))
    st.subheader("How its behaviour was controlled")
    st.table(pd.DataFrame([
        ["Process", "CRISP-DM phase by phase; after every step it stops and waits for my approval"],
        ["Honesty", "every number needs a source (real / calibrated / design choice / tested); no unsourced statistics"],
        ["Repeatable recipes", "skills make it follow the same steps every time (justify → build → check → chart → document)"],
        ["Automatic guards", "tests fail if agent code reads the answers or if dataset names leak into its instructions"],
        ["Communication", "concise, plain language, one recommendation instead of many options"],
    ], columns=["Behaviour", "Rule"]))
    st.markdown("**Full record:** every message, tool call and result of the coding session is saved by Claude Code as a "
                "transcript file (`~/.claude/projects/<project>/<session>.jsonl`, ~10 MB).")
    with st.expander("CLAUDE.md (full text)"):
        st.markdown((ROOT / "CLAUDE.md").read_text())


pages = [st.Page(page_problem, title="1 · Problem & scope", icon="💸", url_path="problem"),
         st.Page(page_data, title="2 · Data", icon="📂", url_path="data"),
         st.Page(page_synthetic, title="3 · Building the company", icon="🏭", url_path="synthetic"),
         st.Page(page_system, title="4 · How it works", icon="🧩", url_path="system"),
         st.Page(page_demo, title="5 · Live demo", icon="▶️", url_path="demo"),
         st.Page(page_model, title="6 · Choosing the model", icon="🧠", url_path="model"),
         st.Page(page_eval, title="7 · Results", icon="📊", url_path="evaluation"),
         st.Page(page_built, title="8 · How it was built", icon="🛠️", url_path="built")]
st.navigation(pages).run()
