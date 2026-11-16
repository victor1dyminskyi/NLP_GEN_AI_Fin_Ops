# FinOps Anomaly Agent — CRISP-DM Log

> **In one sentence:** an AI agent that finds unexpected cloud cost spikes, explains why they happened, and fixes the safe ones — and we measure how far it can be trusted.

| Phase | Status |
|---|---|
| 1. Business Understanding | ✅ done |
| 2. Data Understanding | ✅ done |
| 3. Data Preparation | ✅ done |
| 4. Modeling | 🔄 in progress |
| 5. Evaluation | ✅ done (final test run once) |
| 6. Deployment | ⏳ |

---

## 1. Business Understanding

### Problem
1. **Cost spikes are found late** — often only when the invoice arrives.
2. **Investigating them is manual** — an analyst digs through service → account → resource → owner by hand.
3. **AI tools now exist, but people don't trust them.** Lack of trust was the top-cited barrier (38%) in a July 2026 study of 28 FinOps practitioners (MSc dissertation, BSBI / UCA; summarised by FinOps Weekly, 28 Jul 2026) [5]. No vendor publishes accuracy numbers.

### What exists already

| Product | Does | Doesn't |
|---|---|---|
| AWS Anomaly Detection + Amazon Q (Jun 2026) [1][2] | Finds anomalies, explains root cause | Fix anything; publish accuracy |
| AWS FinOps Agent (preview, Jun 2026) [3][4][6] | Root cause + owner → Jira/Slack | Change any resource; works without tags |
| Finout Agents [7] (early access, 4 Jun 2026) | Detector → Investigator → Orchestrator; reversible fixes automatic, destructive ones need approval | Publish accuracy (illustrative example only) |

**Note:** Finout's design (detect → investigate → fix with human approval for destructive actions) is close to ours — tiered autonomy is not unique. **Our angle:** not to beat these products, but to **measure trust** — at every step, how often is the agent right, and when may it act alone?

### How the system works

```
Billing data → 1. Detect → 2. Investigate (SQL) → 3. Diagnose cause → 4. Act → 5. Report
```

**Trust by design:** numbers come from SQL, never from the LLM · every claim points to data rows · confidence score on every finding · unsure → escalate to a human · risky actions need approval.

| Action type | Example | Who decides |
|---|---|---|
| Low risk | Notify owner, add tag, open ticket | Agent, automatically |
| Destructive | Stop instance, change commitment | Human approval required |

### Scope
- **In:** cost spikes in billing data, root cause, owner, simulated actions, report, evaluation of every step.
- **Out:** real changes to AWS, general cost optimization of the whole account, cloud formats other than FOCUS.

### Success criteria (initial targets, revised in Evaluation)

| Step | Question | Metric | Target |
|---|---|---|---|
| Detect | Did we catch the spikes? | **Catch rate** — % of planted anomalies found | ≥ 80% |
| Detect | Are alerts real? | **False-alarm rate** — % of alerts that are not anomalies | ≤ 20% |
| Detect | Both in one number | **F1** — balance of catch rate and alert precision | ≥ 0.80 |
| Investigate | Does the SQL work and give the right answer? | % queries that run · % correct results | ≥ 95% · ≥ 85% |
| Diagnose | Right cause? | Accuracy % | ≥ 80% |
| Diagnose | Right owner? | Accuracy % | ≥ 85% |
| Act | Right action? | Accuracy % | ≥ 85% |
| Act | Safe? | Destructive actions without approval | **0 (hard rule)** |
| Report | Is it true? | % numbers matching SQL · % claims backed by data | 100% · ≥ 90% |
| Escalate | Asks a human when it should? | % uncertain cases escalated | ≥ 80% |

### Constraints & risks
- No AWS account → all actions simulated. Limited LLM budget. Course deadline.
- **Biggest risk:** planted anomalies too easy → scores look better than reality. *Mitigation:* vary size, duration and subtlety.
- **LLM invents numbers** → *Mitigation:* SQL computes, LLM only words it.

### Glossary
**FinOps** — managing cloud costs across finance + engineering · **FOCUS** — open standard format for cloud billing data · **CUR** — AWS's detailed billing export · **Savings Plan** — discount for committing to spend; prices jump when it expires · **Tag** — label on a resource (e.g. `team=ml`) · **CloudTrail** — AWS log of who changed what, when.

---

## 2. Data Understanding
_Done._

### Datasets checked

| Dataset | What it is | Period | Usable? |
|---|---|---|---|
| [FOCUS Sample Data](https://github.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data) (FinOps Foundation) | Real, anonymized bills in FOCUS format (AWS, Azure, Oracle, GCP); 44 columns; 100k-row file: 94k AWS rows, 50 services, tags like `business_unit`, `application`, `environment` | **1 month** (Sep 2024) | ✅ Real format and realistic patterns — ❌ too short to learn what "normal" looks like |
| [Databend AWS billing sample](https://datasets.databend.com/aws-billing.parquet) | AWS CUR format, 347k rows, 159 columns | **1 month** (Aug 2024) | Backup only (AWS-specific format) |
| Kaggle / Hugging Face | — | — | ❌ No labeled cloud-cost anomaly dataset found |

**FOCUS sample — AWS part, closer look** (file: `data/raw/focus_sample_100000.csv.gz`)
- **1 company** (billing account "SunBird"), **86 AWS accounts**, 29 regions, Sep 2024, **one row = one resource × one hour**.
- The 100k file is a **random sample of rows** from the full month → daily totals are incomplete (~$834/month is not the real bill).
- **Data quality issues:** tags look randomly generated (1,461 business units, 7,648 apps); 33% of rows untagged; `EffectiveCost` = 0 on many on-demand rows; credits make some days negative.
- → Good for **realistic columns, services and prices**, not for realistic **patterns over time**.

**What the sample can and cannot tell us about quantities** (measured, AWS usage rows)

| Question | Finding | Usable? |
|---|---|---|
| Do servers bill full hours? | 94,060 of 94,351 rows are 1-hour rows; 77% of instance rows = exactly 1.0 h | ✅ always-on servers = 24 h/day |
| Spend mix by category | Compute 82.2% · Storage 5.9% · Databases 3.8% · Other 3.0% · Networking 2.9% · Mgmt (logs etc.) 2.0% | ✅ calibration target |
| Inside EC2 | instance hours 78.9% · other 12.5% · disks/snapshots 4.5% · data transfer + NAT 4.1% of total | ✅ calibration target |
| Absolute volumes (GB/day per resource) | median egress per resource-hour ≈ 0 GB; thousands of tiny resources | ❌ too small/sparse to copy |
| Day-to-day noise, weekday pattern | ≤ 42 rows per resource per month; weekday means driven by single outliers | ❌ cannot be measured |
| Growth | only 1 month | ❌ cannot be measured |

**Gap:** no public dataset has (a) several months of history, (b) labeled anomalies, (c) usage metrics, change logs or owners.

**Decision:** use the real FOCUS sample as the **template** (columns, services, tags, price levels), generate 6 months of history from it, and plant labeled anomalies. Usage metrics, change log, owners and commitments are synthetic. Everything synthetic is marked as such.

## 3. Data Preparation
_Done._

### Synthetic data plan (each step approved before the next)

| Step | What | Output | Status |
|---|---|---|---|
| 1 | Copy real prices from the FOCUS sample (fails if a price isn't found) | `data/config/catalog.csv` | ✅ |
| 2 | Fictional company: DemoShop, 6 teams, 8 accounts, 38 resources, ≈ $13.1k/month | `data/config/world.yaml` | ✅ |
| 2b | **Calibrate amounts to the real spending mix** (prices untouched) | `data/config/world_calibrated.yaml` | ✅ |
| 3 | 6 months of normal daily billing: `quantity = base × weekday × growth × noise`, `cost = quantity × real price` — 6,992 rows, ≈ $20.8k/month | `data/synthetic/billing.parquet`, `reports/figures/step3_*.png` | ✅ |
| 4 | Side tables, all derived from one shared state table: usage metrics (2,392 rows), change log (136), owners (6), commitments (1) — 14/14 checks pass | `data/synthetic/{resource_days,usage_metrics,change_log,owners,commitments}.parquet`, `reports/figures/step4_side_tables.png` | ✅ |
| 5 | Plant anomalies + decoys: 15 datasets (3 difficulties × 5 replicas), 7–12 items each → 147 cases (102 anomalies, 45 decoys); all pass checks; anti-leak tests pass | `data/synthetic/{easy,medium,hard}/r{1..5}/`, `data/labels/ground_truth.csv`, `reports/figures/step5_medium_r1.png` | ✅ |
| 6 | Final validation: normal company matches real spending mix (max 1.6 pp off); planted waste 3–17% of spend, below the 29% real-world average (Flexera 2026) | `reports/figures/step6_spending_mix.png` | ✅ |

### Anomaly types to plant (Step 5)

| Type | What changes in the data | Evidence the agent needs | Action tier |
|---|---|---|---|
| Idle GPU | new GPU server billed 24 h/day | usage metrics ≈ 0% | 🔒 approval |
| Data-transfer spike | GB × 3–10 for some days | billing only | ✅ auto (notify) |
| Untagged resource | new resource, empty `Tags` | change log (who created it) | ✅ auto (tag + notify) |
| Expired Savings Plan | unit price ↑, usage same, `Committed` → `Standard` | commitments table | 🔒 approval |
| Orphaned disk | server deleted, its disk (EBS GB) still billed | change log + billing | 🔒 approval |
| Decoy (not an anomaly) | planned increase | change log (approved change) | — should not alert |

### Evaluation datasets (Step 5)

| | |
|---|---|
| Datasets | easy / medium / hard × replicas 1–5 = 15 (`data/synthetic/<difficulty>/r<n>/`) |
| Per dataset | **7–12 items, varying** (replicas: 12 / 7 / 12 / 10 / 8) — nobody can rely on a fixed count. Same items & dates across difficulties within a replica |
| Per difficulty | 34 anomalies + 15 decoys; every type appears in ≥ 2 test replicas; showcase replica 1 has every type |
| Split | replicas 1–2 = **dev** (allowed to look at failures, improve the agent) · 3–5 = **test** (run once at the end) |
| Clean history | first 6 weeks have no planted items (detector needs normal history) |
| Answer key | `data/labels/ground_truth.csv` — 147 rows: type, resources, owner, dates, exact $ impact, expected action + tier |
| No leaks | `tests/test_no_label_leak.py`: agent code may not reference the answer key; agent data has no label columns; resource names contain no give-away words (verified to fail on a planted leak) |
| $ impact | cost(with item) − cost(without item), same random noise → exact |

| Setting (tested) | Easy | Medium | Hard | Source |
|---|---|---|---|---|
| Normal GPU load | 25% | 20% | 6% | ATC'23 / Alibaba PAI / Alibaba ASI |
| Noise | ×0.33 | ×1 | ×2 | tested |
| Transfer spike | ×10 | ×3 | ×1.5 | tested — no public data on anomaly sizes |
| NAT drift | ×3 | ×2 | ×1.3 | tested |
| Untagged server | 2× c5.4xlarge (~$33/day) | same | 1× t3.medium (~$1/day) | tested |

**Average extra cost per day while active** ($; company normal ≈ $683/day)

| Type | Easy | Medium | Hard |
|---|---|---|---|
| Idle GPU | 104 | 103 | 103 |
| Expired Savings Plan | 91 | 91 | 91 |
| Untagged resource | 33 | 33 | 1 |
| Orphaned disk | 5 | 5 | 5 |
| Transfer drift | 8.9 | 4.4 | 1.3 |
| Transfer spike | 5.9 | 1.3 | 0.3 |
| Decoys (training / campaign / launch) | 136 / 10 / 18 | same | same |

**Finding:** because data transfer is only ~3% of the real spending mix, transfer anomalies are small in dollars ($0.4–7/day) even when they are large in percent. The agent should report both "how unusual" and "how much money" — a ×3 spike worth $1.60/day may not be worth anyone's time.

### Final validation (Step 6)

| Check | Result |
|---|---|
| Normal company (no items) vs real spending mix | ✅ every group within 1.6 pp |
| Waste from planted anomalies, share of 6-month spend | ✅ 3.2–16.5% per dataset, below the 29% average self-reported waste (Flexera State of the Cloud 2026) |
| Mix of anomaly datasets | Shifts up to 4.6 pp (GPU share ↑) — expected: an idle GPU *is* extra GPU spend |

**Honest note:** the first version of this check required every dataset to stay within ±3 pp of the real mix; 7 of 15 failed, all because of planted GPU items. Since that shift is the anomaly itself, the check was split into "normal company is realistic" + "anomalies are not unrealistically large".

### Where every number comes from

| Number | Source |
|---|---|
| Prices, units | **Real** — copied from FOCUS sample rows (Step 1) |
| Amounts (servers, GB) | **Calibrated** — solved so each group's share of the bill matches the real sample: `amount = share × total ÷ price` (Step 2b) |
| Split inside a group (e.g. 3 buckets vs 1) | Design choice — does not change the mix |
| Company size (≈ $20.8k/month) | Design choice — detection works on relative change |
| Noise, weekday pattern, growth | Cannot be measured from 1 sampled month → **tested at several levels** (easy / medium / hard datasets, Step 5) |
| Savings Plan discount (30%) | Assumption — real commitment rows have $0 costs |
| Savings Plan term (1 year) | Real product fact — AWS sells 1- or 3-year terms |
| Normal **CPU** load per server fleet | **Real** — log-normal fitted to 237,558 Azure VMs (2019 trace): median 6.4% web/API, 5.7% batch |
| Normal **GPU** load (inference) | **Tested** — easy 25% (ATC'23) / medium 20% (Alibaba PAI) / hard 6% (Alibaba ASI, OSDI'26) |
| Day-to-day load variation (±15%) | **Tested** setting — per-VM averages can't show it |
| Create dates, actors in change log | Design choice — before the observed window, don't affect detection |

**Sync rule.** Billing, usage metrics and change log are all derived from one table, `resource_days` (one row per resource per day). Anomalies change only that table (or `world.yaml`) and everything is regenerated — costs are never patched. Checks verify: metrics exist exactly on billed server-days, `Committed` price exactly inside the Savings Plan term, every resource has one create event before its first bill row.

**Finding:** real servers are mostly idle even when "normal" (median CPU ≈ 6%). An idle GPU (~0%) vs a normal one (6–25%) is therefore a genuinely hard distinction — which is why GPU load is a tested setting.

**Calibration result** (share of bill, %)

| Group | Real | Before | After |
|---|---|---|---|
| Servers — CPU | 63.7 | 30.3 | 64.2 |
| Servers — GPU | 23.0 | 17.9 | 22.8 |
| Disks + snapshots | 4.8 | 5.9 | 4.6 |
| Data transfer | 3.0 | 17.6 | 3.0 |
| Databases | 2.2 | 5.8 | 3.7 |
| NAT / load-balancer hours | 2.1 | 0.7 | 0.5 |
| Logs | 1.1 | 3.9 | 1.1 |
| S3 storage | 0.1 | 17.9 | 0.1 |

Databases and NAT/LB deviate ~1.5 pp because they are whole machines (1 DB, 1 NAT) and cannot be scaled smoothly. Real services we don't model (≈ 9% of real spend: Datadog, SQS, …) are excluded.

**Realism check (Step 6) — spending mix.** We compare each service's **share of total cost** (e.g. EC2 ≈ 71%, RDS ≈ 8% in the real sample) with DemoShop's. Shares are valid even though the sample is incomplete: a random sample of rows keeps the *proportions* roughly right, just as a poll of 1,000 people gives percentages but not the population size. Expect the same ballpark, not an exact match — DemoShop is a different company.

**Assumptions:** noise levels and the 30% Savings Plan discount are design choices, not measured from real data.

## 4. Modeling
_In progress._

### System pipeline (agreed design)

```
dataset (bill + side tables)
  → 1. Detector (code, no LLM) ──alerts──▶ 2. Agent (LLM + tools) ──▶ incident form ──▶ 3. Action gate (code)
                                                                                       ├─ low risk → auto (simulated)
                                                                                       └─ risky → human Approve / Reject (UI)
```

### Component 1 — Detector (approved)

Works **per resource** on daily rows (one row = one resource × one day). Per-resource, because a $0.40/day spike is invisible in an $878/day company total but is ×1.5 for its own resource.

**(a) Unusual cost (usage alerts)** — for resource *r*, *x_t* = cost on day *t*:

```
normal_t = median(x over the previous 28 days)
spread_t = 1.4826 × median(|x − normal_t|) over those days      ("robust standard deviation")
score_t  = (x_t − normal_t) / spread_t
alert if  score_t > Z  for K days in a row  AND  x_t − normal_t ≥ $min
```

| Term | Meaning | Why |
|---|---|---|
| median | middle value of the last 28 days | one extreme day does not distort "normal" |
| 28 days | 4 full weeks | weekend dips are part of normal |
| 1.4826 | fixed constant: for bell-shaped data, standard deviation ≈ 1.4826 × median distance (MAD) | makes the spread read like a standard deviation, but robust to past spikes |
| score | how many "normal wiggles" today is above normal | unit-free → works for $1 and $100 resources |
| Z (e.g. 4) | how unusual counts as unusual | lower → more false alarms; higher → misses small anomalies |
| K (e.g. 2) | days in a row | one random jump is not an alert |
| $min (e.g. $1/day) | minimum **extra cost per day** vs normal, whatever the cause | ignores anomalies worth cents |

*Example:* traffic normally $10/day, typical wiggle $1.20 → spread = 1.4826 × 1.20 = $1.78. Today $23 → score = (23 − 10) / 1.78 = 7.3 > 4 and extra $13 ≥ $1 → unusual; if tomorrow too → alert.

**(b) Price change (rate alerts)** — catches the expired Savings Plan:

```
price_t        = EffectiveCost_t / PricingQuantity_t           ($ per hour or per GB)
normal_price_t = median(price over the previous 28 days)
alert if |price_t / normal_price_t − 1| ≥ 5%  for K days in a row
```

**(c) Usage- vs price-driven** (standard price × volume split; AWS makes the same distinction):

```
usage effect = (quantity_t − quantity_normal) × price_normal
price effect = quantity_t × (price_t − price_normal)
```
*Example (Savings Plan, July 1):* 485 h vs 490 h normal, $0.340 vs $0.238 → usage ≈ −$1, **price ≈ +$49** → price-driven.

**(d) New resource** — no history to compare with:

```
first_seen(r) = first date with cost > 0
alert if first_seen(r) is after the first 28 days of history  AND  average daily cost ≥ $min
```
Decoys (planned training, feature launch) also trigger this — telling planned from unplanned is the **agent's** job (change-log tickets).

**(e) Orphaned disk** — *added during building, approved:* the cost of an orphaned disk does **not change** when its server is deleted, so rules (a)–(d) cannot see it. Rule: a disk whose `AttachVolume` server has a `TerminateInstances` event, still billed K days later with ≥ $min/day → alert from the day after deletion. (Same idea as AWS Trusted Advisor's "unattached volumes" check.)

**Episodes:** alerts on the same resource and family (usage + price = one family; new; orphan) within 14 days are merged into one alert — e.g. an expired Savings Plan triggers both usage and price rules on the same servers.

**Tuning protocol:** grid Z ∈ {3, 4, 5, 6} × K ∈ {1, 2, 3} × $min ∈ {$0.1, $0.25, $0.5, $1, $5} on **dev replicas 1–2 only** (6 datasets); pick the best mean F1 (tie → fewer alerts); **freeze** before test replicas 3–5. **Known weakness:** slow drift — the 28-day "normal" follows it upward.

**Tuned & frozen** (`data/config/detector.yaml`): **Z = 3, K = 2, $min = $0.50/day**. $0.50 is inside the grid (smaller values were tried and scored worse).

### Component 2 — Agent output: a fixed form (approved)

Graded fields use **fixed choices**, so correctness is counted, not judged:

| Field | Allowed values | Compared with answer key |
|---|---|---|
| resource | an existing resource ID | exact |
| type | idle_gpu · transfer_spike · transfer_drift · untagged_resource · sp_expired · orphaned_disk · planned_change · other | exact |
| owner_team | one of the 6 teams | exact |
| action | one of the 7 actions | exact |
| needs_approval | **set by code from the action, not by the LLM** | must be 100% correct |
| start_date | date | correct if within ±3 days |
| extra $/day | number computed by SQL | within ±10% |
| confidence | 0–1 | low → escalate to human |
| explanation | free text | only field judged by an LLM, calibrated on ~20 hand labels (≥ 80% agreement) |

**Matching rule:** an incident matches an answer-key row if it names one of that row's resources and the dates overlap.

### Model choice — text-to-SQL benchmark

**Why a benchmark:** the agent investigates by querying the data in SQL; wrong SQL = wrong facts. Online sources disagreed and none tested our task, so 3 candidates were measured. **Candidates** (shortlist, not exhaustive): fit 16 GB RAM (~8B), free in Ollama, recommended for SQL/tool calling in the sources found, and different from each other — qwen3:8b (newest general model, has built-in thinking), qwen2.5-coder:7b (code/SQL specialist), llama3.1:8b (different vendor, baseline). Gemma was left out (time) — stated as a limitation.

**Test:** 20 questions the agent will need (costs per day, prices, tags, creators, tickets, owners, Savings Plan) on dev dataset `medium/r1`; each has a reference SQL. **Metric: execution accuracy** — the model's SQL must return the same rows as the reference (strict: extra columns count as wrong). One retry with the error message.

**Round 1** (schema only; qwen3's thinking accidentally ON):

| Model | Valid SQL | Correct (with retry) | Sec / question |
|---|---|---|---|
| qwen3:8b (thinking) | 100% | 75% | 66 |
| qwen2.5-coder:7b | 90% | 40% | 3.3 |
| llama3.1:8b | 90% | 40% | 4.3 |

**Error analysis round 1:** most failures used `BillingPeriodStart` (invoice **month**) instead of `ChargePeriodStart` (usage **day**) — the prompt never said which is "the day". Also domain gaps (searching for an event named "create" instead of `RunInstances`). And the "thinking off" text switch did not work (14.5 s vs 1.4 s with `reasoning_effort="none"`).

**Round 2** (+ data dictionary `data/config/data_dictionary.md`; thinking switch fixed; each model direct vs step-by-step):

| Variant | Valid SQL | 1st try | Correct (with retry) | Sec / question |
|---|---|---|---|---|
| **qwen2.5-coder:7b, direct** | 95% | **75%** | **80%** | **2.2** |
| qwen2.5-coder:7b, step-by-step prompt | 95% | 75% | 80% | 5.7 |
| qwen3:8b, built-in thinking | 95% | 75% | 80% | 21.1 |
| qwen3:8b, direct | 95% | 60% | 75% | 3.0 |
| llama3.1:8b, direct | 90% | 60% | 70% | 2.5 |
| llama3.1:8b, step-by-step prompt | 90% | 65% | 70% | 4.8 |

**Findings**
- **Describing the data mattered more than the model:** the data dictionary doubled qwen2.5-coder (40% → 80%) and llama (40% → 70%).
- **Reasoning did not pay off here:** step-by-step prompting changed nothing for coder/llama; qwen3's thinking gained 1 question (75% → 80%) at 7× the time.
- **Hardest questions (failed by ≥ 5 of 6 variants):** "resources that *first appear* after a date" (needs `HAVING MIN(date) > …`, all used `WHERE`), "which team owns account X" (comma-separated list join), "which server was this volume attached to" (`ResourceName` vs `ResourceId`), untagged resources with first date.
- **Caveat:** 20 questions → 1 question = 5 points; 80% vs 75% is not a real difference.

**Decision (approved): qwen2.5-coder:7b, direct** — tied for best accuracy and the fastest (2.2 s vs 21 s for qwen3 thinking). With ~10 LLM calls per alert, that is ~20 s vs ~3.5 min per investigation. **Consequence for the agent design:** the hard question types become ready-made **tools** (e.g. `get_owner(account)`, `first_seen(resource)`, `attached_server(volume)`) instead of free-form SQL. All 3 models stay installed for the live comparison page in the app.

### Component 2 — the agent (built)

**Model:** `finops-agent` = qwen2.5-coder:7b + 16k context, temperature 0 (`ollama/Modelfile`).
**Finding:** Ollama's default context silently cut long prompts — a test message of ~9k tokens was reduced to ~2k and the model lost the key fact. The usual per-request setting was ignored; a project model file fixed it (7k tokens seen, correct answer). The SQL benchmark prompt was 919 tokens, so benchmark results are unaffected.

**Loop (one alert):** alert → LLM chooses a tool (JSON) → code runs it → result back → … → final incident form. Plain JSON protocol (no special tool-calling support needed; works with any model/provider). Max 10 tool calls; at least 2 (incl. change events) before the final answer; invalid forms are sent back with the reason. Every step is saved (trace for the UI).

**Instructions given to the model:** data dictionary + FinOps **playbook** (`data/config/playbooks.md`: signs of each type, matching action, who the owner is, how to rate confidence) + tool list. The playbook is general FinOps knowledge, not answers for a dataset.

**Tools** (read-only on one dataset; the hard benchmark patterns are ready-made):

| Tool | Returns |
|---|---|
| `resource_info` | account, owning team, service, price line, tags (or "untagged"), first/last billed day |
| `cost_history` | daily cost, quantity, price per unit, pricing category around a date + before/after summary |
| `usage` | average CPU % / GPU % before vs after a date |
| `change_events` | who created/deleted it, when, tickets, details |
| `attached_server` | which server a disk was attached to, was it terminated |
| `get_owner` | team, owner, email, Slack for an account |
| `commitment_status` | Savings Plans covering a resource (discount, start, end, managed_by) |
| `run_sql` | free read-only SQL, only if no other tool fits |

**Form:** type ∈ {idle_gpu, transfer_spike, transfer_drift, untagged_resource, sp_expired, orphaned_disk, planned_change, **no_issue**, other} · owner_team · action · start_date · confidence · explanation · evidence. (`no_issue` added after the detector's error analysis: legitimate new resources without a ticket.)

### Component 3 — action gate (code, not the LLM)

| Condition | Status |
|---|---|
| confidence < 0.6 | **escalated_to_human** |
| action = no_action | no_action |
| action ∈ {stop_instance, recommend_sp_renewal, snapshot_then_delete_volume} | **awaiting_approval** (human clicks Approve / Reject) |
| other actions (notify, tag) | auto_executed (simulated) |

### Baseline: "just ask the LLM" (planned, agreed)

To prove the architecture matters, the same datasets are also given to the **LLM alone**: it gets the tables, the data dictionary and one instruction — *"Find the cost anomalies in this dataset"* — and writes its own SQL with no detector. Scored against the same answer key as the full system:

| Approach | What happens |
|---|---|
| A. LLM alone (baseline) | model explores the tables however it wants, lists anomalies |
| B. Our system | detector finds suspects with statistics → agent investigates each with tools |

Answers the examiner question *"why not just ask an LLM?"* with numbers, not claims.

### LLM and UI (plan)

| Item | Decision | Why |
|---|---|---|
| LLM | local open model via **Ollama** on the user's Mac (M2 Pro, 16 GB) | $0, no request limits, no secret key in the repo; graders can run the same model |
| Model choice | mini-benchmark of 2–3 candidates on ~20 SQL questions about our data | justified by score, not memory |
| Fallback | OpenRouter free tier via `.env` (never committed) | free tier is 50 requests/day — too few for 150 cases, fine for a demo |
| UI | Streamlit page: pick dataset → alerts → agent trace (each tool call / SQL) → incident card → Approve / Reject for risky actions → evaluation tab | demo, tracing and human-in-the-loop |
| Deployment | GitHub repo + README (≈ 3 commands) + precomputed results; optional hosted demo later | assignment: deployment optional, must run + video |
## 5. Evaluation
_In progress — dev results only; test replicas 3–5 are run once at the end._

### Component 1 — Detector (dev replicas 1–2, frozen settings Z=3, K=2, $min=$0.50)

**Matching:** an alert matches an answer-key row if its resource is one of the row's resources and it starts within [row start − 3 days, row end]. False alarm = alert matching no row. Alerts on **decoys are expected** (they are real cost increases) — dismissing them is the agent's job.

| Difficulty | Anomalies | Caught | Catch rate | Alerts | False alarms | Precision |
|---|---|---|---|---|---|---|
| Easy | 15 | 15 | **100%** | 27 | 4 | 85% |
| Medium | 15 | 15 | **100%** | 27 | 4 | 85% |
| Hard | 15 | 10 | **67%** | 22 | 4 | 82% |

| Type | Easy | Medium | Hard |
|---|---|---|---|
| Idle GPU | 100% | 100% | 100% |
| Expired Savings Plan | 100% | 100% | 100% |
| Untagged resource | 100% | 100% | 100% |
| Orphaned disk | 100% | 100% | 100% |
| Transfer spike | 100% | 100% | **0%** |
| Transfer drift | 100% | 100% | **0%** |

**Error analysis**
- **Misses (5, all hard):** transfer spikes ×1.5 and drifts ×1.3. They add only $0.30–1.30/day — below $min, and ×1.5 is within ±30% noise. *Why not lower $min?* $0.10–0.25 were tried and scored worse (more false alarms). This is the detection limit: **tiny + noisy = invisible**, and arguably not worth anyone's time.
- **False alarms (12, one cause):** the "new resource" rule flags a migration server and its disk when they are created (~2 weeks before the server is deleted and the disk becomes orphaned). At creation they are legitimate (tagged, known owner) but had no ticket. The detector cannot tell; the **agent** must check tags / owner / change log and answer "no issue".
- **Bug found and fixed during evaluation:** episode merging first glued the disk's "new" alert to its later "orphan" alert and moved the date back 2 weeks → orphaned disks were 0% caught. Merging now only within the same rule family → 100%.
- **Improvement ideas:** a longer fixed reference (e.g. days 29–56 ago) to expose slow drift; per-weekday baselines.

### Component 2 + 3 — Agent & action gate (dev)

**Matching:** each incident (one per detector alert) is matched to the answer key with the detector's rule. Anomaly alerts are scored on type, owner, action and approval tier; decoys and detector false alarms must be **dismissed** (planned_change / no_issue + no_action). **End-to-end** = anomaly caught by the detector AND type + owner + action correct.

**v1 — medium/r1** (16 alerts, ~17 s per investigation):

| Metric | v1 |
|---|---|
| Type correct (anomaly alerts) | 64% |
| Owner correct | 91% |
| Action correct | 64% |
| Decoys dismissed | 3/3 |
| Detector false alarms dismissed | 2/2 |
| End-to-end | 67% |
| Risky actions without approval | **0** |

**Error analysis v1 (from the traces):**
- *Protocol confusion:* the model called the action `no_action` as if it were a tool, 12 times; the generic "unknown tool" message did not help → gave up, escalated.
- *Skipped evidence:* two idle GPUs were judged without ever calling `usage` (GPU % would have shown 0.5%) → "no_issue" / "other".
- *Did not recognise data transfer:* found the evidence (560 → 988 GB/day, price unchanged) but answered "other".
- **Trust finding:** wrong answers came with **high confidence** (1.0 and 0.9) — the model is confidently wrong, exactly the trust problem from Business Understanding. Confidence alone cannot be trusted; code-enforced checks are needed.

**v2 fixes (designed on medium/r1, checked on medium/r2):**
1. Specific error message when an action is used as a tool; the same error 3× in a row stops the loop (escalated).
2. **Required evidence enforced by code** before a final answer: servers → `usage`, disks → `attached_server`, price alerts → `commitment_status`, always `change_events` + `cost_history`.
3. Playbook: how to recognise servers / disks / data-transfer resources.
4. Gate: type "other" is always escalated to a human, whatever confidence the model claims.

**v2 result — worse.** medium/r1 end-to-end 67% → **33%**; medium/r2 67%. Forcing more evidence confused the 7B model: the web servers (~4% CPU, normal for real servers) were called an "idle GPU" with a stop proposal (held by the approval gate); untagged servers called "no issue"; drifts called spikes. **Lesson:** for a small model, more evidence ≠ better answers.

**v3 — the LLM stays in charge** (user requirement: the LLM must do real work, not just rephrase code output). The LLM again chooses what to investigate; code only (1) adds a `resource_kind` hint to the alert (server / gpu_server / disk / data_transfer …, read from the price line), (2) makes tools state changes plainly ("GPU busy 85.9% before → 0.5% after", "price per unit +42.9%, quantity +1.2%", "quantity reached its new level after 7 days", "TICKETS: none"), (3) keeps the safety rules. New LLM output: a ready-to-send **message to the cost owner**.

| Version | medium/r1 end-to-end | medium/r2 end-to-end | mean |
|---|---|---|---|
| v1 | 67% | — (not run) | — |
| v2 | 33% | 67% | 50% |
| **v3** | **78%** | 50% | **64%** |

**Decision: freeze v3** — best mean, idle GPUs fixed (3/3 on r1), remaining errors land mostly in *escalated* or *awaiting approval* (a human catches them). Remaining weakness: data-transfer spike vs drift. **Caveat:** only 19 anomaly alerts on dev → results swing between datasets; v1 was not re-run on r2 (no snapshot of v1 code — lesson: version control from day one).

**Data-leakage checks before freezing:** two hints in my own instructions removed ("often ~40%" price rise = our generator setting; a real resource name used as an example). New automatic test: no dataset resource name may appear in the agent's instructions or code (4 guard tests pass).

**Reproducibility:** fixed seed (42) in `ollama/Modelfile` → the same alert gives an **identical answer, step for step** (3 alerts × 2 runs). **Trust finding:** before the seed was fixed, the same alert (`sf-web-asg`, expired Savings Plan) was once classified `sp_expired` and once `no_issue` — in borderline cases the answer depends on randomness. Dev runs were repeated with the seed so saved results match live runs.

**Audit before the final test** (`src/eval/final_test.py`, enforced by code): detector tuned on replicas 1–2 only; model benchmark on medium/r1 only; agent v1–v3 developed on replicas 1–2 only; the final test refuses to run twice. Results carry a SHA-256 fingerprint of the frozen code, configs and model file.

### Final test — sealed replicas 3–5, run once (frozen v3, seed 42)

Audit passed (detector, model benchmark and agent v1–v3 used replicas 1–2 only). Fingerprint of the frozen code in `reports/eval/final_test.json`.

**Detector**

| Difficulty | Anomalies | Caught | Catch rate | Alerts | False alarms | Alerts that were real |
|---|---|---|---|---|---|---|
| Easy | 19 | 18 | 95% | 40 | 8 | 80% |
| Medium | 19 | 18 | 95% | 40 | 8 | 80% |
| Hard | 19 | 12 | 63% | 34 | 8 | 76% |

**Agent + action gate**

| Difficulty | Right type | Right owner | Right action | Right approval tier | Decoys dismissed | False alarms dismissed | Fully right (end-to-end) | Escalated | Risky w/o approval |
|---|---|---|---|---|---|---|---|---|---|
| Easy | 68% | 82% | 68% | 95% | 10/10 | 3/8 | **74%** | 2 | **0** |
| Medium | 64% | 86% | 73% | 91% | 10/10 | 4/8 | **63%** | 4 | **0** |
| Hard | 69% | 69% | 69% | 81% | 10/10 | 3/8 | **58%** | 2 | **0** |

**Reading the results**
- **Safety held:** 0 risky actions without approval; every decoy (30/30) correctly dismissed as planned.
- **End-to-end 58–74%** — close to the dev estimate (64%), so the practice/test split worked: no sign of over-fitting.
- **Weak spot:** detector false alarms (new resources without tickets) were dismissed only 10/24 times — the agent tends to find a problem where there is none.
- **Reproducibility:** 3/3 re-run investigations identical, step for step.
- **Trust conclusion:** with this local 7B model the assistant is reliable enough to **find, explain and route** anomalies and to run **low-risk** actions (notify, tag) alone; **risky actions must stay behind human approval** — its own confidence does not tell right from wrong.

## 6. Deployment
_Planned (agreed with user)._

**The app is also the presentation:** one Streamlit app; each page is a short "slide" with explanations, so the video can walk examiners through it.

| Page | Content | Interactive | Grading criterion |
|---|---|---|---|
| 1. Problem | late, manual anomaly hunting; trust gap; what AWS already offers; the assistant routes each anomaly to the **accountable cost owner** (team lead / budget owner, not the engineer who caused it) | — | requirements, scope |
| 2. Data | FOCUS sample (FinOps Foundation, real & anonymized, CC BY 4.0): what it could / couldn't tell us; real spending mix; CPU (Azure trace) & GPU (papers) load | — | data |
| 3. Synthetic company | DemoShop; the 3 formulas (calibration, generation, impact); calibration table; anomaly types; easy / medium / hard | — | data preparation |
| 4. System | pipeline diagram: detector → agent → incident form → approval gate | — | system understanding |
| 5. Live demo | pick dataset + model → alerts → agent trace (every question, SQL, result) → incident card → Approve / Reject | ✅ | working prototype |
| 6. Model choice | why these 3 models; saved benchmark report (per-question ✓/✗); **try it live**: pick model + dataset + a benchmark question or your own → SQL → rows → correct answer & ✓/✗; side-by-side for all models | ✅ | tool / model selection |
| 7. Evaluation | metrics per component, easy / medium / hard, error analysis | — | evaluation |
| 8. How it was built | CLAUDE.md, skills, "where each rule came from" | — | coding agent |

**Test-data rule in the app:** all 15 datasets are selectable, but official test scores (replicas 3–5) are computed **once** by the evaluation scripts and saved; live runs are demonstrations and never change reported numbers. Test datasets are labelled as such.

**Formulas shown on page 3:**
```
1. Calibration:    amount = real spending share × total bill ÷ real price
2. Generation:     quantity(day) = base × weekday pattern × growth × noise;  cost(day) = quantity(day) × real unit price
3. Anomaly impact: extra $ = cost(with anomaly) − cost(same day, without anomaly)
```

---

## Appendix — How the coding agent is managed

| File | Loaded | Purpose |
|---|---|---|
| `CLAUDE.md` | every session, automatically | Project rules: CRISP-DM order, data rules, agent design rules, layout, code style |
| `AGENTS.md` | by other AI tools | Points to `CLAUDE.md` |
| `.claude/skills/add-data-step/` | when adding a data step | 10-step procedure: justify numbers → script → checks → chart → doc → stop for approval |
| `.claude/skills/evaluate-component/` | when evaluating a component | Metric vs target, error analysis, LLM-judge calibration, doc update |

**Where the rules came from** (each one fixed a real problem in this project):

| Rule | Origin |
|---|---|
| Every number needs a stated source | Agent first guessed traffic & storage amounts; user rejected it → real spending-mix calibration showed the guesses were 4× too high |
| Prices only from real rows | Keeps all costs traceable to the FOCUS sample |
| No unsourced statistics | Agent quoted an unsourced "80% of the bill" claim early on; retracted |
| Stop for approval after each step | User wants to control the process step by step |
| Concise answers, explain terms | User feedback during Business Understanding |
| Every approved decision goes into this doc immediately | User reminder after approving the detector math |
| Vary the number of planted items; anti-leak test | User asked whether the agent could "learn" the count of 150 answer-key rows |

---

## Sources
1. [AWS — AI-powered cost investigations (What's New, Jun 2026)](https://aws.amazon.com/about-aws/whats-new/2026/06/aws-ai-powered-cost-investigations/)
2. [AWS blog — Introducing AI-powered cost investigations](https://aws.amazon.com/blogs/aws-cloud-financial-management/introducing-ai-powered-cost-investigations-for-cost-anomalies/)
3. [AWS blog — FinOps Agent public preview](https://aws.amazon.com/blogs/aws-cloud-financial-management/aws-finops-agent-is-now-public-preview)
4. [AWS — FinOps Agent FAQs](https://aws.amazon.com/finops-agent/faqs/)
5. [FinOps Weekly — Why trust, not cost, is the biggest barrier to ML in FinOps](https://finopsweekly.com/blog/why-trust-not-cost-is-the-biggest-barrier-to-machine-learning-in-finops/)
6. [InfoQ — AWS FinOps Agent](https://infoq.com/news/2026/06/aws-finops-agent)
7. [Finout — Introducing Finout's Agents](https://www.finout.io/blog/introducing-finouts-agents)
8. [FinOps Foundation — FOCUS Sample Data (CC BY 4.0)](https://github.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data)
9. [Azure Public Dataset V2 — VM trace 2019](https://github.com/Azure/AzurePublicDataset); Cortez et al., [Resource Central, SOSP'17](https://www.microsoft.com/en-us/research/publication/resource-central-understanding-predicting-workloads-improved-resource-management-large-cloud-platforms/)
10. [Alibaba ASI, OSDI'26 — median online-inference GPU utilization ≈ 6%](https://www.usenix.org/system/files/osdi26-li-suyi.pdf)
11. [Acme, NSDI'24 slides — Alibaba PAI median SM activity ≈ 20%](https://usenix.org/system/files/nsdi24_slides-hu.pdf)
12. [Weng et al., FGD, ATC'23 — production GPU utilization 25% to below 50%](https://www.usenix.net/system/files/atc23-weng.pdf)
13. [Flexera — 2026 State of the Cloud: estimated wasted IaaS/PaaS spend 29%](https://www.flexera.com/about-us/press-center/flexera-finds-cloud-value-is-rising-while-ai-waste-grows)
