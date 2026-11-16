# FinOps Anomaly Agent — rules for the coding agent

An AI agent that finds cloud cost anomalies in FOCUS billing data, explains them, and fixes the safe ones — and we measure how far it can be trusted. University assignment (UCU, GenAI): every component must be evaluated; CRISP-DM must be visible in the report.

**Single source of truth:** `docs/CRISP_DM.md`. Read it before starting work; update it after every step.

## How we work (CRISP-DM)
- Follow the phases in order: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.
- **One step at a time.** Finish a step, show the result, then **stop and wait for the user's approval** before the next one.
- After each step update the status table and findings in `docs/CRISP_DM.md` (short tables, no long prose).
- **Every design decision the user approves goes into `docs/CRISP_DM.md` immediately** — with the formula, a worked example, and why. The doc is the report; nothing agreed may live only in chat.
- Before building anything new, check existing data/tools/sources first (assignment rule: real data before synthetic).

## Data rules (non-negotiable)
- **Prices come only from real rows** in `data/raw/` (via `data/config/catalog.csv`). Never invent a price.
- **Every number needs a stated source**, one of: (1) real/measured, (2) calibrated to real data, (3) stated design choice that doesn't affect results, (4) tested at several levels. Write it in the doc. No silent assumptions.
- Never edit cost directly: change quantity or unit price, then `cost = quantity × unit price`.
- `data/raw/` is read-only. `data/config/world_calibrated.yaml` is generated — edit `world.yaml` instead.
- `data/labels/` is the answer key: **the agent code must never read it**; only evaluation code may. Enforced by `tests/test_no_label_leak.py`.
- Never hard-code or prompt the number of planted items — it varies per dataset on purpose.
- Replicas 1–2 = dev (look at failures, improve). Replicas 3–5 = test: run only for final scores.
- Fixed random seed; same config → same data.

## Agent design rules (Modeling phase)
- All numbers in reports are computed by SQL/code — the LLM only phrases them.
- Every claim cites the data rows behind it; every finding has a confidence score; low confidence → escalate to a human.
- Destructive actions (stop/delete/change commitment) always need human approval. This gate is plain code, outside the LLM. All actions are simulated.

## Evaluation rules
- Every component gets its own metric and target (see `docs/CRISP_DM.md` §1). Use the `evaluate-component` skill.
- Report failures honestly, with examples and the reason. Never tune on the test set without saying so.

## Layout
```
data/raw/          real FOCUS sample (FinOps Foundation, CC BY 4.0) — read-only
data/config/       catalog.csv, world.yaml, world_calibrated.yaml, anomalies.yaml
data/synthetic/    generated tables (parquet)
data/labels/       ground truth — evaluation only
src/data_gen/      data generation scripts, run in order (see Commands)
src/agent/         the anomaly agent (later)
src/eval/          evaluation scripts (later)
reports/figures/   charts
docs/CRISP_DM.md   project log + report
.claude/skills/    procedures for repeated tasks
```

## Commands
```
uv run python src/data_gen/build_catalog.py      # Step 1: real prices
uv run python src/data_gen/calibrate_world.py    # Step 2b: match real spending mix
uv run python src/data_gen/generate_billing.py   # Step 3: shared state (resource_days) + billing
uv run python src/data_gen/measure_cpu_utilization.py  # Step 4 research: real CPU levels (Azure trace)
uv run python src/data_gen/generate_side_tables.py     # Step 4: metrics, change log, owners, commitments + checks
uv run python src/data_gen/build_datasets.py           # Step 5: 15 datasets with planted items + answer key
uv run python src/data_gen/validate_datasets.py        # Step 6: realism + waste-size checks
uv run python src/agent/detector.py data/synthetic/medium/r1   # Modeling: run the detector on one dataset
uv run python src/eval/eval_detector.py                # tune detector on dev r1–2 + dev report
uv run pytest -q                                       # guards (e.g. agent never reads data/labels) — run before every commit
uv run python src/data_gen/preview_world.py      # quick cost preview of world.yaml
```

## Code style
- Python ≥ 3.11, run with `uv`. DuckDB for SQL, pandas for tables, matplotlib for charts.
- Each script starts with a docstring: purpose, formula, inputs, outputs, how to run.
- Paths via `ROOT = Path(__file__).resolve().parents[2]`. No hard-coded absolute paths.
- Tunable numbers live in config files, not in code.
- Small functions; fail loudly (raise) instead of guessing.

## Talking to the user
- The user is a FinOps analyst, new to DevOps tooling. Be concise and plain; explain new terms in one line.
- No unsourced statistics. Cite sources. Say clearly what is real and what is assumed.
- Give one recommendation, not a list of options.
