# FinOps Anomaly Agent

An AI FinOps assistant that finds unexpected cloud-cost anomalies in a FOCUS-format bill, investigates each one with a
local LLM agent, routes it to the accountable cost owner, and lets a human approve risky actions — with every
component evaluated. UCU GenAI Assignment 1 ("Build an NLP System").

**Report (3–4 pages):** [docs/REPORT.md](docs/REPORT.md) · [PDF](docs/REPORT.pdf) · detailed working log: [docs/CRISP_DM.md](docs/CRISP_DM.md)

## Requirements

| | |
|---|---|
| Computer | Mac with Apple Silicon, 16 GB RAM (tested on M2 Pro 16 GB); the model uses the built-in GPU |
| Disk | ~5 GB for the agent model; +10 GB for the optional comparison models |
| Software | Ollama, Python ≥ 3.11 with uv |
| Cost / accounts | $0, no API key, internet only for the one-time downloads |
| Speed | detector ~1 s per dataset; agent ~15 s per alert |

## Run it (macOS, ~10 minutes, free)

```bash
# 1. Python env (https://docs.astral.sh/uv/)
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 2. Local LLM (no API key, no cost)
brew install ollama && brew services start ollama
ollama pull qwen2.5-coder:7b
ollama create finops-agent -f ollama/Modelfile      # 16k context, temperature 0, seed 42

# 3. App (presentation + live demo)
uv run streamlit run app/app.py                     # → http://localhost:8501
```

Optional for the model-comparison page: `ollama pull qwen3:8b` and `ollama pull llama3.1:8b`.
Optional cloud LLM instead of Ollama: set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in a `.env` (never committed).

## Reproduce the results

All datasets and results are in the repo; everything below regenerates them deterministically (fixed seeds).

```bash
uv run python src/data_gen/build_catalog.py        # real prices from the FOCUS sample
uv run python src/data_gen/calibrate_world.py      # match the real spending mix
uv run python src/data_gen/generate_billing.py     # normal history
uv run python src/data_gen/generate_side_tables.py # usage metrics, change log, owners, commitments
uv run python src/data_gen/build_datasets.py       # 15 datasets + answer key
uv run python src/data_gen/validate_datasets.py    # realism checks
uv run python src/eval/eval_detector.py            # tune detector on dev replicas 1–2
uv run python src/eval/bench_llm_sql.py qwen2.5-coder:7b@direct   # model benchmark (dev)
uv run python src/eval/verify_reproducibility.py reports/runs/final/medium_r3   # same answers as saved?
uv run pytest -q                                   # guards: the agent never sees the answer key
```

`src/eval/final_test.py` produced the official scores **once** on test replicas 3–5 (it refuses to run twice).
Its output `reports/eval/final_test.json` includes an audit (nothing tuned on test data) and a SHA-256 fingerprint
of the frozen agent code, configs and model file.

**Same results on your machine:** the agent model runs with temperature 0 and seed 42; re-running an investigation
on the same machine reproduces it step for step (`verify_reproducibility.py`). Different hardware can still produce
small numeric differences inside the LLM, so exact equality is guaranteed only on the same setup.

## Layout

```
data/raw/          real FOCUS sample (FinOps Foundation, CC BY 4.0), Azure CPU summary
data/config/       prices, company, scenarios, detector settings, data dictionary, playbook, SQL benchmark
data/synthetic/    15 evaluation datasets (easy|medium|hard × r1–r5)
data/labels/       answer key — evaluation code only
src/data_gen/      data pipeline · src/agent/ detector, tools, agent, LLM client · src/eval/ evaluation
app/app.py         Streamlit app (8 pages) · reports/ figures, runs, evaluation results
CLAUDE.md, .claude/skills/   how the AI coding assistant was steered
```
