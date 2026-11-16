"""Model selection — text-to-SQL benchmark for candidate local LLMs.

Metric: EXECUTION ACCURACY (standard for text-to-SQL): the model's query is correct if it returns the
same rows as the reference (gold) query. Rows are compared as multisets; numbers rounded to 2 decimals,
dates to YYYY-MM-DD; column order and row order ignored.
Each question gets 1 retry with the database error message (the agent will do the same).
Reported: valid SQL %, accuracy on 1st try, accuracy with retry, median seconds per question.

Variants are written MODEL@MODE:
  @direct → answer directly (qwen3: thinking switched off)
  @think  → built-in thinking (only models that have it, e.g. qwen3)
  @cot    → chain-of-thought PROMPT: "reason step by step, then write the SQL" (works for any model)
Round 1 (no data dictionary, broken thinking switch) is kept in reports/eval/llm_sql_benchmark_round1.json.
Round 2 adds data/config/data_dictionary.md to the prompt.

Inputs : data/config/sql_benchmark.yaml, dataset named there (dev replica)
Output : reports/eval/llm_sql_benchmark_round2.json + printed table

Run:  uv run python -u src/eval/bench_llm_sql.py qwen3:8b@direct qwen3:8b@think qwen2.5-coder:7b@direct ...
"""

import json
import re
import statistics
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agent"))
from llm import chat  # noqa: E402

TABLES = ["billing", "usage_metrics", "change_log", "owners", "commitments"]
DICTIONARY = (ROOT / "data/config/data_dictionary.md").read_text()
SYSTEM = """You write ONE DuckDB SQL query that answers the user's question.
Tables:
{schema}

{dictionary}
Rules: date columns are TIMESTAMPs (compare with DATE 'YYYY-MM-DD').
{mode_rule}"""
MODE_RULE = {
    "direct": "Reply with only the SQL inside ```sql ... ```.",
    "think": "Reply with only the SQL inside ```sql ... ```.",
    "cot": "First reason step by step in a few short lines: which table(s), columns, filters and grouping "
           "you need. Then give the final SQL inside ```sql ... ```.",
}


def connect(dataset: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM '{dataset / (t + '.parquet')}'")
    return con


def schema_text(con) -> str:
    lines = []
    for t in TABLES:
        cols = ", ".join(f"{c} {ty}" for c, ty, *_ in con.execute(f"DESCRIBE {t}").fetchall())
        lines.append(f"- {t}({cols})")
    return "\n".join(lines)


def norm(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, (datetime, date)):
        return v.strftime("%Y-%m-%d") if not (isinstance(v, datetime) and (v.hour or v.minute)) else v.isoformat()
    if hasattr(v, "__float__") and not isinstance(v, (int, bool)):
        return round(float(v), 2)
    return v


def rows_key(rows) -> Counter:
    return Counter(tuple(sorted(map(str, map(norm, r)))) for r in rows)


def extract_sql(text: str) -> str:
    blocks = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    return (blocks[-1] if blocks else text).strip().rstrip(";")  # last block = final answer


def run_model(variant: str, con, schema: str, questions: list) -> dict:
    model, mode = variant.split("@")
    think = {"direct": False, "think": True, "cot": False}[mode]
    system = SYSTEM.format(schema=schema, dictionary=DICTIONARY, mode_rule=MODE_RULE[mode])
    res = []
    for item in questions:
        gold = rows_key(con.execute(item["gold"]).fetchall())
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": item["q"]}]
        first_ok = valid = ok = False
        secs, sql, err = 0.0, "", None
        for attempt in range(2):
            text, s = chat(model, msgs, think=think)
            secs += s
            sql = extract_sql(text)
            try:
                got = rows_key(con.execute(sql).fetchall())
                valid, ok, err = True, got == gold, None
            except Exception as e:  # noqa: BLE001 — any DB error counts as invalid SQL
                valid, ok, err = False, False, str(e).splitlines()[0][:200]
            if attempt == 0:
                first_ok = ok
            if ok or attempt == 1:
                break
            hint = f"That query failed: {err}" if err else "That query returned the wrong rows."
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": hint + " Fix it."}]
        res.append({"q": item["q"], "sql": sql, "valid": valid, "first_ok": first_ok, "ok": ok,
                    "seconds": round(secs, 1), "error": err})
        print(f"  {'✓' if ok else '✗'} {secs:5.1f}s  {item['q'][:70]}", flush=True)
    n = len(res)
    return {"model": variant, "valid_%": round(100 * sum(r["valid"] for r in res) / n),
            "first_try_%": round(100 * sum(r["first_ok"] for r in res) / n),
            "with_retry_%": round(100 * sum(r["ok"] for r in res) / n),
            "median_s": round(statistics.median(r["seconds"] for r in res), 1), "details": res}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "data/config/sql_benchmark.yaml").read_text())
    con = connect(ROOT / cfg["dataset"])
    schema = schema_text(con)
    for q in cfg["questions"]:  # every gold query must run
        con.execute(q["gold"]).fetchall()
    models = sys.argv[1:] or ["qwen3:8b@direct", "qwen2.5-coder:7b@direct", "llama3.1:8b@direct"]
    out = []
    for m in models:
        print(f"\n== {m}", flush=True)
        out.append(run_model(m, con, schema, cfg["questions"]))
    print(f"\n{'model':26} {'valid SQL':>10} {'1st try':>8} {'w/ retry':>9} {'median s':>9}")
    for r in out:
        print(f"{r['model']:26} {r['valid_%']:>9}% {r['first_try_%']:>7}% {r['with_retry_%']:>8}% {r['median_s']:>9}")
    path = ROOT / "reports/eval/llm_sql_benchmark_round2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
