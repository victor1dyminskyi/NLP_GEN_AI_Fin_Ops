"""Guard: the agent must never see the answer key.

1. No code outside src/data_gen (writes labels) and src/eval (scores against labels) may mention them.
2. Agent-visible data (data/synthetic/) must not contain label columns or give-away words in resource names.

Run:  uv run pytest -q
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {ROOT / "src/data_gen", ROOT / "src/eval"}
FORBIDDEN_IN_CODE = ["data/labels", "ground_truth", "labels/"]
FORBIDDEN_COLUMNS = {"label", "expected_action", "action_tier", "impact_usd", "case_id"}
GIVEAWAY_WORDS = ["idle", "orphan", "anomal", "decoy", "untagged", "spike", "drift", "expired"]


def agent_code_files():
    for f in (ROOT / "src").rglob("*.py"):
        if not any(a in f.parents for a in ALLOWED):
            yield f


def test_agent_code_never_reads_labels():
    offenders = [f"{f.relative_to(ROOT)}: '{w}'" for f in agent_code_files()
                 for w in FORBIDDEN_IN_CODE if w in f.read_text()]
    assert not offenders, "Agent code references the answer key:\n" + "\n".join(offenders)


def test_agent_data_has_no_label_columns():
    bad = [f"{f.relative_to(ROOT)}: {sorted(FORBIDDEN_COLUMNS & set(pd.read_parquet(f).columns))}"
           for f in (ROOT / "data/synthetic").rglob("*.parquet")
           if FORBIDDEN_COLUMNS & set(pd.read_parquet(f).columns)]
    assert not bad, "Label columns found in agent-visible data:\n" + "\n".join(bad)


def test_resource_names_do_not_give_away_the_answer():
    bad = set()
    for f in (ROOT / "data/synthetic").rglob("billing.parquet"):
        for name in pd.read_parquet(f, columns=["ResourceName"]).ResourceName.unique():
            if any(w in name.lower() for w in GIVEAWAY_WORDS):
                bad.add(name)
    assert not bad, f"Resource names reveal the answer: {sorted(bad)}"


AGENT_INSTRUCTIONS = [ROOT / "data/config/playbooks.md", ROOT / "data/config/data_dictionary.md",
                      *sorted((ROOT / "src/agent").glob("*.py"))]


def test_agent_instructions_contain_no_dataset_specific_names():
    """The agent's prompts/code must be general: no resource names from any dataset (they would hint answers)."""
    names = set()
    for f in (ROOT / "data/synthetic").rglob("billing.parquet"):
        names |= set(pd.read_parquet(f, columns=["ResourceName"]).ResourceName.unique())
    hits = [f"{f.relative_to(ROOT)}: '{n}'" for f in AGENT_INSTRUCTIONS if f.exists()
            for n in names if n in f.read_text()]
    assert not hits, "Dataset-specific names in agent instructions:\n" + "\n".join(hits)
