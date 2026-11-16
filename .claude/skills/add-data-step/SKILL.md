---
name: add-data-step
description: Use when adding or changing a step of the synthetic data pipeline (src/data_gen/) — new table, new anomaly type, new config. Ensures every number is justified, data is checked, charted, and the CRISP-DM log is updated.
---

# Add a synthetic data step

Follow in order. Do not skip the approval stop.

1. **Check the plan.** Read the Data Preparation table in `docs/CRISP_DM.md`. Confirm which step this is and that the previous one is approved.
2. **Justify every new number before coding.** For each new parameter write one line: value + source type:
   - real/measured (from `data/raw/`) → show the query result
   - calibrated to real data → show target vs achieved
   - design choice → say why it doesn't affect results
   - tested setting → list the levels (easy / medium / hard)
   If a number has none of these, search for a public source first; if none exists, ask the user.
3. **Write the script** in `src/data_gen/` with a docstring: purpose, formula, inputs, outputs, run command. Parameters go in `data/config/`, not in code. Fixed seed.
4. **Respect the data rules:** cost = quantity × unit price; never touch `data/raw/`; answer keys only in `data/labels/`.
5. **Run it** with `uv run python src/data_gen/<script>.py`.
6. **Sanity checks** (print pass/fail): row counts, no missing days, no negative quantities, cost = quantity × price, links between tables hold (e.g. metrics exist only on billed days).
7. **Chart** the result into `reports/figures/stepN_*.png`, open it, and check that it looks plausible. Verify 2–3 values by hand (quantity × price).
8. **Update `docs/CRISP_DM.md`:** status table, new numbers in "Where every number comes from", findings. Short tables only.
9. **Add the command** to the Commands block in `CLAUDE.md` if it's a new script.
10. **Stop.** Show the user: what was built, key numbers, chart paths, open questions. Wait for approval.
