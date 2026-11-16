---
name: evaluate-component
description: Use when evaluating any component of the anomaly agent (detector, SQL investigator, diagnosis, owner attribution, action selection, autonomy gate, report, escalation). Produces metrics vs targets, error analysis, and a doc update.
---

# Evaluate a component

1. **State the question in plain words**, e.g. "Did the detector catch the planted anomalies?"
2. **Pick the metric** from the targets table in `docs/CRISP_DM.md` §1. Prefer simple names (catch rate, false-alarm rate, accuracy %); add F1 where both catch rate and false alarms matter. If you change a metric, say why.
3. **Ground truth** comes only from `data/labels/`. Evaluation code may read it; agent code never.
4. **Run on every dataset version** (easy / medium / hard). Save raw results to `reports/eval/<component>.json`.
5. **Results table:** metric | target | easy | medium | hard | pass?
6. **Error analysis — required.** Look at the failures one by one (at least 5, or all if fewer). Group them by cause (e.g. "missed: too subtle", "false alarm: Monday jump"). Give one example row per group.
7. **LLM-as-judge (only if used):** label ≥ 20 cases by hand, compare with the judge, report agreement %. If agreement < 80%, fix the judge prompt before trusting its scores.
8. **Explain the score:** why it's not higher, and the 1–2 most promising fixes. Don't tune on the test data without saying so.
9. **Update `docs/CRISP_DM.md` §5 Evaluation** with the table, failure groups, and fixes. Keep it short.
10. **Stop** and show the user the table and the main failure causes.
