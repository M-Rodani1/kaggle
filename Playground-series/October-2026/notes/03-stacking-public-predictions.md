# 3. Stacking other people's predictions

Many competitors published OOF and test predictions made on the same five folds
(`StratifiedKFold(5, shuffle=True, random_state=42)`), so they can be used as features: one column per model.

A **stacker** is a model trained on predictions instead of raw data. Here: a logistic regression on the log-odds of
about 270 members, with strong L2 regularisation (C = 0.003), blended 0.8 / 0.2 with a small LightGBM stacker.
Regularisation matters because members are near-identical (rank correlation above 0.99); without it the weights swing
to large opposite values that fit noise.

Checks before a member goes in:

| Check | Rejects |
| --- | --- |
| Same row order and folds | files that would leak |
| OOF AUC between 0.90 and 0.9625 | broken or suspiciously high files |
| OOF and test on the same scale | members whose test file was made differently |
| Rank correlation below 0.9999 with a member already in | near-duplicates |
| Not itself a stack or blend | stacks of stacks, which inflate CV |

The stack is trained on four folds and scored on the fifth, and its few settings were fixed early rather than re-tuned
after each result. CV and public agreed (exp14: 0.96222 CV, 0.96184 public).

**Check:** why does a 0.958 member still get weight next to 0.962 members? *The stacker rewards new information, not
standalone strength.*
