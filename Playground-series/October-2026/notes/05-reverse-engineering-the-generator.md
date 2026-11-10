# 5. Reverse-engineering the data generator

With a week left the leader was 0.002 ahead of a field stuck near 0.962. In earlier Playground episodes such gaps came
from a flaw in how the data was made, so I tested five ideas (`scripts/generator_checks.py`). Each adds label-free
features, and a small LightGBM tries to correct the stack's predictions with and without them.

| Check | Finding | Gain |
| --- | --- | ---: |
| Near-copies of real rows (18–21 of 21 columns match) | ~17 exact copies; 22% match on 19 columns | −0.000001 |
| Values the generator invented | almost all values exist in the real data | +0.000004 |
| Mixture modes of the numeric columns (as in CTGAN) | clear modes, no label signal | +0.000008 |
| Classifier: synthetic vs real rows | AUC 0.835, unrelated to satisfaction | −0.000002 |
| Model trained on the real data only | already captured by the lookups | +0.000008 |

The test can find signal: on an older, weaker model the real-data check gained +0.00015 on 5/5 folds. Train and test were
indistinguishable (AUC 0.502).

**What it revealed:** a model trained on the real surveys scores AUC 0.994 on held-out real surveys but only 0.954 on the
competition data. The generator scrambled a large share of the labels, which is why everyone converged near 0.962.

**Check:** why is the synthetic-vs-real classifier useless for the label? *Looking fake says nothing about satisfaction
unless the generator's mistakes depend on the label. Here they didn't.*
