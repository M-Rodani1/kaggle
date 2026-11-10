# Playground Series S6E10 — Airline passenger satisfaction

Predict whether a passenger was satisfied from 21 columns (age, flight distance, delays, 13 ratings, 4 categories).
Metric: ROC AUC. 700k training rows, 300k test rows.

**Result: private 0.96239, 38th place.**

The work went one experiment at a time: change one thing, measure it with out-of-fold (OOF) AUC on fixed folds, and keep
it only if it gains at least +0.0002 on at least 4 of 5 folds.

## Results

| Experiment | What changed | OOF AUC | Public | Private |
| --- | --- | ---: | ---: | ---: |
| exp00 | LightGBM + CatBoost baseline | 0.95899 | — | — |
| exp03 | Tuned with Optuna | 0.95957 | 0.95888 | |
| exp05 | Wider LightGBM + XGBoost | 0.96014 | 0.95942 | |
| exp09 | Lookup features from the original dataset | 0.96061 | 0.96019 | |
| exp10 | CatBoost with every column as a category | 0.96102 | 0.96068 | |
| exp11 | Neural network with one embedding per exact value | 0.96115 | 0.96095 | |
| exp13 | More seeds, tuned network | 0.96129 | 0.96100 | |
| exp15c | Route features, RealMLP, 10-fold members | 0.96156 | 0.96132 | |
| exp14 | Logistic stack of public OOF predictions | 0.96222 | 0.96184 | |
| **exp16** | **Stack + my exp15 models (final pick 1)** | **0.96223** | **0.96186** | **0.96239** |
| pick 2 | A public blend tuned on the leaderboard | — | 0.96213 | 0.96222 |

Private scores of the earlier experiments weren't recorded before the competition closed.

## What mattered

1. **The original dataset as a lookup table** (+0.0004). The data was generated from a real survey; its satisfaction
   rates became features. → [notes/01](notes/01-original-data-lookups.md)
2. **CatBoost with every column as a category** (+0.0004). → [notes/02](notes/02-every-column-a-category.md)
3. **Stacking public OOF predictions** (+0.0007). Hundreds of published predictions on the same folds, combined by a
   regularised logistic regression. → [notes/03](notes/03-stacking-public-predictions.md)

## What didn't work

Pseudo-labelling, DART boosting, appending the original rows, XGBoost (weight 0), more public sources in the stack
(+0.000003), and five checks for flaws in the data generator (+0.000008 at best, see
[notes/05](notes/05-reverse-engineering-the-generator.md)).

## Final picks

About 180 teams moved to exactly 0.96213 in the last week with one public blend, tuned against the public leaderboard
with no OOF predictions to check it. I picked my validated stack first and that blend second. On the private board the
blend scored 0.96222 and the stack 0.96239: the leaderboard-tuned file lost its lead, as expected from selecting on
the public score. → [notes/04](notes/04-leaderboard-noise-and-final-picks.md)

## Files

```
airline-baseline.ipynb   the whole pipeline: features, models, blend, refit (each experiment is a CFG switch)
experiments.csv          one row per experiment and model: fold scores, OOF AUC, public score
notes/                   five short concept notes and a longer theory PDF
scripts/
  stack_public.py        the public-pool stack
  build_features.py      route and auxiliary features
  run_realmlp.py, run_lgb_10fold.py, blend_own.py   extra members and my own blend
  generator_checks.py    the five generator checks
  common.py, run_notebook.py                        helpers
```

## Running it

Put `train.csv`, `test.csv`, `sample_submission.csv` in `data/`, and the original Airline Passenger Satisfaction
dataset (train + test combined) in `data/original/original_all.csv`. Open the notebook from this folder; it creates the
fixed folds on its first run. The stack also needs the public OOF datasets listed in `scripts/stack_public.py`,
downloaded into `data/public_oof/`. The data itself is not included.
