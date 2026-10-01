# Playground Series S6E9 — EV purchase prediction

Binary classification: predict `Will_Buy_EV` from 13 survey-style features. Metric: ROC AUC.
The work was done one experiment at a time: change one thing, measure it with out-of-fold (OOF) validation on fixed folds, submit, then move on.

## Results

| Experiment | What changed | OOF AUC (blend) | Public LB | Private LB |
| --- | --- | ---: | ---: | ---: |
| baseline | LightGBM + CatBoost | 0.94187 | 0.94163 | 0.94103 |
| exp02 | In-fold target encoding of 5 numeric columns | 0.94500 | 0.94519 | 0.94423 |
| exp03 | Count encoding | 0.94556 | 0.94573 | 0.94482 |
| exp06 | Digit features | 0.94572 | 0.94588 | 0.94497 |
| exp07 | XGBoost + hill-climbing blend weights | 0.94574 | 0.94589 | 0.94498 |
| exp08 | Neural network (MLP with periodic embeddings) | 0.94575 | 0.94586 | 0.94497 |
| **exp09** | **Wide target encoding (17 keys, two smoothings)** | **0.94614** | **0.94623** | **0.94527** |
| exp10 | Shallower LightGBM settings | 0.94616 | not submitted | not submitted |

Experiments 01, 04, 05 (clip flags, CatBoost numeric categories, original dataset) and the recipe-score features moved the score by 0.0001 or less. They are left in the notebook but switched off, or (for the recipe-score features) only noted here.
Validation tracked the public leaderboard closely (public ≈ OOF + 0.0002), and the private ordering matched.

## What mattered

- **Target encoding done inside each fold** (with an inner split for the training rows) was the biggest gain, +0.0031.
- **Encoding many more keys** (individual digits of income and commute, coarse income bins) was the second, +0.0004 on top of count encoding.
- **More models added little**: XGBoost and a neural network are too similar to the existing trees (rank correlation about 0.996).

## Files

- `ev-baseline.ipynb` — the full pipeline; each experiment is a switch in the `CFG` class. This copy is the executed exp10 run.
- `experiments.csv` — one row per experiment and model: fold AUCs, OOF AUC, change vs. the previous experiment, folds improved, run time, public score.
- `01-…04-*.md` — short notes on the concepts behind each step (out-of-fold validation, target encoding, count encoding, blending).

The competition data is not included. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the competition page and place them beside the notebook.
