# Concept 1: Out-of-fold validation

Before we improve the model, we need a fair way to decide whether a change helped. Our first implementation step will be to make experiments use the same validation folds and record their results side by side.

## The problem with training score

A model can score well on rows it has already seen without predicting new rows well. We therefore set aside some labelled rows during training and score the model on those held-out rows.

Our competition measures **ROC AUC**. Informally, AUC asks how often the model ranks a randomly chosen buyer above a randomly chosen non-buyer. Higher is better. The current notebook's out-of-fold AUC is **0.94187**.

## How five-fold validation works

We divide the training rows into five groups, called *folds*. For each run, we train on four folds and predict the fifth:

```text
Run 1: train on folds 2–5       → predict fold 1
Run 2: train on folds 1, 3–5    → predict fold 2
Run 3: train on folds 1–2, 4–5  → predict fold 3
Run 4: train on folds 1–3, 5    → predict fold 4
Run 5: train on folds 1–4       → predict fold 5
```

We put each run's held-out predictions back beside their original rows. At the end, every training row has one prediction from a model that **did not train on that row**. These are *out-of-fold* (OOF) predictions. We calculate one overall AUC from all of them.

The notebook uses **stratified** folds: each fold has approximately the same proportion of buyers and non-buyers. That matters here because buyers are about 17.5% of the training rows.

## Why the folds must stay fixed

Suppose the baseline scores 0.94187 and a new feature scores 0.94220. The difference is **+0.00033 AUC**. If both models used the same folds, we can attribute the difference more confidently to the feature. If we also changed the fold assignment, some of the difference might come from an easier or harder split.

For each experiment we will therefore keep the fold IDs, metric, and other model settings fixed. We will change one feature or setting at a time and record:

| Experiment | Fold AUCs | Overall OOF AUC | Change from baseline | Training time |
| --- | --- | ---: | ---: | ---: |
| Current baseline | Recorded during the run | 0.94187 | — | Recorded during the run |
| One proposed change | Recorded during the run | To be measured | To be measured | To be measured |

Small gains can be noise or a result of repeatedly choosing what looks best on the same folds. Before replacing the baseline, we will check a promising change using a second fold assignment.

## OOF predictions versus test predictions

The **OOF predictions** let us measure model quality because we know the training labels. The **test predictions** are for `submission.csv`; the test labels are hidden, so we cannot calculate their AUC locally. Our notebook averages test predictions from the five trained models.

## What we will code next

We will update the central notebook to store the fixed fold IDs and produce a compact experiment table. We will first run the unchanged baseline through that table. Only then will we try a model improvement.

**Check your understanding:** Can an OOF prediction for a row come from a model trained on that same row? **No.** That separation is the key idea.
