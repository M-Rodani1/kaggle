# Concept 2: Target encoding

Our models treat `Age`, `Annual_Income_USD`, and the charging-station counts as ordinary numbers. A tree splits a number into ranges, such as "income below 61,000". This competition's data is synthetic, and many of these columns have only a few distinct values:

| Column | Distinct values |
| --- | ---: |
| Age | 45 |
| Charging_Stations_Near_Home | 15 |
| Charging_Stations_Near_Work | 20 |
| Daily_Commute_km | 805 |
| Annual_Income_USD | 13,214 |

With so few values, each exact value can behave like a category with its own purchase rate. For example, age 38 may differ from age 37 in a way that no single split on a range captures. **Target encoding** gives the model that rate directly. It adds a feature containing the average of `Will_Buy_EV` among rows that share the same value.

## Smoothing

A value seen in only three rows may have a buying rate of 0% or 100% purely by chance. We therefore shrink each group's rate toward the overall rate (about 17.5%):

```text
encoded = (buyers in group + 17.5% × 20) / (rows in group + 20)
```

The `20` is `CFG.TE_SMOOTH`. It acts like adding 20 "average" rows to every group, so large groups keep their own rate and tiny groups stay near the overall rate.

## The leakage trap

Suppose we computed the rates from all of `train.csv` and then ran cross-validation. Each validation row's own label would already be inside its encoding. The model would look better in validation than it will on the test set, and our OOF score would stop being trustworthy.

We apply the rule from Concept 1 a second time:

- **Validation and test rows** get rates computed only from the current fold's training rows.
- **Training rows** get rates from an *inner* 5-fold split of the training rows. This way no training row's encoding includes its own label. Without the inner split, the model would learn to trust the encodings more than it should.

This happens inside `add_target_encoding(...)` in the notebook, which is called fresh in every fold.

## What the screen showed

FAST screen (LightGBM only, learning rate 0.05, same folds):

| Variant | Fold AUCs | OOF AUC | Change | Folds improved |
| --- | --- | ---: | ---: | :---: |
| Clip flags (reference) | .94055 .94147 .94278 .94227 .94181 | 0.94176 | — | — |
| + target encoding, 5 columns | .94388 .94447 .94572 .94505 .94482 | 0.94478 | +0.00302 | 5/5 |
| + 2 pairs (`Age × City_Type`, `Stations_Home × Home_Charging`) | .94390 .94444 .94572 .94505 .94492 | 0.94479 | +0.00303 | 5/5 |
| + count encoding | .94438 .94496 .94623 .94564 .94533 | 0.94530 | +0.00354 | 5/5 |

The pairs add only +0.00001, so they are left out (`CFG.TE_PAIRS = []`). Count encoding helps, but it is a separate idea, so it becomes the next experiment.

**Check your understanding:** Why can't we compute one set of target encodings from all of `train.csv` before the CV loop? **Each validation row's label would be baked into its own feature, so the OOF score would be too optimistic.**
