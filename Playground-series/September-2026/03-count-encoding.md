# Concept 3: Count encoding

Target encoding (Concept 2) tells the model *how often people with this value buy*. Count encoding tells it *how common this value is*. For each of the five `TE_COLS`, we add a `CNT_` column holding the number of rows in `train.csv` and `test.csv` combined that share the row's value.

## Why frequency helps

In synthetic Playground data, common and rare values often behave differently:

- A value seen 60,000 times, such as the 30,000 income floor, is a "default" value the data generator produces a lot.
- A value seen 3 times sits in the sparse tail of the distribution.

Count encoding also tells the model how much to trust the target encoding. A target rate built from 3 rows is mostly the smoothing prior, while one built from 60,000 rows is solid. With both columns, the tree can learn "trust `TE_` more when `CNT_` is large".

## Why using the test rows is safe here

Counting uses **no labels**. We only ask how often a value appears, never whether those people bought an EV. Including `test.csv` in the counts makes them more accurate, and it cannot leak the answer, because the test labels are never read. So counts are computed once, outside the CV loop.

Compare this with target encoding. Target encoding reads `Will_Buy_EV`, so it must be rebuilt inside every fold.

## What the screen showed

In the FAST screen, adding counts on top of target encoding improved every fold. OOF AUC went from 0.94479 to 0.94530, a gain of +0.00051. The full run confirmed it: the blend went from 0.94500 to **0.94556** (+0.00056), and all 5 folds improved for LightGBM, CatBoost, and the blend.

**Check your understanding:** Why can count encoding use `test.csv` when target encoding cannot? **Counts never look at labels, so there is nothing about the answer to leak.**
