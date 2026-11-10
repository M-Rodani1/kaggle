"""LightGBM with the notebook's tuned settings on 5 or 10 folds, as a blend member.

10 folds refine the fixed 5: each fixed fold is split in half (StratifiedKFold(2, shuffle=True, random_state=42)), and
each half is scored by a model trained on the other 90% of the rows. The OOF is still one prediction per row, so it
blends with every 5-fold member. As in the notebook, early stopping watches the rows being scored.
Usage: python3 scripts/run_lgb_10fold.py <experiment> <feature spec> [nfold=10] [lr=0.02] [seeds=5]
Saves results/preds/<experiment>_LightGBM_{oof,test}.npy (results/screen/ for experiment names starting "screen").
"""
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from common import features, fold_id, save_member, split_ids, y

TUNED = dict(num_leaves=80, min_child_samples=7, subsample=0.9557, colsample_bytree=0.5715, reg_lambda=0.1676,
             reg_alpha=0.4640, min_split_gain=0.3337, max_bin=1023, cat_smooth=16.41, cat_l2=2.798, path_smooth=12.14)
experiment, spec = sys.argv[1], sys.argv[2]
opts = dict(a.split("=") for a in sys.argv[3:])
nfold, lr, seeds = int(opts.get("nfold", 10)), float(opts.get("lr", 0.02)), int(opts.get("seeds", 5))

x, x_test = features(spec)
split = split_ids(nfold)
print(f"LightGBM {experiment}: {x.shape[1]} features ({spec}), {nfold} folds, lr {lr}, {seeds} seeds", flush=True)
oof, test_pred, started = np.zeros(len(y)), np.zeros(len(x_test)), time.time()
for k in range(nfold):
    tr, va = split != k, split == k
    for seed in range(seeds):
        model = lgb.LGBMClassifier(n_estimators=20000, learning_rate=lr, subsample_freq=1, random_state=42 + seed,
                                   verbose=-1, **TUNED)
        model.fit(x[tr], y[tr], eval_set=[(x[va], y[va])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200 if lr < 0.05 else 100, verbose=False)])
        oof[va] += model.predict_proba(x[va], num_iteration=model.best_iteration_)[:, 1] / seeds
        test_pred += model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1] / seeds / nfold
    print(f"  split {k}: auc {roc_auc_score(y[va], oof[va]):.5f} | {(time.time() - started) / 60:.1f} min", flush=True)
folds = [roc_auc_score(y[fold_id == k], oof[fold_id == k]) for k in range(5)]
print(f"LightGBM OOF auc {roc_auc_score(y, oof):.5f} | fixed folds {' '.join(f'{f:.5f}' for f in folds)}", flush=True)
if experiment.startswith("screen"):
    np.save(f"results/screen/{experiment}_LightGBM_oof.npy", oof)
else:
    save_member(experiment, "LightGBM", oof, test_pred)
