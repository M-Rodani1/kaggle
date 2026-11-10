"""RealMLP (pytabkit RealMLP_TD) on the fixed 5 folds, as a blend member.

Settings follow the public S6E10 RealMLPs (8-network internal ensemble, 4 epochs, 512-256-128, mish, PBLD numeric
embeddings). The best epoch is picked on a 5% split of each fold's training rows, never on the scored fold.
With nfold=10, each fixed fold is halved and each half is scored by a model trained on the other 90% of the rows.
Usage: python3 scripts/run_realmlp.py <experiment> <feature spec> [folds=0,1,2,3,4] [device=mps] [seed=0] [nfold=5]
Saves results/preds/<experiment>_RealMLP_{oof,test}.npy when all 5 folds ran.
"""
import sys
import time

import numpy as np
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score

from common import features, fold_id, save_member, split_ids, y

if sys.platform == "darwin":  # Same OpenMP guard as the notebook.
    torch.set_num_threads(1)

experiment, spec = sys.argv[1], sys.argv[2]
opts = dict(a.split("=") for a in sys.argv[3:])
nfold = int(opts.get("nfold", 5))
folds = [int(k) for k in opts.get("folds", ",".join(map(str, range(nfold)))).split(",")]
split = split_ids(nfold)
device, seed = opts.get("device", "mps"), int(opts.get("seed", 0))
PARAMS = dict(n_ens=8, n_epochs=4, batch_size=256, hidden_sizes=[512, 256, 128], act="mish", lr=0.04, wd=0.0236,
              p_drop=0.05, p_drop_sched="flat_cos", embedding_size=8, max_one_hot_cat_size=18, num_emb_type="pbld",
              plr_sigma=2.33, plr_lr_factor=0.115, plr_hidden_1=16, plr_hidden_2=8, first_layer_lr_factor=0.6,
              use_ls=True, ls_eps=0.0115, ls_eps_sched="coslog4", val_metric_name="1-auc_ovr",
              use_early_stopping=False, val_fraction=0.05, predict_batch_size=8192)

x, x_test = features(spec, fill_missing=True)
print(f"RealMLP {experiment}: {x.shape[1]} features ({spec}), device {device}, folds {folds}", flush=True)
oof, test_pred = np.zeros(len(y)), np.zeros(len(x_test))
for k in folds:
    started = time.time()
    tr, va = split != k, split == k
    model = RealMLP_TD_Classifier(device=device, random_state=k + 1000 * seed, verbosity=0, **PARAMS)
    model.fit(x[tr], y[tr])
    oof[va] = model.predict_proba(x[va])[:, 1]
    test_pred += model.predict_proba(x_test)[:, 1] / nfold
    print(f"RealMLP fold {k}: auc {roc_auc_score(y[va], oof[va]):.5f} | {(time.time() - started) / 60:.1f} min", flush=True)
if len(folds) == nfold:
    save_member(experiment, "RealMLP", oof, test_pred)
    fixed = " ".join(f"{roc_auc_score(y[fold_id == k], oof[fold_id == k]):.5f}" for k in range(5))
    print(f"RealMLP OOF auc {roc_auc_score(y, oof):.5f} | fixed folds {fixed}; saved", flush=True)
