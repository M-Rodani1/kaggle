"""Hill-climb a blend of saved members (our own models only), write its submission copies and log it.

Members are Label=experiment or Label=experiment/model (the file stem in results/preds/ when it differs from Label).
Test predictions use the member's refit file (<experiment>_<model>_refit_test.npy) when it exists, else its
fold-averaged test file. Writes submission_<name>.csv (refit where available) and submission_<name>_foldavg.csv;
never touches submission.csv. Adds one "Blend" row to experiments.csv, compared with exp13's blend.
Usage: python3 scripts/blend_own.py <name> LightGBM=exp15_route_aux CatBoost=exp10_catboost_cat ... [--no-log]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.chdir(Path(__file__).resolve().parents[1])  # Paths below are relative to the project root.
BASELINE = "exp13_nn_tuned_cb_orig"

name, specs = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(str).str.lower().eq("true").to_numpy()
fold = pd.read_csv("data/folds_seed42.csv")["fold"].to_numpy()
oofs, tests, foldavg, sources = {}, {}, {}, {}
for spec in specs:
    label, source = spec.split("=")
    experiment, model = source.split("/") if "/" in source else (source, label)
    oofs[label] = np.load(f"results/preds/{experiment}_{model}_oof.npy")
    foldavg[label] = np.load(f"results/preds/{experiment}_{model}_test.npy")
    refit = Path(f"results/preds/{experiment}_{model}_refit_test.npy")
    tests[label] = np.load(refit) if refit.is_file() else foldavg[label]
    sources[label] = f"{source}{' refit' if refit.is_file() else ''}"


def hill_climb(steps=50):
    """The notebook's ensemble selection: add (with replacement) the member that most improves the OOF AUC."""
    counts = dict.fromkeys(oofs, 0)
    first = max(oofs, key=lambda n: roc_auc_score(y, oofs[n]))
    counts[first] = 1
    blend, best, best_counts = oofs[first].copy(), roc_auc_score(y, oofs[first]), dict(counts)
    for step in range(1, steps):
        trials = {n: (blend * step + oofs[n]) / (step + 1) for n in oofs}
        scores = {n: roc_auc_score(y, v) for n, v in trials.items()}
        pick = max(scores, key=scores.get)
        counts[pick] += 1
        blend = trials[pick]
        if scores[pick] > best:
            best, best_counts = scores[pick], dict(counts)
    total = sum(best_counts.values())
    return {n: c / total for n, c in best_counts.items()}


weights = hill_climb()
oof = sum(w * oofs[n] for n, w in weights.items())
test_refit = sum(w * tests[n] for n, w in weights.items())
test_avg = sum(w * foldavg[n] for n, w in weights.items())
folds = [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(5)]
score = roc_auc_score(y, oof)
log = pd.read_csv("results/experiments.csv")
base = log[(log.experiment == BASELINE) & (log.model == "Blend")].iloc[0]
base_folds = [float(v) for v in base.fold_scores.split()]
note = ", ".join(f"{n} {w:.2f}" for n, w in weights.items()) + " | " + ", ".join(f"{n}: {s}" for n, s in sources.items())
for n in oofs:
    print(f"  {n:14s} OOF {roc_auc_score(y, oofs[n]):.5f}  weight {weights[n]:.2f}  ({sources[n]})")
print(f"{name}: blend OOF {score:.5f} | folds {' '.join(f'{f:.5f}' for f in folds)} | vs {BASELINE} "
      f"{score - base.oof_score:+.5f}, {sum(round(f, 5) > b for f, b in zip(folds, base_folds))}/5 folds better")

sample = pd.read_csv("data/sample_submission.csv")[["id"]]
test_ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
for suffix, values in (("", test_refit), ("_foldavg", test_avg)):
    sub = sample.copy()
    sub["satisfaction"] = sub["id"].map(pd.Series(values, index=test_ids))
    assert sub["satisfaction"].notna().all() and sub["satisfaction"].between(0, 1).all()
    sub.to_csv(f"results/submissions/submission_{name}{suffix}.csv", index=False)
np.save(f"results/preds/{name}_Blend_oof.npy", oof)
np.save(f"results/preds/{name}_Blend_test.npy", test_avg)
np.save(f"results/preds/{name}_Blend_refit_test.npy", test_refit)
print(f"wrote results/submissions/submission_{name}.csv (refit where available) and _foldavg.csv")
if "--no-log" not in sys.argv:
    row = dict(experiment=name, model="Blend", n_features=np.nan, fold_scores=" ".join(f"{f:.5f}" for f in folds),
               oof_score=round(score, 5), delta_vs_baseline=round(score - base.oof_score, 5),
               folds_improved=sum(round(f, 5) > b for f, b in zip(folds, base_folds)), minutes=0.0, fold_seed=42,
               notes=note, public_lb=np.nan, logged_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), device="")
    log = pd.concat([log[~((log.experiment == name) & (log.model == "Blend"))], pd.DataFrame([row])], ignore_index=True)
    log.to_csv("results/experiments.csv", index=False)
