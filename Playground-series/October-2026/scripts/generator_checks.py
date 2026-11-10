"""Look for traces of the synthetic-data generator that carry label signal the exp16 stack has not captured.

Five checks, each building label-free features for every competition row (train + test):
  1. copies:   original rows that match a competition row on 21, 20, 19 or 18 of the 21 shared columns, and their
               satisfaction rate (near-copies of real surveys keep the real label)
  2. values:   how often each value appears in the competition data relative to the original (values the generator
               invents or over-produces), plus a never-seen-in-original flag
  3. modes:    Gaussian-mixture mode of each numeric column, as in CTGAN's mode-specific normalisation
  4. realness: out-of-fold probability that a row is synthetic rather than original (adversarial classifier)
  5. teacher:  logit of a model trained only on the original data (plus how well it transfers, a label-noise gauge)

Test: a small LightGBM boosts on top of the exp16 stack's OOF score (Platt-calibrated init_score, and a feature) on the fixed folds. The control gets
the 21 raw columns; each check gets raw + its features. A check matters only if it beats the control by >= +0.0002
on >= 4/5 folds. Features are saved to results/features/gen_<check>.parquet (train rows first, then test).
"""
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

os.chdir(Path(__file__).resolve().parents[1])  # Paths below are relative to the project root.

BASE = "results/preds/exp16_stack_exp15_Stack_oof.npy"
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMERIC = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
SMOOTH = 20

train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y = train.pop("satisfaction").astype(str).str.lower().map({"true": 1, "false": 0}).to_numpy()
train.pop("id"), test.pop("id")
COLS = list(train.columns)  # the 21 shared columns ("Inflight service" exists only in the original)
original = pd.read_csv("data/original/original_all.csv")
y_orig = original["satisfaction"].astype(int).to_numpy()
comp = pd.concat([train, test], ignore_index=True)
N_TRAIN, N_COMP = len(train), len(comp)
fold_id = pd.read_csv("data/folds_seed42.csv")["fold"].to_numpy()
prior = y_orig.mean()

# Shared integer codes for every column over competition + original rows (missing -> its own code).
both = pd.concat([comp[COLS], original[COLS]], ignore_index=True)
codes = np.stack([pd.factorize(both[c].fillna(-1))[0] for c in COLS], axis=1)
codes_comp, codes_orig = codes[:N_COMP], codes[N_COMP:]


def log(msg):
    print(msg, flush=True)


def smoothed(total, count):
    return ((total + prior * SMOOTH) / (count + SMOOTH)).astype("float32")


def check_copies():
    """Additive 64-bit row hashes: the hash of a column subset is the full hash minus the dropped columns' parts."""
    rng = np.random.default_rng(0)
    parts = np.stack([rng.integers(0, 2**63, size=codes[:, j].max() + 1, dtype=np.uint64)[codes[:, j]]
                      for j in range(len(COLS))])  # (21, rows)
    full = parts.sum(axis=0, dtype=np.uint64)
    out = {}
    for dropped_n in range(4):
        level = len(COLS) - dropped_n
        count = np.zeros(N_COMP, dtype=np.float64)
        total = np.zeros(N_COMP, dtype=np.float64)
        for dropped in combinations(range(len(COLS)), dropped_n):
            key = full - parts[list(dropped)].sum(axis=0, dtype=np.uint64) if dropped else full
            keys, inv = np.unique(key[N_COMP:], return_inverse=True)
            n = np.bincount(inv).astype(np.float64)
            s = np.bincount(inv, weights=y_orig).astype(np.float64)
            pos = np.clip(np.searchsorted(keys, key[:N_COMP]), 0, len(keys) - 1)
            hit = keys[pos] == key[:N_COMP]
            count[hit] += n[pos[hit]]
            total[hit] += s[pos[hit]]
        out[f"copy_n{level}"] = np.log1p(count).astype("float32")
        out[f"copy_rate{level}"] = smoothed(total, count)
        log(f"   match on {level}/21 columns: {np.mean(count[:N_TRAIN] > 0):.4%} of train rows, "
            f"{np.mean(count[N_TRAIN:] > 0):.4%} of test rows have one")
    best = np.zeros(N_COMP, dtype="float32")
    for level in (18, 19, 20, 21):
        best[out[f"copy_n{level}"] > 0] = level
    out["copy_best_level"] = best
    return pd.DataFrame(out)


def check_values():
    out = {}
    for j, c in enumerate(COLS):
        n_comp = np.bincount(codes_comp[:, j], minlength=codes[:, j].max() + 1)
        n_orig = np.bincount(codes_orig[:, j], minlength=codes[:, j].max() + 1)
        ratio = np.log((n_comp + 1) / N_COMP) - np.log((n_orig + 1) / len(original))
        out[f"val_ratio_{c}"] = ratio[codes_comp[:, j]].astype("float32")
        unseen = n_orig[codes_comp[:, j]] == 0
        out[f"val_unseen_{c}"] = unseen.astype("float32")
        if unseen.any():
            log(f"   {c}: {unseen[:N_TRAIN].mean():.3%} of train rows have a value never seen in the original")
    return pd.DataFrame(out)


def check_modes():
    out = {}
    rng = np.random.default_rng(0)
    for c in NUMERIC:
        x = comp[c].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        v = np.log1p(x[ok]) if "Delay" in c else x[ok]
        gm = GaussianMixture(n_components=10, random_state=0, max_iter=200).fit(
            v[rng.choice(len(v), min(len(v), 200_000), replace=False)].reshape(-1, 1))
        post = gm.predict_proba(v.reshape(-1, 1))
        mode = post.argmax(axis=1)
        sd = np.sqrt(gm.covariances_.reshape(-1))
        for name, val in [("mode", mode), ("mode_conf", post.max(axis=1)),
                          ("mode_pos", (v - gm.means_.reshape(-1)[mode]) / (4 * sd[mode]))]:
            col = np.full(N_COMP, np.nan, dtype="float32")
            col[ok] = val
            out[f"{name}_{c}"] = col
        log(f"   {c}: mode sizes {np.bincount(mode, minlength=10)}")
    return pd.DataFrame(out)


LGB_SMALL = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)


def oof_lgb(X, target, rounds=400, n_splits=5):
    oof = np.zeros(len(target))
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=0).split(X, target):
        m = lgb.train(LGB_SMALL, lgb.Dataset(X[tr], target[tr]), rounds)
        oof[va] = m.predict(X[va], raw_score=True)
    return oof


def check_realness():
    X = codes.astype("float32")
    is_synth = np.r_[np.ones(N_COMP), np.zeros(len(original))]
    oof = oof_lgb(X, is_synth)
    log(f"   synthetic-vs-original AUC: {roc_auc_score(is_synth, oof):.4f} (0.5 = indistinguishable)")
    is_test = np.r_[np.zeros(N_TRAIN), np.ones(N_COMP - N_TRAIN)]
    tt = oof_lgb(codes_comp.astype("float32"), is_test, rounds=200)
    log(f"   train-vs-test AUC: {roc_auc_score(is_test, tt):.4f}")
    r = oof[:N_COMP]
    log(f"   train AUC of the realness score alone: {roc_auc_score(y, r[:N_TRAIN]):.4f}")
    return pd.DataFrame({"realness_logit": r.astype("float32")})


def check_teacher():
    Xo = codes_orig.astype("float32")
    oof = oof_lgb(Xo, y_orig, rounds=600)
    log(f"   original model on held-out original rows: AUC {roc_auc_score(y_orig, oof):.5f}")
    m = lgb.train(LGB_SMALL, lgb.Dataset(Xo, y_orig), 600)
    t = m.predict(codes_comp.astype("float32"), raw_score=True)
    log(f"   same model on competition train rows:    AUC {roc_auc_score(y, t[:N_TRAIN]):.5f}")
    back = lgb.train(LGB_SMALL, lgb.Dataset(codes_comp[:N_TRAIN].astype("float32"), y), 600)
    log(f"   competition-trained model on original:   AUC {roc_auc_score(y_orig, back.predict(Xo)):.5f}")
    return pd.DataFrame({"teacher_logit": t.astype("float32")})


CHECKS = {"copies": check_copies, "values": check_values, "modes": check_modes,
          "realness": check_realness, "teacher": check_teacher}

BOOST = dict(objective="binary", learning_rate=0.02, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)


def residual_boost(extra):
    """Fold AUCs of a LightGBM boosting on the stack's OOF logit with raw columns (+ extra features)."""
    score = np.load(BASE)  # a blend score on a log-odds-like scale, not a probability
    X = train.copy()
    X["stack_score"] = score
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    if extra is not None:
        X = pd.concat([X, extra.iloc[:N_TRAIN].reset_index(drop=True)], axis=1)
    aucs, rounds = [], []
    for f in range(5):
        tr, va = np.where(fold_id != f)[0], np.where(fold_id == f)[0]
        rng = np.random.default_rng(f)
        stop = rng.random(len(tr)) < 0.1  # inner early-stopping split, never the validation fold
        fit, es = tr[~stop], tr[stop]
        platt = LogisticRegression(C=1e6).fit(score[tr, None], y[tr])  # calibrated starting logit, from training rows
        base = platt.decision_function(score[:, None])
        m = lgb.train(BOOST, lgb.Dataset(X.iloc[fit], y[fit], init_score=base[fit]), 3000,
                      valid_sets=[lgb.Dataset(X.iloc[es], y[es], init_score=base[es])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        aucs.append(roc_auc_score(y[va], base[va] + m.predict(X.iloc[va], raw_score=True)))
        rounds.append(m.best_iteration)
    return np.array(aucs), rounds


def main():
    feats = {}
    for name, fn in CHECKS.items():
        path = Path(f"results/features/gen_{name}.parquet")
        t0 = time.time()
        log(f"\n== CHECK {name.upper()}")
        if path.exists():
            feats[name] = pd.read_parquet(path)
            log(f"   (features loaded from {path})")
        else:
            feats[name] = fn()
            feats[name].to_parquet(path)
        log(f"   {feats[name].shape[1]} features, {time.time() - t0:.0f}s")

    p = np.load(BASE)
    base_folds = np.array([roc_auc_score(y[fold_id == f], p[fold_id == f]) for f in range(5)])
    log(f"\n== RESIDUAL BOOST TEST (exp16 stack OOF: fold mean {base_folds.mean():.6f})")
    ctrl, r = residual_boost(None)
    log(f"   control (raw columns): {ctrl.mean():.6f} ({ctrl.mean() - base_folds.mean():+.6f} vs stack), rounds {r}")
    runs = list(feats.items()) + [("all five", pd.concat(feats.values(), axis=1))]
    rows = []
    for name, extra in runs:
        a, r = residual_boost(extra)
        d = a - ctrl
        verdict = "IMPROVES" if d.mean() >= 2e-4 and (d > 0).sum() >= 4 else "no"
        log(f"   {name:9s}: {a.mean():.6f} | vs control {d.mean():+.6f}, better on {(d > 0).sum()}/5 folds | "
            f"rounds {r} | {verdict}")
        rows.append((name, a.mean(), d.mean(), int((d > 0).sum()), verdict))
    log("\nSUMMARY")
    for row in rows:
        log("   {:9s} AUC {:.6f}  delta {:+.6f}  folds {}/5  {}".format(*row))


if __name__ == "__main__":
    main()
