"""Stack the public S6E10 OOF/test prediction pairs together with our own models (logistic stack + LightGBM blend).

Adapted from Dariush Afshar's public pool notebook and its follow-up ("a stack of stacks inflates nested CV"):
same sources, loaders, checks and near-copy rule, and the clean recipe they recommend (no published stacks as members).
Our members are the exp13 blend's models from preds/. Every public source sits in public_oof/<slug>/; missing
sources are skipped. Only prediction arrays (.npy/.npz/.csv) are read; nothing from the downloads is executed.

Nested CV on our fixed folds (identical to the pool's StratifiedKFold(5, shuffle=True, random_state=42)): each fold's
stacker is fit on the other four folds' OOF predictions. Four settings are compared (public pool with/without our
members x plain LR flat C = 0.003 / 0.8 LR + 0.2 LightGBM with family-wise L2); the best by nested CV is refit on all
rows and written as submission_<EXPERIMENT>.csv, with a row in experiments.csv. submission.csv is replaced only when
our members beat the public-only stack in the same run.
Usage: python3 stack_public.py [--ours-only]   (--ours-only: smoke test on our own members)
"""
import glob
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])  # Paths below are relative to the project root.

warnings.filterwarnings("ignore")
EXPERIMENT = "exp18_stack_tabpfn_ne4"
CURRENT_UPLOAD = "exp16_stack_exp15"  # The stack behind submission.csv; replaced only if clearly beaten (see the end).
ROOT = "data/public_oof"
OURS = {  # member name: (experiment, model) in results/preds/. Single models only: a blend fitted on these rows would
    # inflate the stack's nested CV. (exp14 used the exp13 members and found they added nothing.)
    "LightGBM_k10": ("exp15_lgb_k10", "LightGBM"), "LightGBM_route": ("exp15_route_aux", "LightGBM"),
    "RealMLP_k10": ("exp15_realmlp_k10x2", "RealMLP"), "CatBoost": ("exp10_catboost_cat", "CatBoost"),
    "CatBoostFast_route": ("exp15_route_aux", "CatBoostFast"), "CatBoostOrig": ("exp13_nn_tuned_cb_orig", "CatBoostOrig"),
    "NN_route": ("exp15_route_aux", "NN"),
}
MIN_AUC, MAX_AUC, AGREE_GAP, DRIFT, DUP_RHO = 0.90, 0.9625, 0.03, 0.03, 0.9999
C_LR, W_LGB = 0.003, 0.2
OURS_ONLY = "--ours-only" in sys.argv
T0 = time.time()

train = pd.read_csv("data/train.csv", usecols=["id", "satisfaction"])
train_ids, test_ids = train["id"].to_numpy(), pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
y = train["satisfaction"].astype(str).str.lower().map({"true": 1, "false": 0}).to_numpy().astype(int)
fold = pd.read_csv("data/folds_seed42.csv")["fold"].to_numpy()


# ---------- loaders (one per public source; each yields name, oof, test) ----------
def first(d, name):
    hits = sorted(glob.glob(os.path.join(d, "**", name), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def by_id(df, ids, col):
    s = df.set_index("id")[col]
    assert s.index.is_unique, "duplicate ids"
    s = s.reindex(ids)
    assert s.notna().all(), f"{int(s.isna().sum()):,} ids missing"
    return s.to_numpy(np.float64)


def npy_pairs(d, prefix, pattern="oof_*.npy"):
    for f in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        name = os.path.basename(f)[4:-4]
        yield prefix + name, np.load(f), np.load(os.path.join(os.path.dirname(f), f"test_{name}.npy"))


def npz_pairs(d, prefix, pattern="*.npz", strip=0):
    for f in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        with np.load(f) as z:
            if "oof" in z.files and "test" in z.files:
                yield prefix + os.path.basename(f)[strip:-4], z["oof"], z["test"]


def load_golem(d):
    man = pd.read_csv(f"{d}/manifest.csv")
    for r in man.itertuples():
        if "stack" not in r.wide_column:  # The library also ships its own stack; never a member.
            yield "gl_" + r.wide_column, np.load(f"{d}/{r.oof_file}"), np.load(f"{d}/{r.test_file}")


def load_abdullah(d):
    for m in ("tabpfn_catfd10", "tabpfn_raw", "tabpfn_te_cnt", "cat_v3", "xgb_v3"):
        o, t = pd.read_csv(f"{d}/{m}_oof.csv"), pd.read_csv(f"{d}/{m}_test.csv")
        yield "ab_" + m, by_id(o, train_ids, "pred"), by_id(t, test_ids, "pred")


def load_sadam(d):
    yield from npz_pairs(d, "st_", "preds_*.npz", strip=6)
    for n in ("orig_cat_pred", "orig_model_pred"):
        with np.load(first(d, f"{n}.npz")) as z:
            yield "st_" + n, z["train"], z["test"]


def load_koumeimaki(d):
    assert (pd.read_csv(f"{d}/train_ids_folds.csv")["id"].to_numpy() == train_ids).all()
    assert (pd.read_csv(f"{d}/test_ids.csv")["id"].to_numpy() == test_ids).all()
    yield from npy_pairs(d, "km_")


def load_gp_tabpfn(d):
    for f in sorted(glob.glob(f"{d}/oof_*.csv")):
        n = os.path.basename(f)[4:-4]
        yield "gpt_" + n, by_id(pd.read_csv(f), train_ids, "pred"), by_id(pd.read_csv(f"{d}/test_{n}.csv"), test_ids, "pred")


def load_gp_extra(d):
    for name, o, t in npy_pairs(d, "gpx_"):
        if "stack" not in name:  # A published stack was fit on every row: as a member it inflates nested CV.
            yield name, o, t


def load_yekenot(d):
    o, t = pd.read_csv(first(d, "oof_preds.csv")), pd.read_csv(first(d, "submission.csv"))
    yield "yk_realmlp", by_id(o, train_ids, "satisfaction"), by_id(t, test_ids, "satisfaction")


def load_kagan_enc(d):
    o, t = pd.read_csv(first(d, "oof_members.csv")), pd.read_csv(first(d, "test_members.csv"))
    for c in [c for c in o.columns if c != "id"]:
        yield "kg_" + c, by_id(o, train_ids, c), by_id(t, test_ids, c)


def load_aman_v5(d):
    with np.load(first(d, "oof_artifacts.npz")) as z:
        assert (z["y"] == y).all(), "labels in a different row order"
        for m in ("lgbm_champion", "lgbm_te", "lgbm_te_rows", "xgb_gpu", "catboost_gpu"):
            yield "av5_" + m, z["oof_" + m], z["test_" + m]


def load_parquet_library(prefix):
    """Wide OOF/test parquet libraries: id (+ fold, label) and one probability column per model."""
    def load(d):
        o, t = pd.read_parquet(f"{d}/oof_predictions.parquet"), pd.read_parquet(f"{d}/test_predictions.parquet")
        for c in [c for c in o.columns if c not in ("id", "fold", "satisfaction") and c in t.columns]:
            yield prefix + c, by_id(o, train_ids, c), by_id(t, test_ids, c)
    return load


def load_najiama(d):  # Only the single XGBoost; the other six files are blends.
    yield ("nj_xgb", by_id(pd.read_csv(f"{d}/07_XGBoost_oof.csv"), train_ids, "oof_pred"),
           by_id(pd.read_csv(f"{d}/07_submission.csv"), test_ids, "satisfaction"))


def load_ours(_):
    for name, (exp, model) in OURS.items():
        yield "ours_" + name, np.load(f"results/preds/{exp}_{model}_oof.npy"), np.load(f"results/preds/{exp}_{model}_test.npy")


SOURCES = [  # key, folder under public_oof/, loader
    ("c124", "s6e10-124-public-oof-and-test-predictions", lambda d: npy_pairs(d, "")),
    ("tfm", "s6e10-tfm-oof-and-test-predictions", lambda d: npy_pairs(d, "tfm_")),
    ("golem", "s6e10-golem-oof-library", load_golem),
    ("abdullah", "oof-dataset", load_abdullah),
    ("sadam", "ps-s6e10-airline-satisfaction-artifacts", load_sadam),
    ("forge", "s6e10-forge-own-pairs", lambda d: npy_pairs(d, "am_", "oof_forge_*.npy")),
    ("sachith", "s6e10-stack-oof-predictions", lambda d: npz_pairs(d, "s7_")),
    ("koumeimaki", "s6e10-route-tfm-nn-oof", load_koumeimaki),
    ("gp_tabpfn", "s6e10-tabpfn-member", load_gp_tabpfn),
    ("gp_extra", "s6e10-goodpjw2008-extra-members-oof", load_gp_extra),
    ("dmitriy", "s6e10-catboost-all-categorical-oof-and-test",
     lambda d: iter([("dk_cb_cat", np.load(first(d, "oof_cb_cat.npy")), np.load(first(d, "test_cb_cat.npy")))])),
    ("kagan_abl", "s6e10-ablation-members-oof", lambda d: npy_pairs(d, "kga_")),
    ("yekenot", "ps-s6-e10-realmlp-pytabkit", load_yekenot),
    ("hermengardo", "single-catboost-eda-ps6e10",
     lambda d: iter([("hm_cat", np.load(first(d, "oof_train.npy")), np.load(first(d, "oof_test.npy")))])),
    ("kratosyan", "s6e10-xgboost-realmlp-route-id-teacher", lambda d: npz_pairs(d, "kr_", "*_route_ogaux.npz")),
    ("kagan_enc", "s6e10-encodings-catboost-realmlp-lb-0-961", load_kagan_enc),
    ("aman_v5", "s6e10-the-original-prior-stack-v5", load_aman_v5),
    ("kozy", "s6e10-source-transfer-foundation-model-oof",
     lambda d: iter([("kz_source_transfer", np.load(first(d, "source_transfer_oof.npy")),
                      np.load(first(d, "source_transfer_test.npy")))])),
    # Added for exp17 (published after the 247-member pool, or not in it).
    ("busyaprime", "s6e10-busyaprime-five-members-oof", lambda d: npz_pairs(d, "bpx_", "bp_*.npz", strip=3)),
    ("megayak", "s6e10-oof-library", load_parquet_library("mg_")),
    ("arhancanli", "s6e10-oof-test-predictions", load_parquet_library("ar_")),
    ("najiama", "s6e10-oof", load_najiama),
    ("ours", None, load_ours),
]
GB_EXCLUDE = {"tfm", "golem", "sadam", "koumeimaki", "kagan_enc", "aman_v5", "forge", "kozy"}  # As in Trust the CV.


# ---------- load and check ----------
def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def is_rank(v):
    q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    return bool(v.min() >= 0 and v.max() <= 1 and np.abs(q - [0.1, 0.25, 0.5, 0.75, 0.9]).max() < 0.01)


members, report = [], []
for key, slug, load in SOURCES:
    if OURS_ONLY and key != "ours":
        continue
    d = os.path.join(ROOT, slug) if slug else None
    if slug and not os.path.isdir(d):
        report.append((key, "not downloaded", 0, 0))
        continue
    try:
        got = list(load(d))
    except Exception as e:
        report.append((key, f"skipped: {type(e).__name__}: {e}"[:90], 0, 0))
        continue
    kept = 0
    for name, o, t in got:
        o, t = np.asarray(o, np.float64).ravel(), np.asarray(t, np.float64).ravel()
        if o.shape != y.shape or t.shape != test_ids.shape or not (np.isfinite(o).all() and np.isfinite(t).all()):
            continue
        is_logit = o.min() < 0 or o.max() > 1 or t.min() < 0 or t.max() > 1
        if not is_logit and is_rank(o) != is_rank(t):
            continue  # OOF and test on different scales
        p_o, p_t = (1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))) if is_logit else (o, t)
        zo, zt = (o, t) if is_logit else (logit(o), logit(t))
        auc = roc_auc_score(y, zo)
        if not (MIN_AUC <= auc <= MAX_AUC) or abs(p_t.mean() - p_o.mean()) > DRIFT:
            continue  # wrong rows, not out of fold, or train/test drift
        members.append(dict(source=key, name=name, auc=auc, oof=np.clip(zo, -30, 30).astype(np.float32),
                            test=np.clip(zt, -30, 30).astype(np.float32)))
        kept += 1
    report.append((key, "loaded", len(got), kept))
print("\n".join(f"{k:12s} {s:40s} {n:4d} pairs, {m:4d} passed checks" for k, s, n, m in report))
assert members, "no members loaded"

# Agreement check: each member's test half must agree with the pool median about as well as its OOF half.
Zo = np.column_stack([m["oof"] for m in members]).astype(np.float64)
Zt = np.column_stack([m["test"] for m in members]).astype(np.float64)
ref_o, ref_t = np.median(Zo, 1), np.median(Zt, 1)
corr = lambda Z, r: ((Z - Z.mean(0)) * (r - r.mean())[:, None]).sum(0) / np.sqrt(((Z - Z.mean(0)) ** 2).sum(0) * ((r - r.mean()) ** 2).sum())
agree_gap = corr(Zo, ref_o) - corr(Zt, ref_t)
ok = agree_gap <= AGREE_GAP
print(f"agreement check dropped {int((~ok).sum())}")
members = [m for m, k in zip(members, ok) if k]

# Near-copies: rank correlation above DUP_RHO with an earlier member (public base first, ours last).
R = np.column_stack([rankdata(m["oof"]) for m in members])
R -= R.mean(0)
R /= np.sqrt((R ** 2).mean(0))
RHO = (R.T @ R) / len(R)
del R
keep = []
for j in range(len(members)):
    if not keep or max(RHO[j, i] for i in keep) <= DUP_RHO:
        keep.append(j)
print(f"near-copies dropped {len(members) - len(keep)}")
members = [members[j] for j in keep]
NAMES = [m["name"] for m in members]
SRC = np.array([m["source"] for m in members])
Z = np.column_stack([m["oof"] for m in members]).astype(np.float64)
ZT = np.column_stack([m["test"] for m in members]).astype(np.float64)
del members, Zo, Zt
ours = SRC == "ours"
print(f"pool: {Z.shape[1]} members ({int((~ours).sum())} public, {int(ours.sum())} ours) | best single "
      f"{max(roc_auc_score(y, Z[:, j]) for j in range(Z.shape[1])):.6f} | {time.time() - T0:.0f}s", flush=True)


# ---------- stackers ----------
def family(n):
    n = n.lower()
    if "tabpfn" in n or n.startswith("pfn_"):
        return 3.0
    if n.startswith("tfm_") or "tabicl" in n or "tabm" in n:
        return 1.5
    return 1.0  # GBDTs, MLPs and ours


SCALE = np.array([family(n) for n in NAMES])
GB_COLS_ALL = np.array([s not in GB_EXCLUDE for s in SRC])
P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)


def fit_lr(A, ya, C):
    return LogisticRegression(C=C, tol=1e-6, max_iter=20000).fit(A, ya)


def fit_gb(A, ya):
    idx = np.random.default_rng(0).permutation(len(A))
    cut = int(len(idx) * 0.9)
    return lgb.train(P, lgb.Dataset(A[idx[:cut]], ya[idx[:cut]]), 3000,
                     valid_sets=[lgb.Dataset(A[idx[cut:]], ya[idx[cut:]])], callbacks=[lgb.early_stopping(100, verbose=False)])


def _lr_fold(A, k, C):
    a, b = fold != k, fold == k
    return b, fit_lr(A[a], y[a], C).decision_function(A[b])


def nested_lr(A, C):
    o = np.zeros(len(y))
    for b, v in Parallel(n_jobs=5)(delayed(_lr_fold)(A, k, C) for k in range(5)):
        o[b] = v
    return o


def nested_gb(A):
    o = np.zeros(len(y))
    for k in range(5):
        a, b = fold != k, fold == k
        m = fit_gb(A[a], y[a])
        o[b] = m.predict(A[b], raw_score=True, num_iteration=m.best_iteration)
    return o


zs = lambda v, ref: (v - ref.mean()) / ref.std()
fold_mean = lambda o: np.mean([roc_auc_score(y[fold == k], o[fold == k]) for k in range(5)])
pools = {"public": ~ours, "public+ours": np.ones(len(NAMES), bool)} if (~ours).any() else {"ours": ours}
results = {}
for pool_name, cols in pools.items():
    t = time.time()
    A_flat, A_fam = Z[:, cols], Z[:, cols] * SCALE[cols]
    lr_flat, lr_fam = nested_lr(A_flat, C_LR), nested_lr(A_fam, C_LR)
    gb_cols = cols & GB_COLS_ALL
    gb = nested_gb(Z[:, gb_cols]) if gb_cols.any() else None
    results[pool_name, "LR flat"] = (lr_flat, None)
    if gb is not None:
        results[pool_name, "blend family"] = ((1 - W_LGB) * zs(lr_fam, lr_fam) + W_LGB * zs(gb, gb), (lr_fam, gb))
    for (p, s), (o, _) in results.items():
        if p == pool_name:
            print(f"{pool_name:12s} {s:13s} nested OOF: fold-mean {fold_mean(o):.6f}, pooled {roc_auc_score(y, o):.6f}"
                  f" | {time.time() - t:.0f}s", flush=True)
best = max(results, key=lambda k: fold_mean(results[k][0]))
print(f"best by nested CV: {best} ({fold_mean(results[best][0]):.6f})", flush=True)
if OURS_ONLY:
    sys.exit(0)

# ---------- refit the best setting on every row, write the submission ----------
pool_name, setting = best
cols = pools[pool_name]
lr_A, lr_T = (Z[:, cols], ZT[:, cols]) if setting == "LR flat" else (Z[:, cols] * SCALE[cols], ZT[:, cols] * SCALE[cols])
lr_test = fit_lr(lr_A, y, C_LR).decision_function(lr_T)
if setting == "LR flat":
    test_score, oof_score = lr_test, results[best][0]
else:
    lr_oof, gb_oof = results[best][1]
    gb_cols = cols & GB_COLS_ALL
    g = fit_gb(Z[:, gb_cols], y)
    gb_test = g.predict(ZT[:, gb_cols], raw_score=True, num_iteration=g.best_iteration)
    test_score = (1 - W_LGB) * zs(lr_test, lr_oof) + W_LGB * zs(gb_test, gb_oof)  # OOF scale, so test matches
    oof_score = results[best][0]
prob = 1 / (1 + np.exp(-test_score))
sub = pd.read_csv("data/sample_submission.csv")[["id"]]
sub["satisfaction"] = sub["id"].map(pd.Series(prob, index=test_ids))
assert sub["satisfaction"].notna().all() and sub["satisfaction"].between(0, 1).all()
sub.to_csv(f"results/submissions/submission_{EXPERIMENT}.csv", index=False)
# Replace the upload only if this stack beats the current one by >= 0.00002 fold-mean with >= 4/5 folds better
# (same meta folds, both nested OOF).
current = np.load(f"results/preds/{CURRENT_UPLOAD}_Stack_oof.npy")
cur_folds = [roc_auc_score(y[fold == k], current[fold == k]) for k in range(5)]
new_folds = [roc_auc_score(y[fold == k], oof_score[fold == k]) for k in range(5)]
gain = np.mean(new_folds) - np.mean(cur_folds)
replace_upload = gain >= 2e-5 and sum(n > c for n, c in zip(new_folds, cur_folds)) >= 4
print(f"vs {CURRENT_UPLOAD}: fold-mean {gain:+.6f}, {sum(n > c for n, c in zip(new_folds, cur_folds))}/5 folds better", flush=True)
if replace_upload:
    sub.to_csv("results/submissions/submission.csv", index=False)
np.save(f"results/preds/{EXPERIMENT}_Stack_oof.npy", oof_score)
np.save(f"results/preds/{EXPERIMENT}_Stack_test.npy", test_score)

log = pd.read_csv("results/experiments.csv")
folds = [roc_auc_score(y[fold == k], oof_score[fold == k]) for k in range(5)]
base = log[(log.experiment == "exp14_public_stack") & (log.model == "Stack")]
row = dict(experiment=EXPERIMENT, model="Stack", n_features=int(cols.sum()), fold_scores=" ".join(f"{f:.5f}" for f in folds),
           oof_score=round(roc_auc_score(y, oof_score), 5),
           delta_vs_baseline=round(roc_auc_score(y, oof_score) - base.oof_score.iloc[0], 5) if len(base) else np.nan,
           folds_improved=sum(round(f, 5) > float(b) for f, b in zip(folds, base.fold_scores.iloc[0].split())) if len(base) else np.nan,
           minutes=round((time.time() - T0) / 60, 1), fold_seed=42,
           notes=f"{pool_name}, {setting}, C {C_LR}; n_features = stack members; nested OOF", public_lb=np.nan,
           logged_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), device="cpu")
log = pd.concat([log[log.experiment != EXPERIMENT], pd.DataFrame([row])], ignore_index=True)
log.to_csv("results/experiments.csv", index=False)
print(f"wrote results/submissions/submission_{EXPERIMENT}.csv ({pool_name}, {setting}); submission.csv "
      f"{'replaced' if replace_upload else 'left unchanged'}; "
      f"nested OOF {row['oof_score']:.5f} | total {(time.time() - T0) / 60:.1f} min")
