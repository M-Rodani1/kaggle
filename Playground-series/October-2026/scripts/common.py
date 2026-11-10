"""Shared data and feature loading for the standalone member scripts (run_realmlp.py, run_lgb_10fold.py).

Feature parts, joined with "+":
  raw  the 21 columns (always included)
  A    one original-dataset lookup per column            (notebook FE_ORIGINAL)
  B    original lookup of the rating profile (+ count) and of the category combination   (notebook FE_ORIGINAL)
  R    route profile: per Flight Distance, the mean of every other column and the row count (build_features.py)
  U    auxiliary-task expected values of the 13 ratings (build_features.py)
  T    categorical twins of Age, Flight Distance and both delays (one category per exact value)
  D    auxiliary-task residuals: each rating minus its expected value (needs U's file)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parents[1])  # Paths below are relative to the project root.

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
               "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
               "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
LONG_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
SMOOTH = 20

train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y = train.pop("satisfaction").astype(str).str.lower().map({"true": 1, "false": 0}).to_numpy()
train_ids, test_ids = train.pop("id").to_numpy(), test.pop("id").to_numpy()
BASE_COLS = list(train.columns)
fold_id = pd.read_csv("data/folds_seed42.csv")["fold"].to_numpy()
N_TRAIN = len(train)


def _lookup(frame, original, cols):
    key = lambda d: d[cols].fillna(-1).astype(str).agg("|".join, axis=1) if len(cols) > 1 else d[cols[0]].fillna(-1)
    stats = original.groupby(key(original))["satisfaction"].agg(["sum", "count"])
    prior = original["satisfaction"].mean()
    total, count = key(frame).map(stats["sum"]).fillna(0), key(frame).map(stats["count"]).fillna(0)
    return ((total + prior * SMOOTH) / (count + SMOOTH)).astype("float32"), count.astype("float32")


def features(spec, fill_missing=False):
    """Train and test frames for a feature spec such as "A+B+R+U"; categories share one vocabulary.

    fill_missing: for models that reject NaN, a missing arrival delay becomes the departure delay plus a 0/1 flag.
    """
    parts = set(spec.split("+"))
    full = pd.concat([train, test], ignore_index=True)
    if fill_missing:
        arrival = full["Arrival Delay in Minutes"]
        full["arrival_missing"] = arrival.isna().astype("float32")
        full["Arrival Delay in Minutes"] = arrival.fillna(full["Departure Delay in Minutes"])
    extra = {}
    if parts & {"A", "B"}:
        original = pd.read_csv("data/original/original_all.csv")
        original["satisfaction"] = original["satisfaction"].astype(str).str.strip().str.lower().isin(
            ["satisfied", "true", "1"]).astype("int8")
        if "A" in parts:
            for c in BASE_COLS:
                extra[f"orig_te_{c}"] = _lookup(full, original, [c])[0]
        if "B" in parts:
            extra["orig_te_profile"], extra["orig_n_profile"] = _lookup(full, original, RATING_COLS + CAT_COLS)
            extra["orig_te_categories"] = _lookup(full, original, CAT_COLS)[0]
    if "R" in parts:
        route = pd.read_parquet("results/features/route.parquet")
        extra.update({c: route[c] for c in route.columns if c.startswith("route_")})
    if "U" in parts:
        aux = pd.read_parquet("results/features/aux_ev.parquet")
        extra.update({c: aux[c] for c in aux.columns if c != "id"})
    if "D" in parts:
        aux = pd.read_parquet("results/features/aux_ev.parquet")
        for c in RATING_COLS:
            extra[f"aux_res_{c}"] = (full[c] - aux[f"aux_ev_{c}"]).astype("float32")
    if "T" in parts:
        for c in LONG_COLS:
            extra[f"cat_{c}"] = full[c].fillna(-1).astype("int64").astype(str)
    full = pd.concat([full, pd.DataFrame(extra, index=full.index)], axis=1)
    for c in CAT_COLS + [f"cat_{c}" for c in LONG_COLS if "T" in parts]:
        full[c] = full[c].astype("category")
    return full.iloc[:N_TRAIN].reset_index(drop=True), full.iloc[N_TRAIN:].reset_index(drop=True)


def split_ids(nfold):
    """Fold of each training row: the fixed 5, or 10 that halve each fixed fold (StratifiedKFold(2, shuffle, seed 42))."""
    from sklearn.model_selection import StratifiedKFold
    if nfold == 5:
        return fold_id.copy()
    split = np.zeros(len(y), dtype=int)
    for k in range(5):
        rows = np.flatnonzero(fold_id == k)
        for half, (_, part) in enumerate(StratifiedKFold(2, shuffle=True, random_state=42).split(rows, y[rows])):
            split[rows[part]] = 2 * k + half
    return split


def save_member(experiment, model, oof, test_pred):
    Path("results/preds").mkdir(parents=True, exist_ok=True)
    np.save(f"results/preds/{experiment}_{model}_oof.npy", oof)
    np.save(f"results/preds/{experiment}_{model}_test.npy", test_pred)
