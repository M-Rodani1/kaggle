"""Build label-free feature tables over train + test and save them in results/features/ (train rows first, then test).

route   For every Flight Distance value (a route), the mean of every other column and the route's row count; value
        counts of Age, Flight Distance and both delays; Flight Distance // 10, // 100, % 100, % 1000 and digit count.
aux     Auxiliary-task expected values: for each of the 13 ratings, E[rating | the other 20 columns] = sum_k k P(k) from
        a multiclass LightGBM, cross-fitted in 5 folds over train + test so no row is scored by a model that saw it.

Neither uses a label, so both are safe in every competition fold (they are transductive: test features are used).
Recipe from the public S6E10 "route profile" and "aux EV" members (golem library manifest, sachith7's aux_ev.parquet).
Usage: python3 scripts/build_features.py [route] [aux]
"""
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

os.chdir(Path(__file__).resolve().parents[1])  # Paths below are relative to the project root.

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
               "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
               "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
OUT = Path("results/features")
OUT.mkdir(parents=True, exist_ok=True)

train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
FEATURES = [c for c in test.columns if c != "id"]
full = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
ids = pd.concat([train["id"], test["id"]], ignore_index=True)
for c in CAT_COLS:
    full[c] = full[c].astype("category").cat.codes.astype("int16")


def build_route():
    fd = full["Flight Distance"]
    group = full.groupby(fd)
    out = pd.DataFrame(index=full.index)
    for c in FEATURES:
        if c != "Flight Distance":
            out[f"route_mean_{c}"] = fd.map(group[c].mean()).astype("float32")
    out["route_n"] = fd.map(group.size()).astype("float32")
    for c in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        out[f"count_{c}"] = full[c].map(full[c].value_counts()).fillna(0).astype("float32")
    out["fd_div10"], out["fd_div100"] = (fd // 10).astype("float32"), (fd // 100).astype("float32")
    out["fd_mod100"], out["fd_mod1000"] = (fd % 100).astype("float32"), (fd % 1000).astype("float32")
    out["fd_digits"] = fd.astype(str).str.len().astype("float32")
    out.to_parquet(OUT / "route.parquet")
    print(f"route: {out.shape[1]} columns -> {OUT / 'route.parquet'}", flush=True)


def build_aux():
    params = dict(objective="multiclass", num_class=6, learning_rate=0.1, num_leaves=63, min_child_samples=50,
                  feature_fraction=0.9, verbose=-1, num_threads=os.cpu_count(), seed=0)
    out = pd.DataFrame(index=full.index)
    folds = list(KFold(5, shuffle=True, random_state=42).split(full))
    for rating in RATING_COLS:
        started, cols = time.time(), [c for c in FEATURES if c != rating]
        x, target = full[cols], full[rating].to_numpy()
        expected = np.zeros(len(full), dtype="float32")
        for fit_idx, score_idx in folds:
            model = lgb.train(params, lgb.Dataset(x.iloc[fit_idx], target[fit_idx], categorical_feature=CAT_COLS), 300)
            expected[score_idx] = model.predict(x.iloc[score_idx]) @ np.arange(6)
        out[f"aux_ev_{rating}"] = expected
        print(f"aux {rating}: corr with actual {np.corrcoef(expected, target)[0, 1]:.3f} | {time.time() - started:.0f}s",
              flush=True)
    out.insert(0, "id", ids)
    out.to_parquet(OUT / "aux_ev.parquet")
    print(f"aux: {out.shape[1]} columns -> {OUT / 'aux_ev.parquet'}", flush=True)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["route", "aux"]
    if "route" in parts:
        build_route()
    if "aux" in parts:
        build_aux()
