"""Experiment 1, class-balanced subsample: per target (and subset), keep every extractable sequence and an equal number
of non-extractable ones drawn uniformly at random, then fit OLS on every subset of the three features.
Repeated over several sampling seeds. Reads memorization_benchmark.parquet written by fit.py.
  splits     random    one stratified 80/20 split (AUC on the 20%)
             task      leave-one-task-out (mean of per-task AUCs, and AUC of the pooled held-out predictions)
  baselines  random    uniform random score
             xmodel    cross-model: mean percentile rank of -suffix NLL under the *other* nine perturbed models
                       (models that were themselves trained on the injected copies), used as the only feature
"""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATS = ["log_freq", "recency", "hardness"]
FEATSETS = [list(c) for k in (3, 2, 1) for c in combinations(FEATS, k)] + [["xmodel"], ["random"]]
KEY = ["task", "source_row"]

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
O = Path(a.out_dir)
bench = pd.read_parquet(O / "memorization_benchmark.parquet")
targets = bench.target.unique().tolist()
easy = pd.concat({t: pd.read_parquet(O / "scores" / f"{t}.parquet").set_index(KEY).sfx_nll_mean.rank(pct=True, ascending=False)
                  for t in targets}, axis=1)  # higher = lower suffix NLL under that perturbed model


def add_xmodel(d, t):
    others = [o for o in targets if o != t]
    d = d.copy()
    d["xmodel"] = easy.loc[pd.MultiIndex.from_frame(d[KEY]), others].mean(axis=1).to_numpy()
    return d


def balance(d, seed):
    pos, neg = d[d.extract == 1], d[d.extract == 0]
    out = pd.concat([pos, neg.sample(len(pos), random_state=seed)]).reset_index(drop=True)
    out["random"] = np.random.default_rng(seed).random(len(out))
    return out


def evaluate(d, feats, split):
    X, y = d[feats].to_numpy(), d["extract"].to_numpy()
    if split == "random":
        folds = [train_test_split(np.arange(len(d)), test_size=0.2, random_state=0, stratify=y)]
    else:
        folds = [(np.where(d.task != t)[0], np.where(d.task == t)[0]) for t in d.task.unique()]
    preds, te_all, fold_auc = [], [], []
    for tr, te in folds:
        if y[tr].min() == y[tr].max():
            continue
        p = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
        preds.append(p); te_all.append(te)
        if y[te].min() != y[te].max():
            fold_auc.append(roc_auc_score(y[te], p))
    te = np.concatenate(te_all); yt = y[te]
    return dict(n_test=int(len(te)), n_pos_test=int(yt.sum()),
                auc=float(roc_auc_score(yt, np.concatenate(preds))) if split == "random" else float(np.mean(fold_auc)),
                auc_pooled=float(roc_auc_score(yt, np.concatenate(preds))))


rows, subsets = [], []
for t, d in bench.groupby("target", sort=False):
    d = add_xmodel(d, t)
    for subset, dd in [("all", d), ("inserted", d[d.freq > 0])]:
        if dd.extract.sum() < 10:  # same threshold as fit.py
            continue
        for seed in range(a.seeds):
            db = balance(dd, seed)
            if seed == 0:
                subsets.append(db.assign(subset=subset))
            for split in ["random", "task"]:
                for feats in FEATSETS:
                    rows.append(dict(target=t, subset=subset, split=split, features="+".join(feats), seed=seed,
                                     n=len(db), n_pos=int(db.extract.sum()), **evaluate(db, feats, split)))
res = pd.DataFrame(rows)
res.to_csv(O / "results_balanced.csv", index=False)
pd.concat(subsets).to_parquet(O / "balanced_subset.parquet", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for subset in ["all", "inserted"]:
    for split in ["random", "task"]:
        r = res[(res.subset == subset) & (res.split == split)]
        t = r.pivot_table(index="target", columns="features", values="auc", sort=False)[["+".join(f) for f in FEATSETS]]
        sd = r[r.features == "+".join(FEATS)].groupby("target", sort=False).auc.std().rename("sd(3 feats)")
        st = r[r.features == "random"].groupby("target", sort=False)[["n", "n_pos", "n_test", "n_pos_test"]].first()
        print(f"\nsubset={subset} split={split}: AUC-ROC, mean over {a.seeds} seeds")
        print(st.join(t).join(sd).round(3).to_string())
