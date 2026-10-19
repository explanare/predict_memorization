"""Experiment 3, frequency- and recency-controlled ablation: is intrinsic hardness predictive of extraction?

Test set: (target, duplicates) cells where 20-80% of inserted sequences are extractable. Within each cell, positives
and negatives are matched exactly on stratum = (task, bin of last_step --step-bin training steps wide): each stratum
keeps min(#pos, #neg) of each class (random draw), so the set is 50/50 and frequency, recency (up to the bin width)
and task carry no label information by construction. Bins must be narrow: most matchable sequences were last seen in
the final ~2% of training, and within that window a later last copy still predicts extraction.
Then OLS on every subset of the three standardized features (log_freq, recency, hardness), per cell and pooled.
  splits     group     5-fold GroupKFold over sequences (a sequence appears under several targets); AUC of the
                       pooled out-of-fold predictions
             task      leave-one-task-out (mean of per-task AUCs)
  baselines  random    uniform random score
             xmodel    cross-model: mean percentile rank of -suffix NLL under the *other* nine perturbed models
  hardness   every feature subset containing `hardness` is also fit with `hardness_sfx` in its place: the mean
             per-token NLL over the suffix (extraction window) only, averaged over the same reference models
  check      resid_recency_auc  AUC of later-last-copy within strata (all strata, ties = 0.5): ~0.5 = recency controlled
Reads memorization_benchmark.parquet written by fit.py.
"""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATS = ["log_freq", "recency", "hardness"]
FEATSETS = [list(c) for k in (3, 2, 1) for c in combinations(FEATS, k)] + [["xmodel"], ["random"]]
FEATSETS_SFX = [[("hardness_sfx" if f == "hardness" else f) for f in fs] for fs in FEATSETS if "hardness" in fs]
STD = ["1b-100b_toks-standard", "1b-500b_toks-standard", "8b-100b_toks-standard", "8b-500b_toks-standard"]
CF = {"1b-500b_toks-perturbed": "1b-500b_toks-standard", "8b-100b_toks-perturbed": "8b-100b_toks-standard",
      "8b-500b_toks-perturbed": "8b-500b_toks-standard"}  # all 1b-100b targets -> 1b-100b_toks-standard
KEY = ["task", "source_row"]

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--step-bin", type=int, default=100)
ap.add_argument("--tag", default="")
a = ap.parse_args()
O = Path(a.out_dir)
b = pd.read_parquet(O / "memorization_benchmark.parquet")
targets = b.target.unique().tolist()
easy = pd.concat({t: pd.read_parquet(O / "scores" / f"{t}.parquet").set_index(KEY).sfx_nll_mean.rank(pct=True, ascending=False)
                  for t in targets}, axis=1)  # higher = lower suffix NLL under that perturbed model
b = b[b.freq > 0].copy()
b["seq"] = b.task + ":" + b.source_row.astype(str)
b["rbin"] = (b.last_step // a.step_bin).astype(int)
rate = b.groupby(["target", "duplicates"]).extract.mean()
cells = rate[(rate > 0.2) & (rate < 0.8)].index
b = b.set_index(["target", "duplicates"]).loc[cells].reset_index()
b["xmodel"] = np.nan
for t in b.target.unique():
    i = b.target == t
    b.loc[i, "xmodel"] = easy.loc[pd.MultiIndex.from_frame(b.loc[i, KEY]), [o for o in targets if o != t]].mean(axis=1).to_numpy()

sfx = {m: pd.read_parquet(O / "scores" / f"{m}.parquet").set_index(KEY).sfx_nll_mean for m in STD}
b["hardness_sfx"] = np.nan  # like `hardness`, but the mean NLL over the suffix (extraction window) only
for t in b.target.unique():
    i = b.target == t
    refs = [m for m in STD if m != CF.get(t, "1b-100b_toks-standard")]
    idx = pd.MultiIndex.from_frame(b.loc[i, KEY])
    b.loc[i, "hardness_sfx"] = np.mean([sfx[m].reindex(idx).to_numpy() for m in refs], axis=0)


def matched(d, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in d.groupby(["target", "duplicates", "task", "rbin"]):
        pos, neg = g.index[g.extract == 1], g.index[g.extract == 0]
        k = min(len(pos), len(neg))
        if k:
            keep += list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, k, replace=False))
    out = d.loc[keep].reset_index(drop=True)
    out["random"] = rng.random(len(out))
    return out


def resid_recency(g):
    """Size-weighted within-stratum AUC of a later last copy (ties and constant strata count as 0.5)."""
    w = [(roc_auc_score(x.extract, -x.recency) if x.recency.nunique() > 1 else 0.5, len(x))
         for _, x in g.groupby(["target", "duplicates", "task", "rbin"])]
    return float(np.average([u for u, _ in w], weights=[n for _, n in w]))


def cv_auc(d, feats, split):
    X, y = d[feats].to_numpy(), d.extract.to_numpy()
    if split == "group":
        p = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(X, y, d.seq):
            p[te] = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
        return float(roc_auc_score(y, p))
    aucs = []
    for t in d.task.unique():
        tr, te = np.where(d.task != t)[0], np.where(d.task == t)[0]
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr]).predict(X[te])))
    return float(np.mean(aucs))


rows, sets = [], []
for seed in range(a.seeds):
    m = matched(b, seed)
    if seed == 0:
        sets.append(m)
    groups = [("pooled", m)] + [(f"{t}|dup={k}", g.reset_index(drop=True)) for (t, k), g in m.groupby(["target", "duplicates"])]
    for name, g in groups:
        base = dict(seed=seed, cell=name, n=len(g), n_pos=int(g.extract.sum()), n_tasks=g.task.nunique(),
                    resid_recency_auc=resid_recency(g))
        for split in ["group", "task"]:
            for feats in FEATSETS + FEATSETS_SFX:
                rows.append(dict(base, split=split, features="+".join(feats), auc=cv_auc(g, feats, split)))
res = pd.DataFrame(rows)
res.to_csv(O / f"results_hardness{a.tag}.csv", index=False)
pd.concat(sets).to_parquet(O / f"hardness_ablation{a.tag}.parquet", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("cells (inserted, 20-80% extractable before matching):"); print(rate.loc[cells].round(3).to_string())
for split in ["group", "task"]:
    r = res[res.split == split]
    t = r.pivot_table(index="cell", columns="features", values="auc", sort=False)[["+".join(f) for f in FEATSETS + FEATSETS_SFX]]
    sd = r[r.features == "+".join(FEATS)].groupby("cell", sort=False).auc.std().rename("sd(3 feats)")
    st = r[r.features == "random"].groupby("cell", sort=False)[["n", "n_pos", "n_tasks", "resid_recency_auc"]].mean()
    print(f"\nsplit={split}: AUC-ROC, mean over {a.seeds} seeds")
    print(st.join(t).join(sd).round(3).to_string())
