"""Memorization-prediction baseline: linear models on (frequency, recency, intrinsic hardness).

For each target (perturbed) Hubble model:
  label  extract   1 iff greedy decoding from the 32-token prefix reproduces the next 32 tokens (binary)
         dloss     nll(counterfactual standard model) - nll(target): log-likelihood gained from the injections
  feats  log_freq  log(1 + #insertions)                        (batch-info of that training run)
         recency   (T - last_step) / T, 1.0 if never inserted   (T = #training steps)
         hardness mean per-token NLL under reference standard models that never saw the injections,
                    excluding the target's counterfactual model (used in dloss)
Evaluated on (a) a random 80/20 split and (b) leave-one-task-out, on clean sequences, both on all of them
(incl. never-inserted controls) and on inserted-only (freq >= 1), where the problem is harder.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

STD = ["1b-100b_toks-standard", "1b-500b_toks-standard", "8b-100b_toks-standard", "8b-500b_toks-standard"]
TARGETS = {  # target -> (batch-info bench, counterfactual standard model, training steps)
    "1b-100b_toks-perturbed": ("perturbed-100b", "1b-100b_toks-standard", 48000),
    "8b-100b_toks-perturbed": ("perturbed-100b", "8b-100b_toks-standard", 48000),
    "1b-500b_toks-perturbed": ("perturbed-500b", "1b-500b_toks-standard", None),
    "8b-500b_toks-perturbed": ("perturbed-500b", "8b-500b_toks-standard", None),
    **{f"1b-100b_toks-injectrange_{r}": (f"perturbed-100b-insert-{b}", "1b-100b_toks-standard", 48000)
       for r, b in [("0_25", "0-25"), ("25_50", "25-50"), ("50_75", "50-75"), ("75_100", "75-100"),
                    ("0_50", "first50"), ("50_100", "last50")]},
}
FEATS = ["log_freq", "recency", "hardness"]

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
O = Path(a.out_dir)
scores = {m: pd.read_parquet(O / "scores" / f"{m}.parquet") for m in list(TARGETS) + STD
          if (O / "scores" / f"{m}.parquet").exists()}
key = ["task", "source_row"]


def build(target):
    bench, cf, T = TARGETS[target]
    refs = [m for m in STD if m != cf and m in scores]
    if target not in scores or cf not in scores or not refs:
        return None, refs
    b = pd.read_parquet(O / f"bench_{bench}.parquet")
    b = b[b.clean].drop(columns=["text"])
    T = T or int(pd.read_csv(O.parent / "batchinfo" / f"pi_{bench}.csv.gz", usecols=["insertion_batch"]).max().iloc[0]) + 1
    d = b.merge(scores[target][key + ["n_tok", "nll_mean", "extract", "sfx_argmax_acc"]], on=key)
    d = d.merge(scores[cf][key + ["nll_mean"]].rename(columns={"nll_mean": "nll_cf"}), on=key)
    for m in refs:
        d = d.merge(scores[m][key + ["nll_mean"]].rename(columns={"nll_mean": f"nll_{m}"}), on=key)
    d["log_freq"] = np.log1p(d["freq"])
    d["recency"] = np.where(d["freq"] > 0, (T - d["last_step"].fillna(0)) / T, 1.0)
    d["hardness"] = d[[f"nll_{m}" for m in refs]].mean(axis=1)
    d["dloss"] = d["nll_cf"] - d["nll_mean"]
    d["target"] = target
    return d, refs


def evaluate(d, feats, split):
    """split: 'random' or 'task' (leave-one-task-out). Returns metrics over held-out predictions."""
    X, y, yc = d[feats].to_numpy(), d["extract"].to_numpy(), d["dloss"].to_numpy()
    if y.sum() < 10:  # too few memorized sequences to split and fit (e.g. early-injection timing models)
        return dict(n_test=0, pos_rate=float(y.mean()), auc_ols=None, auc_logreg=None, ap_ols=None,
                    r2_ols_extract=None, r2_ols_dloss=None)
    if split == "random":
        folds = [train_test_split(np.arange(len(d)), test_size=0.2, random_state=0, stratify=y)]
    else:
        folds = [(np.where(d.task != t)[0], np.where(d.task == t)[0]) for t in d.task.unique()]
    p_ols, p_log, p_reg, te_all = [], [], [], []
    for tr, te in folds:
        if y[tr].min() == y[tr].max():
            continue
        ols = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr])
        log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        reg = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], yc[tr])
        p_ols.append(ols.predict(X[te])); p_log.append(log.predict_proba(X[te])[:, 1]); p_reg.append(reg.predict(X[te]))
        te_all.append(te)
    te = np.concatenate(te_all); yt, yct = y[te], yc[te]
    p_ols, p_log, p_reg = map(np.concatenate, (p_ols, p_log, p_reg))
    ok = yt.min() != yt.max()
    return dict(n_test=int(len(te)), pos_rate=float(yt.mean()),
                auc_ols=float(roc_auc_score(yt, p_ols)) if ok else None,
                auc_logreg=float(roc_auc_score(yt, p_log)) if ok else None,
                ap_ols=float(average_precision_score(yt, p_ols)) if ok else None,
                r2_ols_extract=float(r2_score(yt, p_ols)), r2_ols_dloss=float(r2_score(yct, p_reg)))


rows, coefs, frames = [], {}, []
for t in TARGETS:
    d, refs = build(t)
    if d is None:
        continue
    frames.append(d)
    for subset, dd in [("all", d), ("inserted", d[d.freq > 0])]:
        for split in ["random", "task"]:
            for feats in [FEATS, ["log_freq"], ["recency"], ["hardness"], ["log_freq", "recency"]]:
                r = evaluate(dd, feats, split)
                rows.append(dict(target=t, subset=subset, split=split, features="+".join(feats), n=len(dd), **r))
        if dd.extract.sum() < 10:
            continue
        full = make_pipeline(StandardScaler(), LinearRegression()).fit(dd[FEATS], dd["extract"])
        coefs[f"{t}|{subset}"] = dict(zip(FEATS, full[-1].coef_.round(4).tolist()),
                                      intercept=round(float(full[-1].intercept_), 4), refs=refs)
    # within a fixed duplication level frequency is constant, so this isolates recency and hardness
    for dup in [16, 64, 256]:
        dd = d[d.duplicates == dup]
        if dd.extract.sum() < 10:
            continue
        for feats in [["recency"], ["hardness"], ["recency", "hardness"]]:
            r = evaluate(dd, feats, "random")
            rows.append(dict(target=t, subset=f"dup={dup}", split="random", features="+".join(feats), n=len(dd), **r))
        full = make_pipeline(StandardScaler(), LinearRegression()).fit(dd[["recency", "hardness"]], dd["extract"])
        coefs[f"{t}|dup={dup}"] = dict(zip(["recency", "hardness"], full[-1].coef_.round(4).tolist()),
                                       intercept=round(float(full[-1].intercept_), 4))
res = pd.DataFrame(rows)
res.to_csv(O / "results.csv", index=False)
json.dump(coefs, open(O / "ols_coefficients.json", "w"), indent=1)
bench = pd.concat(frames)
bench.to_parquet(O / "memorization_benchmark.parquet", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(f"benchmark: {len(bench)} (target, sequence) instances, {bench.groupby('target').size().to_dict()}")
print(bench.groupby(["target", "duplicates"]).extract.mean().unstack().round(3).to_string())
print(bench[bench.target == bench.target.iloc[0]].pivot_table(index="task", columns="duplicates", values="extract").round(3).to_string())
print(res[res.subset.str.startswith("dup=")].pivot_table(index=["target", "subset"], columns="features", values="auc_ols").round(3).to_string())
print(res[res.features == "+".join(FEATS)].drop(columns="features").round(3).to_string(index=False))
print(res[(res.split == "random")].pivot_table(index=["target", "subset"], columns="features", values="auc_ols").round(3).to_string())
print(json.dumps(coefs, indent=1))
