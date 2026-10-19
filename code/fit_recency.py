"""Experiment 2, frequency-controlled, recency-varying ablation: the same sequence under different insertion schedules.

The six injectrange models (1B, 100B tokens) see every inserted sequence with exactly the same number of copies and
differ only in *when* the copies are inserted. For every sequence that is extractable under at least one run of a pool
and not extractable under at least one other, draw one positive run and one negative run at random: each such
sequence contributes one positive and one negative example. Frequency, sequence content (hence task and hardness),
model size and token count are identical within a pair; only the insertion schedule (recency) differs.
  pools      quarters  injectrange 0_25, 25_50, 50_75, 75_100 (disjoint 12k-step windows of equal width)
             all6      + 0_50, 50_100 (24k-step windows)
             late      50_100 vs 75_100 (both windows end at the end of training)
  splits     group     5-fold GroupKFold over sequences (both examples of a pair on the same side); AUC of the
                       pooled out-of-fold predictions, and within-pair AUC: fraction of pairs whose positive scores
                       higher than its negative (ties = 0.5), from the same out-of-fold predictions
             task      leave-one-task-out (mean of per-task AUCs)
  baselines  random      uniform random score
             window_end  end of the run's insertion window as a fraction of training (0.25, 0.5, 0.75, 1.0)
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
FEATSETS = [list(c) for k in (3, 2, 1) for c in combinations(FEATS, k)] + [["window_end"], ["random"]]
QUARTERS = ["0_25", "25_50", "50_75", "75_100"]
POOLS = {"quarters": QUARTERS, "all6": QUARTERS + ["0_50", "50_100"], "late": ["50_100", "75_100"]}
WINDOW_END = {"0_25": 0.25, "25_50": 0.5, "50_75": 0.75, "75_100": 1.0, "0_50": 0.5, "50_100": 1.0}

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
O = Path(a.out_dir)
b = pd.read_parquet(O / "memorization_benchmark.parquet")
b = b[b.target.str.contains("injectrange") & (b.freq > 0)].copy()
b["run"] = b.target.str.replace("1b-100b_toks-injectrange_", "", regex=False)
b["seq"] = b.task + ":" + b.source_row.astype(str)
b["window_end"] = b.run.map(WINDOW_END)
T = int(b.last_step.max()) + 1
b["steps_since_last"] = T - 1 - b.last_step


def pairs(d, seed):
    """One random positive run and one random negative run for every sequence with mixed outcomes."""
    rng = np.random.default_rng(seed)
    d = d.sample(frac=1, random_state=seed)  # random order -> first row per (seq, class) is a random run
    mixed = d.groupby("seq").extract.transform(lambda x: x.nunique() == 2)
    out = d[mixed].groupby(["seq", "extract"], sort=False).head(1).sort_values(["seq", "extract"]).reset_index(drop=True)
    out["random"] = rng.random(len(out))
    return out


def evaluate(d, feats):
    X, y = d[feats].to_numpy(), d.extract.to_numpy()
    p = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, y, d.seq):
        p[te] = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    w = pd.DataFrame(dict(seq=d.seq, y=y, p=p)).pivot_table(index="seq", columns="y", values="p")
    within = float(np.mean(np.where(w[1] > w[0], 1.0, np.where(w[1] == w[0], 0.5, 0.0))))
    task = []
    for t in d.task.unique():
        tr, te = np.where(d.task != t)[0], np.where(d.task == t)[0]
        task.append(roc_auc_score(y[te], make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr]).predict(X[te])))
    return dict(auc_group=float(roc_auc_score(y, p)), auc_within_pair=within, auc_task=float(np.mean(task)))


rows, sets = [], []
for pool, runs in POOLS.items():
    for seed in range(a.seeds):
        m = pairs(b[b.run.isin(runs)], seed)
        if seed == 0:
            sets.append(m.assign(pool=pool))
        for name, g in [("pooled", m)] + [(f"dup={k}", g.reset_index(drop=True)) for k, g in m.groupby("duplicates")]:
            for feats in FEATSETS:
                rows.append(dict(pool=pool, seed=seed, stratum=name, n=len(g), n_pairs=g.seq.nunique(),
                                 n_tasks=g.task.nunique(), features="+".join(feats), **evaluate(g, feats)))
res = pd.DataFrame(rows)
res.to_csv(O / "results_recency.csv", index=False)
sets = pd.concat(sets)
sets.to_parquet(O / "recency_ablation.parquet", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print("seed-0 pairs: examples per run and class")
print(sets.pivot_table(index=["pool", "run"], columns="extract", values="seq", aggfunc="count", fill_value=0).to_string())
print("\nseed-0 pairs per pool x copies (n_pairs; #tasks; steps since last copy median [IQR] over both examples)")
for (pool, k), g in sets.groupby(["pool", "duplicates"], sort=False):
    s = g.steps_since_last
    print(f"  {pool:9s} dup={k:<4d} pairs={g.seq.nunique():5d} tasks={g.task.nunique():2d} "
          f"since-last {int(s.median())} [{int(s.quantile(.25))}-{int(s.quantile(.75))}]")
order = ["+".join(f) for f in FEATSETS]
for metric in ["auc_group", "auc_within_pair", "auc_task"]:
    for pool in POOLS:
        r = res[res.pool == pool]
        t = r.pivot_table(index="features", columns="stratum", values=metric).reindex(order)
        t = t[[c for c in ["dup=1", "dup=4", "dup=16", "dup=64", "dup=256", "pooled"] if c in t.columns]]
        t.loc["sd (3 features)"] = r[r.features == order[0]].groupby("stratum")[metric].std()
        print(f"\n{metric}, pool={pool} (mean over {a.seeds} seeds)"); print(t.round(3).to_string())
