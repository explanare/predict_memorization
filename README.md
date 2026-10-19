# Predicting memorization

> Given a dataset in a fixed order, can you predict which specific sequences will be memorized by a large model trained on that data without training the full model? (Biderman et al. 2026)

Is memorization of individual sequence predictable? This repo contains a prilimary experiment using [Hubble](https://huggingface.co/collections/allegrolab/hubble-datasets), a suite of models trained with injected sequences to measure memorization. The task is simple -- given sequences injected into or holdout from model pretraining, predict which ones the final model memorizes.

For prediction, we consider a simple regression model using three features of a sequence studied in the literature: frequency (how many times a sequence occurred in training), recency (what is the last training step at which the sequence occurs), and intrinsic hardness (how hard to predict given an LLM trained without this sequence).


## Common setup

**Sequences.** All 16 Hubble perturbation datasets (`data/*_nodup.jsonl.gz`, one row per unique sequence, including
never-inserted controls), restricted to `clean` sequences (exposure count unambiguous): 80,504 sequences, of which
41,387 were inserted at least once (1, 4, 16, 64 or 256 copies) and 39,117 are controls (0 copies).
Exposure per sequence and training run (`freq`, `first_step`, `last_step`) comes from the Hubble batch-info files.

**Target models.** Perturbed Hubble models; the label is computed separately for each:

| short name | model | training steps T |
|---|---|---|
| 1b-100b, 8b-100b | `{1b,8b}-100b_toks-perturbed` | 48,000 |
| 1b-500b, 8b-500b | `{1b,8b}-500b_toks-perturbed` | 238,500 |
| ir50_75, ir75_100, ir50_100 | `1b-100b_toks-injectrange_{50_75,75_100,50_100}`: copies inserted only in that % range of training | 48,000 |

`injectrange_{0_25,25_50,0_50}` have 2, 3 and 6 positives in total and are excluded from Experiments 1 and 3;
Experiment 2 uses all six `injectrange` models.

**Label.** `extract` = 1 iff greedy decoding from the first P = min(32, n/2) tokens reproduces the next min(32, n−P)
tokens exactly (computed with teacher forcing: the target token is the argmax at every suffix position).

**Features.** Each is standardized before fitting:
1. `log_freq` = log(1 + #copies inserted)
2. `recency` = (T − last_step) / T, 1.0 if never inserted
3. `hardness` (intrinsic hardness) = mean per-token NLL of the whole sequence, averaged over the standard (non-perturbed) Hubble models,
   excluding the target's counterfactual standard model (same size and token count)

**Memorization prediction model.** Ordinary least-squares linear regression on the 0/1 label (`StandardScaler` → `LinearRegression`);
the regression output is the ranking score for AUC-ROC. Fit on all 7 non-empty subsets of the three features.

**Metric.** All results are AUC-ROC on held-out examples: under the `random` split (Experiment 1) and the `group`
split (Experiments 2 and 3), the held-out predictions of all folds are pooled into a single AUC-ROC, which measures how
well the model ranks new sequences from tasks seen in training. Under the leave-one-task-out (`task`) split, AUC-ROC is
computed within each held-out task and averaged over tasks with equal weight, which measures whether the model
transfers to a task it was not trained on, comparing only sequences of the same task with each other.

**Baselines.**
- `random`: uniform random score, evaluated with the same splits.
- `cross-model` (Experiments 1 and 3; Experiment 2 has its own baselines): for each sequence, the percentile rank of its suffix NLL (lower = higher score) under each of the
  *other* nine perturbed models, which were themselves trained on the injected copies, averaged over those models.
  Used as the only feature, with the same splits.

All numbers are means over 5 sampling seeds. `sd` is the standard deviation over seeds of the 3-feature AUC.

---

## Experiment 1: subsample of all sequences

**Sampling.** For each target, keep every positive and draw the same number of negatives uniformly at random from that
target's negatives (`all`: inserted sequences and controls). No other property is controlled.

**Splits.**
- `random`: one stratified 80/20 split (`random_state=0`); AUC on the 20% test set.
- `task`: leave-one-task-out over the 16 tasks; mean of the 16 per-task AUCs.

**Dataset.** Control counts among negatives are from seed 0; all other counts are the same for every seed.

| target | #examples | #pos | #neg | controls among pos / neg | test set, random split (#pos) |
|---|---|---|---|---|---|
| 1b-100b | 7,052 | 3,526 | 3,526 | 301 / 1,798 | 1,411 (705) |
| 8b-100b | 20,794 | 10,397 | 10,397 | 497 / 5,673 | 4,159 (2,079) |
| 1b-500b | 2,706 | 1,353 | 1,353 | 327 / 696 | 542 (271) |
| 8b-500b | 9,072 | 4,536 | 4,536 | 485 / 2,320 | 1,815 (907) |
| ir50_75 | 300 | 150 | 150 | 66 / 72 | 60 (30) |
| ir75_100 | 9,814 | 4,907 | 4,907 | 318 / 2,574 | 1,963 (981) |
| ir50_100 | 9,590 | 4,795 | 4,795 | 333 / 2,509 | 1,918 (959) |

**AUC-ROC, random split.**

| features | 1b-100b | 8b-100b | 1b-500b | 8b-500b | ir50_75 | ir75_100 | ir50_100 |
|---|---|---|---|---|---|---|---|
| log_freq + recency + hardness | 0.923 | 0.934 | 0.778 | 0.883 | 0.784 | 0.926 | 0.930 |
| log_freq + recency | 0.904 | 0.932 | 0.733 | 0.871 | 0.551 | 0.910 | 0.915 |
| log_freq + hardness | 0.921 | 0.932 | 0.772 | 0.883 | 0.792 | 0.923 | 0.926 |
| recency + hardness | 0.875 | 0.908 | 0.710 | 0.848 | 0.784 | 0.843 | 0.870 |
| log_freq | 0.907 | 0.926 | 0.735 | 0.874 | 0.624 | 0.913 | 0.916 |
| recency | 0.895 | 0.910 | 0.727 | 0.864 | 0.600 | 0.894 | 0.898 |
| hardness | 0.567 | 0.515 | 0.563 | 0.544 | 0.770 | 0.537 | 0.538 |
| *cross-model baseline* | 0.867 | 0.800 | 0.904 | 0.874 | 0.954 | 0.862 | 0.862 |
| *random baseline* | 0.489 | 0.504 | 0.510 | 0.499 | 0.520 | 0.503 | 0.506 |
| sd (3 features) | 0.002 | 0.003 | 0.016 | 0.004 | 0.058 | 0.002 | 0.002 |

**AUC-ROC, leave-one-task-out (mean over held-out tasks).**

| features | 1b-100b | 8b-100b | 1b-500b | 8b-500b | ir50_75 | ir75_100 | ir50_100 |
|---|---|---|---|---|---|---|---|
| log_freq + recency + hardness | 0.945 | 0.953 | 0.846 | 0.930 | 0.356 | 0.949 | 0.944 |
| log_freq + recency | 0.946 | 0.956 | 0.872 | 0.936 | 0.560 | 0.953 | 0.948 |
| log_freq + hardness | 0.948 | 0.947 | 0.855 | 0.933 | 0.336 | 0.953 | 0.946 |
| recency + hardness | 0.901 | 0.928 | 0.784 | 0.900 | 0.284 | 0.877 | 0.895 |
| log_freq | 0.950 | 0.951 | 0.881 | 0.940 | 0.525 | 0.957 | 0.951 |
| recency | 0.937 | 0.936 | 0.872 | 0.930 | 0.501 | 0.942 | 0.935 |
| hardness | 0.489 | 0.486 | 0.434 | 0.481 | 0.284 | 0.486 | 0.478 |
| *cross-model baseline* | 0.897 | 0.811 | 0.924 | 0.893 | 0.962 | 0.887 | 0.892 |
| *random baseline* | 0.500 | 0.503 | 0.527 | 0.505 | 0.518 | 0.497 | 0.498 |
| sd (3 features) | 0.002 | 0.001 | 0.009 | 0.003 | 0.040 | 0.002 | 0.002 |

The same tables restricted to inserted sequences (controls removed before balancing) are in
`out/fit_balanced_report.txt` (`subset=inserted`).

---

## Experiment 2: ablation test set with frequency controlled and recency varying

**Sampling.** Uses the six `1b-100b_toks-injectrange_*` models, which see every inserted sequence with exactly the same
number of copies and differ only in *when* the copies are inserted (0_25, 25_50, 50_75, 75_100, 0_50, 50_100: the
% range of training). Within a pool of runs, take every inserted sequence that is a positive under at least one run
and a negative under at least one other. For each such sequence, draw one positive run and one negative run at random.
Each sequence therefore contributes a pair of examples: one positive and one negative.

**Controlled properties.** Within a pair: number of copies, sequence content (hence task and `hardness`), model size
and training tokens are identical. Only the insertion schedule, and with it `recency`, differs.

**Pools.**
- `quarters`: 0_25, 25_50, 50_75, 75_100 (disjoint 12,000-step windows of equal width)
- `all6`: the quarters plus 0_50 and 50_100 (24,000-step windows)
- `late`: 50_100 and 75_100 only (both windows end at the end of training). With two runs, each sequence has exactly one
  possible pair, so the set is the same for every seed.

**Splits.**
- `group`: 5-fold cross-validation grouped by sequence (both examples of a pair on the same side); AUC-ROC of the
  pooled out-of-fold predictions.

**Baselines.**
- `random`: uniform random score.
- `window-end`: end of the run's insertion window as a fraction of training (0.25, 0.5, 0.75 or 1.0), used as the only
  feature.

The cross-model baseline of Experiments 1 and 3 is not used here: it averages over the other perturbed models, which
include the partner run of the same pair.

**Dataset.** Statistics from the seed-0 pairs.

| pool | #copies | #pairs | #examples (pos / neg) | #tasks | steps since last copy, median (IQR) |
|---|---|---|---|---|---|
| quarters | 1 | 164 | 328 (164 / 164) | 7 | 14,399 (5,567–29,333) |
| quarters | 4 | 212 | 424 (212 / 212) | 5 | 12,817 (1,921–26,837) |
| quarters | 16 | 599 | 1,198 (599 / 599) | 11 | 12,035 (373–24,520) |
| quarters | 64 | 2,064 | 4,128 (2,064 / 2,064) | 12 | 12,000 (118–24,128) |
| quarters | 256 | 1,573 | 3,146 (1,573 / 1,573) | 16 | 12,000 (32–24,034) |
| **quarters** | all | **4,612** | **9,224 (4,612 / 4,612)** | 16 | 12,003 (92–24,104) |
| all6 | 1 | 253 | 506 (253 / 253) | 7 | 15,901 (6,269–29,923) |
| all6 | 4 | 288 | 576 (288 / 288) | 7 | 9,746 (1,918–25,830) |
| all6 | 16 | 825 | 1,650 (825 / 825) | 12 | 1,803 (391–24,402) |
| all6 | 64 | 2,388 | 4,776 (2,388 / 2,388) | 12 | 867 (151–24,171) |
| all6 | 256 | 1,628 | 3,256 (1,628 / 1,628) | 16 | 412 (45–24,040) |
| **all6** | all | **5,382** | **10,764 (5,382 / 5,382)** | 16 | 4,187 (136–24,127) |
| late | 1 | 181 | 362 (181 / 181) | 5 | 7,395 (3,387–11,121) |
| late | 4 | 192 | 384 (192 / 192) | 6 | 2,157 (919–4,294) |
| late | 16 | 570 | 1,140 (570 / 570) | 12 | 601 (230–1,369) |
| late | 64 | 648 | 1,296 (648 / 648) | 12 | 208 (83–439) |
| late | 256 | 96 | 192 (96 / 96) | 15 | 47 (20–98) |
| **late** | all | **1,687** | **3,374 (1,687 / 1,687)** | 16 | 427 (140–1,489) |

Examples per run (positive / negative), seed 0:

| pool | 0_25 | 25_50 | 50_75 | 75_100 | 0_50 | 50_100 |
|---|---|---|---|---|---|---|
| quarters | 1 / 1,566 | 2 / 1,517 | 48 / 1,521 | 4,561 / 8 | — | — |
| all6 | 1 / 1,279 | 1 / 1,296 | 38 / 1,228 | 2,787 / 153 | 3 / 1,255 | 2,552 / 171 |
| late | — | — | — | 907 / 780 | — | 780 / 907 |

**AUC-ROC, group split, pool `quarters`.**

| features | 1 copy | 4 | 16 | 64 | 256 | pooled |
|---|---|---|---|---|---|---|
| log_freq + recency + hardness | 0.975 | 0.978 | 0.996 | 1.000 | 1.000 | 0.997 |
| log_freq + recency | 0.975 | 0.977 | 0.997 | 1.000 | 1.000 | 0.997 |
| log_freq + hardness | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| recency + hardness | 0.975 | 0.978 | 0.996 | 1.000 | 1.000 | 0.997 |
| log_freq | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| recency | 0.975 | 0.977 | 0.997 | 1.000 | 1.000 | 0.996 |
| hardness | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| *window-end baseline* | 0.965 | 0.974 | 0.996 | 1.000 | 1.000 | 0.997 |
| *random baseline* | 0.473 | 0.473 | 0.495 | 0.490 | 0.493 | 0.495 |
| sd (3 features) | 0.003 | 0.004 | 0.001 | 0.000 | 0.000 | 0.000 |

**AUC-ROC, group split, pool `all6`.**

| features | 1 copy | 4 | 16 | 64 | 256 | pooled |
|---|---|---|---|---|---|---|
| log_freq + recency + hardness | 0.914 | 0.928 | 0.945 | 0.978 | 0.994 | 0.966 |
| log_freq + recency | 0.914 | 0.928 | 0.945 | 0.978 | 0.995 | 0.966 |
| log_freq + hardness | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| recency + hardness | 0.914 | 0.928 | 0.945 | 0.978 | 0.994 | 0.976 |
| log_freq | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| recency | 0.914 | 0.928 | 0.945 | 0.978 | 0.995 | 0.978 |
| hardness | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| *window-end baseline* | 0.909 | 0.913 | 0.919 | 0.971 | 0.994 | 0.966 |
| *random baseline* | 0.491 | 0.488 | 0.502 | 0.499 | 0.488 | 0.495 |
| sd (3 features) | 0.010 | 0.016 | 0.006 | 0.001 | 0.002 | 0.001 |

**AUC-ROC, group split, pool `late`.**

| features | 1 copy | 4 | 16 | 64 | 256 | pooled |
|---|---|---|---|---|---|---|
| log_freq + recency + hardness | 0.525 | 0.591 | 0.664 | 0.616 | 0.536 | 0.574 |
| log_freq + recency | 0.524 | 0.592 | 0.664 | 0.614 | 0.535 | 0.574 |
| log_freq + hardness | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| recency + hardness | 0.525 | 0.591 | 0.664 | 0.616 | 0.536 | 0.573 |
| log_freq | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| recency | 0.524 | 0.592 | 0.664 | 0.614 | 0.535 | 0.581 |
| hardness | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| *window-end baseline* | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| *random baseline* | 0.515 | 0.490 | 0.484 | 0.476 | 0.486 | 0.494 |
| sd (3 features) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

---

## Experiment 3: ablation test set with frequency and recency controlled

**Sampling.**
1. Restrict to inserted sequences (≥ 1 copy).
2. Keep the (target, #copies) cells in which 20–80% of sequences are positive. Within a cell, every sequence has
   exactly the same number of copies.
3. Within each cell, group sequences into strata by (task, `last_step` // 100), i.e. same task and last copy seen
   within the same 100-step window. In each stratum keep min(#pos, #neg) of each class, drawn at random.

**Controlled properties.** Number of copies (exact), step of the last copy (within 100 steps), task. The resulting set is
exactly 50/50 within every stratum. Check: the size-weighted within-stratum AUC of a later last copy is reported as
`residual recency AUC` (0.5 = no recency signal left).

**Hardness measures.** Every feature subset that contains `hardness` is fit twice, once per measure. Both are averaged
over the same reference models (the standard models other than the target's counterfactual):
- `full sequence`: `hardness` as defined in the common setup, the mean per-token NLL over the whole sequence
- `suffix`: the mean per-token NLL over the suffix only, i.e. the min(32, n−P) tokens that the label requires the model
  to reproduce (column `hardness_sfx`)

**Splits.**
- `group`: 5-fold cross-validation grouped by sequence, so a sequence that appears under several targets is never
  in both train and test; AUC of the pooled out-of-fold predictions.
- `task`: leave-one-task-out; mean of the per-task AUCs.

Splits are applied within each cell, and to all cells pooled. Within a cell, `log_freq` is constant.

**Dataset.** Statistics from the seed-0 matched set.

| cell | #copies | positive rate before matching | #examples | #pos / #neg | #tasks | #strata | steps since last copy, median (IQR) | residual recency AUC |
|---|---|---|---|---|---|---|---|---|
| 1b-100b | 64 | 0.421 | 1,360 | 680 / 680 | 12 | 185 | 497 (215–892) | 0.512 |
| 8b-100b | 16 | 0.578 | 2,428 | 1,214 / 1,214 | 15 | 622 | 2,417 (1,078–4,096) | 0.485 |
| 1b-500b | 256 | 0.364 | 704 | 352 / 352 | 12 | 163 | 495 (191–956) | 0.494 |
| 8b-500b | 64 | 0.561 | 878 | 439 / 439 | 11 | 329 | 2,365 (1,071–4,204) | 0.495 |
| ir75_100 | 64 | 0.735 | 1,206 | 603 / 603 | 12 | 81 | 140 (61–260) | 0.523 |
| ir50_100 | 64 | 0.735 | 1,070 | 535 / 535 | 12 | 140 | 317 (136–583) | 0.498 |
| **pooled** | — | — | 7,646 | 3,823 / 3,823 | 15 | 1,520 | — | 0.500 |

The pooled set contains 5,504 unique sequences.

**AUC-ROC, group split.**

| features | hardness | 1b-100b | 8b-100b | 1b-500b | 8b-500b | ir75_100 | ir50_100 | pooled |
|---|---|---|---|---|---|---|---|---|
| log_freq + recency + hardness | full sequence | 0.470 | 0.502 | 0.492 | 0.451 | 0.524 | 0.518 | 0.520 |
| log_freq + recency + hardness | suffix | 0.581 | 0.609 | 0.553 | 0.576 | 0.614 | 0.618 | 0.604 |
| log_freq + recency | — | 0.451 | 0.469 | 0.438 | 0.448 | 0.466 | 0.454 | 0.479 |
| log_freq + hardness | full sequence | 0.482 | 0.510 | 0.503 | 0.467 | 0.525 | 0.529 | 0.522 |
| log_freq + hardness | suffix | 0.584 | 0.610 | 0.559 | 0.582 | 0.615 | 0.622 | 0.604 |
| recency + hardness | full sequence | 0.470 | 0.502 | 0.492 | 0.451 | 0.524 | 0.518 | 0.521 |
| recency + hardness | suffix | 0.581 | 0.609 | 0.553 | 0.576 | 0.614 | 0.618 | 0.604 |
| log_freq | — | 0.463 | 0.478 | 0.455 | 0.471 | 0.471 | 0.470 | 0.483 |
| recency | — | 0.451 | 0.469 | 0.438 | 0.448 | 0.466 | 0.454 | 0.482 |
| hardness | full sequence | 0.482 | 0.510 | 0.503 | 0.467 | 0.525 | 0.529 | 0.523 |
| hardness | suffix | 0.584 | 0.610 | 0.559 | 0.582 | 0.615 | 0.622 | 0.604 |
| *cross-model baseline* | — | 0.608 | 0.636 | 0.565 | 0.598 | 0.642 | 0.648 | 0.617 |
| *random baseline* | — | 0.462 | 0.476 | 0.463 | 0.489 | 0.493 | 0.466 | 0.485 |
| sd (3 features) | full sequence | 0.004 | 0.009 | 0.009 | 0.022 | 0.010 | 0.023 | 0.004 |
| sd (3 features) | suffix | 0.009 | 0.004 | 0.006 | 0.010 | 0.005 | 0.009 | 0.002 |

**AUC-ROC, leave-one-task-out (mean over held-out tasks).**

| features | hardness | 1b-100b | 8b-100b | 1b-500b | 8b-500b | ir75_100 | ir50_100 | pooled |
|---|---|---|---|---|---|---|---|---|
| log_freq + recency + hardness | full sequence | 0.509 | 0.540 | 0.540 | 0.493 | 0.543 | 0.559 | 0.529 |
| log_freq + recency + hardness | suffix | 0.563 | 0.620 | 0.548 | 0.572 | 0.619 | 0.617 | 0.592 |
| log_freq + recency | — | 0.510 | 0.497 | 0.509 | 0.500 | 0.508 | 0.497 | 0.500 |
| log_freq + hardness | full sequence | 0.511 | 0.542 | 0.540 | 0.493 | 0.546 | 0.559 | 0.529 |
| log_freq + hardness | suffix | 0.564 | 0.620 | 0.551 | 0.574 | 0.619 | 0.618 | 0.591 |
| recency + hardness | full sequence | 0.509 | 0.540 | 0.540 | 0.493 | 0.543 | 0.559 | 0.529 |
| recency + hardness | suffix | 0.563 | 0.620 | 0.548 | 0.572 | 0.619 | 0.617 | 0.591 |
| log_freq | — | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| recency | — | 0.510 | 0.497 | 0.509 | 0.500 | 0.508 | 0.497 | 0.499 |
| hardness | full sequence | 0.511 | 0.542 | 0.540 | 0.493 | 0.546 | 0.559 | 0.529 |
| hardness | suffix | 0.564 | 0.620 | 0.551 | 0.574 | 0.619 | 0.618 | 0.592 |
| *cross-model baseline* | — | 0.598 | 0.660 | 0.557 | 0.604 | 0.658 | 0.656 | 0.613 |
| *random baseline* | — | 0.484 | 0.494 | 0.512 | 0.502 | 0.503 | 0.485 | 0.494 |
| sd (3 features) | full sequence | 0.009 | 0.005 | 0.011 | 0.036 | 0.008 | 0.013 | 0.003 |
| sd (3 features) | suffix | 0.008 | 0.009 | 0.006 | 0.007 | 0.007 | 0.008 | 0.005 |

---

## Reproducing

```bash
code/run_all.sh                              # build per-run exposure tables (code/build_benchmark.py) and
                                             # score every sequence under all 14 models (code/score.py)
python code/fit.py            --out-dir out  # joins labels + features -> out/memorization_benchmark.parquet
python code/fit_balanced.py   --out-dir out  # Experiment 1
python code/fit_recency.py    --out-dir out  # Experiment 2
python code/fit_hardness.py   --out-dir out  # Experiment 3 (--step-bin sets the last_step bin width, default 100)
```

| file | content |
|---|---|
| `code/build_benchmark.py` | sequences + exposure (`freq`, `first_step`, `last_step`) per batch-info file → `out/bench_<run>.parquet` |
| `code/score.py` | teacher-forced NLL, suffix NLL, suffix argmax accuracy and `extract` label per sequence → `out/scores/<model>.parquet` |
| `code/fit.py` | features and labels for all targets → `out/memorization_benchmark.parquet` |
| `code/fit_balanced.py` | Experiment 1 → `out/results_balanced.csv` (per seed), `out/balanced_subset.parquet` (seed-0 sets), `out/fit_balanced_report.txt` |
| `code/fit_recency.py` | Experiment 2 → `out/results_recency.csv` (per seed), `out/recency_ablation.parquet` (seed-0 pairs), `out/fit_recency_report.txt` |
| `code/fit_hardness.py` | Experiment 3 → `out/results_hardness.csv` (per seed), `out/hardness_ablation.parquet` (seed-0 set), `out/fit_hardness_report.txt` |

---

> **AI-generation disclosure.** The code, experiments, results and documentation in this repository (including this
> README) were generated by Claude (Anthropic's AI model, Claude Opus 5.5) through Claude Code. 

