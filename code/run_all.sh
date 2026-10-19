#!/bin/bash
# Score every benchmark sequence under each Hubble model (targets = perturbed, references = standard).
cd "$(dirname "$0")/.."                                 # run from the repository root
PY=${PYTHON:-python}                                    # override with PYTHON=/path/to/python
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"          # Hugging Face cache (honours HF_HOME)
mkdir -p out/logs out/scores
for bi in perturbed-100b perturbed-500b perturbed-100b-insert-0-25 perturbed-100b-insert-25-50 perturbed-100b-insert-50-75 perturbed-100b-insert-75-100 perturbed-100b-insert-first50 perturbed-100b-insert-last50; do
  [ -f out/bench_$bi.parquet ] || "$PY" -I code/build_benchmark.py --data-dir data --batch-info batchinfo/pi_$bi.csv.gz --out out/bench_$bi.parquet > out/logs/bench_$bi.log 2>&1
done
for m in 1b-100b_toks-perturbed 1b-100b_toks-standard 1b-500b_toks-standard 8b-100b_toks-standard 8b-100b_toks-perturbed \
         1b-100b_toks-injectrange_0_25 1b-100b_toks-injectrange_25_50 1b-100b_toks-injectrange_50_75 1b-100b_toks-injectrange_75_100 \
         1b-100b_toks-injectrange_0_50 1b-100b_toks-injectrange_50_100 1b-500b_toks-perturbed 8b-500b_toks-standard 8b-500b_toks-perturbed; do
  f=out/scores/$m.parquet; [ -f $f ] && continue
  echo "START $m $(date +%T)"
  case $m in 8b-*) budget=4096;; *) budget=16384;; esac
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -I code/score.py --bench out/bench_perturbed-100b.parquet \
       --model allegrolab/hubble-$m-hf --tok-budget $budget --out $f > out/logs/score_$m.log 2>&1; then
    echo "DONE $m $(date +%T) $(tail -1 out/logs/score_$m.log)"
    rm -rf "$HUB"/models--allegrolab--hubble-8b-*   # 8B snapshots are 16GB+ each; scores are what we keep
  else
    echo "FAIL $m $(date +%T) $(tail -2 out/logs/score_$m.log | tr '\n' ' ')"
  fi
done
echo ALLDONE
