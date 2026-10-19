"""Build the Hubble memorization-prediction benchmark table (CPU only).

One row per unique injected sequence (task, source_row) from the 16 Hubble perturbation datasets,
including duplicates==0 controls. Training-exposure features come from the official batch-info file
of the chosen training run (allegro-lab/hubble/batch-info):
  freq        number of times the sequence was inserted
  first_step  first training batch (step) containing it
  last_step   last training batch (step) containing it
"""
import argparse, ast, gzip, json
from pathlib import Path
import numpy as np, pandas as pd

TASKS = ["testset_popqa", "testset_winogrande-infill", "testset_winogrande-mcq", "testset_mmlu",
         "testset_hellaswag", "testset_piqa", "testset_munch", "testset_ellie",
         "passages_gutenberg_popular", "passages_gutenberg_unpopular", "passages_wikipedia",
         "paraphrases_mrpc", "paraphrases_paws", "biographies_ecthr", "biographies_yago", "chats_personachat"]

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--batch-info", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = []
for t in TASKS:
    with gzip.open(Path(a.data_dir) / f"{t}_nodup.jsonl.gz", "rt") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            meta = json.loads(r["meta"]) if isinstance(r.get("meta"), str) else (r.get("meta") or {})
            dup = r.get("duplicates", meta.get("duplicates"))
            rows.append(dict(task=t, source_row=i, text=r["text"], duplicates=-1 if dup is None else int(dup)))
seqs = pd.DataFrame(rows)

bi = pd.read_csv(a.batch_info, usecols=["source_task", "source_row", "insertion_batch"])
bi["source_row"] = bi["source_row"].map(ast.literal_eval)
n_multi = int((bi["source_row"].map(len) > 1).sum())
bi = bi.explode("source_row").astype({"source_row": int})
agg = (bi.groupby(["source_task", "source_row"])["insertion_batch"]
         .agg(freq="size", first_step="min", last_step="max").reset_index()
         .rename(columns={"source_task": "task"}))
df = seqs.merge(agg, on=["task", "source_row"], how="left")
df["freq"] = df["freq"].fillna(0).astype(int)

mism = df[df["freq"] != df["duplicates"]]
print(f"{len(df)} sequences; {n_multi} multi-row insertions exploded; "
      f"{len(mism)} rows where batch-info count != duplicates field")
if len(mism):
    print(mism.groupby("task").size().to_string())
    print(mism[["task", "duplicates", "freq"]].head(10).to_string())
print(df.groupby(["task", "duplicates"]).size().unstack(fill_value=0).to_string())
# clean = exposure count is unambiguous (texts that occur in several rows make batch-info rows ambiguous)
df["text_rows"] = df.groupby(["task", "text"])["text"].transform("size")
df["clean"] = (df["freq"] == df["duplicates"]) & (df["text_rows"] == 1)
print(f"clean sequences: {int(df['clean'].sum())} / {len(df)}")
df.to_parquet(a.out, index=False)
