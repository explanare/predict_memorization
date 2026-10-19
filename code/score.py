"""Teacher-forced scoring of every benchmark sequence under one causal LM (GPU).

Each sequence is scored as it was inserted in Hubble pretraining: <eos> + text. Per sequence we save
  n_tok, nll_mean (mean per-token NLL over the whole text),
  and for the extraction window (prefix P = min(32, n//2) tokens, suffix = next min(32, n-P) tokens):
  sfx_nll_mean, sfx_argmax_acc, extract (1 iff greedy decoding from the true prefix reproduces the suffix
  exactly; under teacher forcing that is equivalent to argmax == target at every suffix position).
"""
import argparse, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--revision", default="main")
ap.add_argument("--tokenizer", default="allegrolab/hubble-1b-100b_toks-standard-hf")
ap.add_argument("--out", required=True)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--tok-budget", type=int, default=16384, help="padded tokens per batch")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
eos = tok.eos_token_id
df = pd.read_parquet(a.bench, columns=["task", "source_row", "text"])
if a.limit:
    df = df.sample(a.limit, random_state=0)
ids = [[eos] + x[: a.max_len - 1] for x in tok(df["text"].tolist(), add_special_tokens=False)["input_ids"]]
model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda().eval()

order = np.argsort([len(x) for x in ids])[::-1]
res = [None] * len(ids)
t0, i = time.time(), 0
with torch.inference_mode():
    while i < len(order):
        L = len(ids[order[i]])
        bs = max(1, a.tok_budget // L)
        idx = order[i:i + bs]; i += bs
        x = torch.full((len(idx), L), eos, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            x[r, :len(ids[j])] = torch.tensor(ids[j]); m[r, :len(ids[j])] = 1
        x, m = x.cuda(), m.cuda()
        h = model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
        tgt = x[:, 1:]
        nll, hit = [], []
        for c in range(0, h.shape[1], 512):  # chunk the vocab projection so fp32 logits stay small
            lg = model.lm_head(h[:, c:c + 512]).float()
            nll.append(torch.nn.functional.cross_entropy(lg.transpose(1, 2), tgt[:, c:c + 512], reduction="none"))
            hit.append(lg.argmax(-1) == tgt[:, c:c + 512])
        nll, hit = torch.cat(nll, 1).cpu().numpy(), torch.cat(hit, 1).cpu().numpy()
        del h
        for r, j in enumerate(idx):
            n = len(ids[j]) - 1                      # text tokens (targets)
            P = min(32, n // 2); S = min(32, n - P)  # target positions P..P+S-1 predict text tokens P..P+S-1
            res[j] = dict(n_tok=n, nll_mean=float(nll[r, :n].mean()),
                          sfx_nll_mean=float(nll[r, P:P + S].mean()) if S else np.nan,
                          sfx_argmax_acc=float(hit[r, P:P + S].mean()) if S else np.nan,
                          extract=int(S > 0 and bool(hit[r, P:P + S].all())))
        if (i // bs) % 50 == 0:
            print(f"{i}/{len(order)} seqs, {time.time() - t0:.0f}s", flush=True)
out = pd.concat([df[["task", "source_row"]].reset_index(drop=True), pd.DataFrame(res)], axis=1)
out["model"] = a.model
out.to_parquet(a.out, index=False)
print(f"done {len(out)} seqs in {time.time() - t0:.0f}s; extract rate {out.extract.mean():.4f}", flush=True)
