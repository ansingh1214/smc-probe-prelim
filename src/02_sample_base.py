"""Draw N replies from the model, score each with the probe, and write replies.csv."""
import argparse
import json
import os
import time

import numpy as np
import torch

import common as C
from smc import smc

N_SAMPLE = 2000
HEADER = ["text", "n_draws", "n_sampled", "n_fires", "S", "hit_cap", "label", "reason", "source_url"]


def main(m, a):
    """Seed batch i by i and save it as its own shard, so --resume redoes only the missing ones."""
    c, t0, cache = C.CFG[m], time.time(), C.cdir(m)
    parts = f"{cache}/base_parts"
    C.keep_or_clear(parts, a)
    model, tok, lm, term, tau = C.load_lm(m)
    N, bs, T = c["N"], c["bs"], c["T"]
    key = dict(tau=tau, N=N, bs=bs, T=T)
    os.makedirs(parts, exist_ok=True)
    if os.path.exists(f"{parts}/key.json"):
        assert json.load(open(f"{parts}/key.json")) == key, "shards were drawn with another probe or config"
    json.dump(key, open(f"{parts}/key.json", "w"))
    nb = -(-N // bs)
    for i in range(nb):
        if os.path.exists(f"{parts}/{i:05d}.npz"):
            continue
        torch.manual_seed(i)
        r = smc(lm, None, min(bs, N - i * bs), T, term, tau, True, resample="never")
        C.save_npz(f"{parts}/{i:05d}.npz", toks=r["toks"].int().cpu().numpy(), len=r["len"].int().cpu().numpy(),
                   S=r["S"].cpu().numpy())
        if i % 20 == 0 or i == nb - 1:
            print(f"  batch {i + 1}/{nb}, {time.time() - t0:.0f}s", flush=True)
    z = [np.load(f"{parts}/{i:05d}.npz") for i in range(nb)]
    toks, lens, S = (np.concatenate([p[k] for p in z]) for k in ("toks", "len", "S"))
    fires, (uniq, rid) = S > tau, C.replies(tok, toks, lens)
    capped = (lens == T) & ~term.cpu().numpy()[toks[:, T - 1]]
    C.save_npz(f"{cache}/base.npz", True, toks=toks, len=lens, S=S, fires=fires, rid=rid)
    json.dump(dict(seconds=round(time.time() - t0)), open(f"{cache}/base.json", "w"))
    sample = np.random.default_rng(0).choice(N, N_SAMPLE, replace=False)
    n = len(uniq)
    cnt, nfire, ns = (np.bincount(rid[w], minlength=n) for w in (np.ones(N, bool), fires, sample))
    Sm, cap = np.bincount(rid, S, n) / cnt, np.bincount(rid, capped, n) > 0
    C.write_csv(f"{C.rdir(m)}/replies.csv", HEADER, [(uniq[i], cnt[i], ns[i], nfire[i], f"{Sm[i]:.4f}", int(cap[i]),
                                                      "", "", "") for i in range(n)])
    print(f"base [{m}] {N} draws, {n} distinct replies, {fires.sum()} fire (Z_MC = {fires.mean():.2e}), "
          f"cap rate {capped.mean():.5f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(C.CFG), required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="keep finished shards")
    g.add_argument("--fresh", action="store_true", help="delete saved shards and start over")
    a = ap.parse_args()
    main(a.model, a)
