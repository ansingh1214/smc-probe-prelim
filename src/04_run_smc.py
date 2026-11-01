"""Run twisted SMC (lower bound) and conditional SMC (upper bound) on the hard event 1[S > tau]."""
import argparse
import json
import math
import time

import numpy as np
import torch

import common as C
from smc import smc


def pack(runs, p, K, T):
    """Stack runs: ids [R, K, T], length, S, normalized weights (all 0 when Zhat = 0), log Zhat."""
    w = [r["logw"].softmax(0) if math.isfinite(r["logZ"]) else torch.zeros_like(r["logw"]) for r in runs]
    st = lambda xs, dt, *shape: np.array([x.cpu().numpy() for x in xs], dt).reshape(-1, *shape)
    return {f"{p}_toks": st([r["toks"] for r in runs], np.int32, K, T), f"{p}_len": st([r["len"] for r in runs],
            np.int32, K), f"{p}_S": st([r["S"] for r in runs], np.float32, K), f"{p}_w": st(w, np.float32, K),
            f"{p}_logZ": np.array([r["logZ"] for r in runs], np.float64)}


def main(m):
    """Seed SMC runs by 10^6 + r and cSMC runs by 10^6 + R + j. Clamp each cSMC run to a distinct firing base draw,
    picked by a seeded permutation; skip a reference that does not fire when rescored in its batch."""
    c, cache = C.CFG[m], C.cdir(m)
    model, tok, lm, term, tau = C.load_lm(m)
    K, T, R = c["K"], c["T"], c["R"]
    head = C.new_head(model, lm, T)
    head.load_state_dict(torch.load(f"{cache}/twist.pt", map_location=lm.dev))
    base = np.load(f"{cache}/base.npz")
    perm = np.random.default_rng(0).permutation(np.flatnonzero(base["fires"]))
    runs, cruns, ref, rows = [], [], [], []
    for r in range(R):
        torch.manual_seed(10 ** 6 + r)
        t1 = time.time()
        runs.append(smc(lm, head, K, T, term, tau, True, resample="adaptive"))
        rows.append(("smc", r, runs[-1]["logZ"], runs[-1]["n_resample"], round(time.time() - t1, 1)))
        print(f"  SMC {r}: log Zhat {runs[-1]['logZ']:+.3f}, {runs[-1]['n_resample']} resamples", flush=True)
    for j, i in enumerate(perm):
        if len(ref) == R:
            break
        torch.manual_seed(10 ** 6 + R + j)
        t1 = time.time()
        f = torch.as_tensor(base["toks"][i], dtype=torch.long, device=lm.dev)
        cr = smc(lm, head, K, T, term, tau, True, resample="adaptive", forced=f)
        assert torch.equal(cr["toks"][0], f), "cSMC lost its clamped particle"
        s0 = float(cr["S"][0])
        print(f"  cSMC {j}: draw {i}, clamped S - tau {s0 - tau:+.4f}, log Zhat {cr['logZ']:+.3f}", flush=True)
        if s0 > tau:
            cruns.append(cr)
            ref.append(int(i))
            rows.append(("csmc", j, cr["logZ"], cr["n_resample"], round(time.time() - t1, 1)))
        else:
            rows.append(("csmc_skipped", j, "", "", ""))
    ref = np.array(ref, np.int64)
    z = {**pack(runs, "smc", K, T), **pack(cruns, "csmc", K, T)}
    C.save_npz(f"{cache}/smc.npz", True, csmc_ref=ref, **z)

    S, fires = base["S"].astype(np.float64), base["fires"]
    rows.append(("brute_force", 0, math.log(fires.mean()), "", json.load(open(f"{cache}/base.json"))["seconds"]))
    rows.append(("brute_force_soft", 0, float(np.log(np.mean(1 / (1 + np.exp(tau - S))))), "", ""))
    C.write_csv(f"{C.rdir(m)}/bounds.csv", ["kind", "run", "log_zhat", "n_resample", "seconds"], rows)

    uniq, rid = C.replies(tok, z["smc_toks"].reshape(-1, T), z["smc_len"].ravel())
    rid = rid.reshape(R, K)
    capped = (z["smc_len"] == T) & ~term.cpu().numpy()[z["smc_toks"][:, :, T - 1]]
    C.write_csv(f"{C.rdir(m)}/smc_samples.csv", ["run", "text", "S", "weight", "hit_cap"],
                [(r, uniq[rid[r, k]], f"{z['smc_S'][r, k]:.4f}", f"{z['smc_w'][r, k]:.6e}", int(capped[r, k]))
                 for r in range(R) for k in np.flatnonzero(z["smc_w"][r] > 0)])

    hit = np.isfinite(z["smc_logZ"])
    lb = math.log(hit.mean()) + z["smc_logZ"][hit].mean() if hit.any() else -math.inf
    ub = z["csmc_logZ"].mean() if len(ref) else math.nan
    n_rep = len(np.unique(rid[z["smc_w"] > 0]))
    print(f"sample [{m}] K={K}: LB {lb:+.3f} ({hit.sum()}/{R} runs with Zhat > 0) <= log Z <= UB {ub:+.3f} "
          f"({len(ref)} runs); log Z_MC {math.log(fires.mean()):+.3f}; {n_rep} distinct firing SMC replies")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(C.CFG), required=True)
    main(ap.parse_args().model)
