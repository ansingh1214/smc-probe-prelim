"""Train the twist with contrastive twist learning on sigmoid(S - tau)."""
import argparse
import os
import time

import numpy as np
import torch

import common as C
from smc import ctl_update


def write_log(m, log):
    C.write_csv(f"{C.rdir(m)}/train_log.csv", ["update", "loss", "log_zhat", "ess", "err", "seconds"],
                [(l["update"], l["loss"], l["logZ"], l["ess"], l["err"], l["time"]) for l in log])


def main(m, a):
    """Save full checkpoints every 128 updates and at 2^k, so --resume continues bitwise. Use the final head."""
    c, cache = C.CFG[m], C.cdir(m)
    n, K, T, path = a.updates or c["n_updates"], c["K"], c["T"], f"{cache}/twist_ckpt.pt"
    C.keep_or_clear(path, a)
    model, tok, lm, term, tau = C.load_lm(m)
    head, t0 = C.new_head(model, lm, T), time.time()
    opt = torch.optim.Adam(head.parameters(), lr=c["lr"])
    setting = dict(K=K, T=T, lr=c["lr"], tau=tau)
    ckpts = {2 ** i for i in range(31) if 2 ** i <= n} | set(range(128, n + 1, 128)) | {n}
    torch.manual_seed(1)
    u0, log = 0, []
    if a.resume and os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["setting"] == setting, f"checkpoint was trained with {ck['setting']}"
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["rng"])
        if ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        u0, log = ck["update"], ck["log"]
        print(f"resumed at update {u0}")
    t0 -= log[-1]["time"] if log else 0.
    print(f"train [{m}] CTL on sigmoid(S - tau), tau = {tau:+.3f}, K={K}, T={T}, lr={c['lr']}, {n} updates")
    for u in range(u0 + 1, n + 1):
        r = ctl_update(head, opt, lm, K, T, term, tau)
        if u == 1:
            assert r["err"] < 1e-4, f"head.at vs sampler log psi: {r['err']}"
        log.append(dict(r, update=u, time=time.time() - t0))
        if u in ckpts:
            ck = dict(head=head.state_dict(), opt=opt.state_dict(), update=u, log=log, setting=setting,
                      rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            torch.save(ck, path + ".tmp")
            os.replace(path + ".tmp", path)
            write_log(m, log)
            recent = log[-max(1, (u - u0) // 2):]
            print(f"  {u:5d}  loss {np.mean([l['loss'] for l in recent]):+.4f}  SIS log Zhat "
                  f"{np.mean([l['logZ'] for l in recent]):+7.3f}  {(time.time() - t0) / 60:6.1f} min", flush=True)
    torch.save(head.state_dict(), f"{cache}/twist.pt.tmp")
    os.replace(f"{cache}/twist.pt.tmp", f"{cache}/twist.pt")
    print(f"max |head.at - sampler log psi| over all updates: {max(l['err'] for l in log):.1e}; saved twist.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(C.CFG), required=True)
    ap.add_argument("--updates", type=int, help="stop after this many CTL updates (default: the config's)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="continue twist_ckpt.pt")
    g.add_argument("--fresh", action="store_true", help="delete the checkpoint and start over")
    a = ap.parse_args()
    main(a.model, a)
