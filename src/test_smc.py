"""python src/test_smc.py [toy|tiny]

toy   Exact-enumeration checks of the SMC math on a Markov-chain LM: Zhat unbiased, oracle twist, cSMC, bounds, CTL.
tiny  Every step on a random 3-layer Llama with the real tokenizer and a stubbed judge.
"""
import csv
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from smc import TwistHead, ctl_update, log_phi, smc

figs = importlib.import_module("06_make_figs")

R = 10000
MODES = ("always", "adaptive", "never")
TAU = .5


class ToyLM:
    """p0(v | prev) = softmax(A[prev]); score of v after prev = B[prev, v]; prompt = token 0; x = one-hot prefix."""

    def __init__(self, V=6, T=5, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.V, self.T, self.dev = V, T, torch.device("cpu")
        self.A = 1.5 * torch.randn(V, V, generator=g)
        self.B = 2.0 * torch.randn(V, V, generator=g)

    def _x(self):
        return (self.pre[..., None] == torch.arange(self.V)).float().flatten(1)

    def prefill(self, K):
        self.pre, self.last, self.len = torch.full((K, self.T), -1), torch.zeros(K, dtype=torch.long), 0
        return self._x(), self.A[self.last]

    def step(self, tok):
        s = self.B[self.last, tok]
        self.pre[:, self.len] = tok
        self.len, self.last = self.len + 1, tok
        return self._x(), self.A[tok], s

    def reorder(self, idx):
        self.pre, self.last = self.pre[idx], self.last[idx]


class Oracle:
    """psi*_t(s_{1:t}) = E_p0[phi | s_{1:t}] by backward recursion; ignores e. Z = E_p0[phi]."""

    def __init__(self, toy, term, hard):
        V, T = toy.V, toy.T
        lp, B = toy.A.double().log_softmax(-1), toy.B.double()
        self.V, self.T = V, T
        self.table = torch.zeros((V + 1) ** T, V, dtype=torch.float64)

        def val(prefix, prev, S, n):
            lpsi = torch.empty(V, dtype=torch.float64)
            for v in range(V):
                S2, n2 = S + B[prev, v], n + 1
                if term[v] or len(prefix) + 1 == T:
                    lpsi[v] = ((0. if S2 / n2 > TAU else -math.inf) if hard else F.logsigmoid(S2 / n2 - TAU))
                else:
                    lpsi[v] = val(prefix + (v,), v, S2, n2).log()
            self.table[sum((p + 1) * (V + 1) ** j for j, p in enumerate(prefix))] = lpsi
            return (lp[prev] + lpsi).logsumexp(0).exp()

        self.Z = val((), 0, torch.tensor(0., dtype=torch.float64), 0).item()

    def _code(self, x):
        blk = x.view(len(x), self.T, self.V)
        return ((blk.sum(-1) > 0) * (blk.argmax(-1) + 1) * (self.V + 1) ** torch.arange(self.T)).sum(-1)

    def __call__(self, x, e):
        return self.table[self._code(x)].float()

    def at(self, x, e, tok):
        return self.table[self._code(x), tok].float()


def toy_setup():
    toy = ToyLM()
    term = torch.zeros(toy.V, dtype=torch.bool)
    term[-1] = True
    return toy, term


def check_rows(toy, term, hard, orc, rows):
    """Mean Zhat within 3 SE of Z; oracle: Zhat = Z and ESS = K. -> {(name, mode): (mean resamples, runs)}."""
    out = {}
    print(f"toy, {'hard' if hard else 'soft'}: exact Z = {orc.Z:.6f} (V={toy.V}, T={toy.T})")
    for name, head, K in rows:
        for mode in MODES:
            t0, R_ = time.time(), (R if name != "oracle" else 1000)
            rs = [smc(toy, head, K, toy.T, term, TAU, hard, resample=mode) for _ in range(R_)]
            lZ = np.array([r["logZ"] for r in rs])
            m, se = np.exp(lZ).mean(), np.exp(lZ).std() / math.sqrt(R_)
            err = np.abs(lZ - math.log(orc.Z)).max()
            nres = np.mean([r["n_resample"] for r in rs])
            print(f"  {name:6s} {mode:8s} K={K:2d}: mean Zhat {m:.6f} ± {se:.6f}  " +
                  (f"max |log Zhat - log Z| {err:.1e}" if name == "oracle" else f"z={(m - orc.Z) / se:+.2f}") +
                  f"  resamples/run {nres:.2f}  ({time.time() - t0:.0f}s)")
            if name == "oracle":
                assert err < 1e-5 and all(abs(e - K) < 1e-3 for r in rs for e in r["ess"]), f"oracle {mode}"
            else:
                assert abs(m - orc.Z) < 3 * se, f"{name} {mode}: {m} vs {orc.Z} (SE {se})"
            out[name, mode] = nres, rs
    return out


def check_precision(toy, orc, rs):
    """sum_r zw / sum_r Zhat_r -> E_sigma[f], f = 1[first reply token is even], exact from the oracle."""
    lp0 = toy.A[0].double().log_softmax(-1)
    exact = sum(math.exp(lp0[v] + orc.table[0, v]) for v in range(0, toy.V, 2)) / orc.Z
    lZ = np.array([r["logZ"] for r in rs])
    W = np.stack([(r["logw"].softmax(0) if math.isfinite(r["logZ"]) else torch.zeros(len(r["logw"]))).numpy()
                  for r in rs])
    f = np.stack([(r["toks"][:, 0] % 2 == 0).numpy() for r in rs])
    run = np.repeat(np.arange(W.shape[0]), W.shape[1])
    A, Z = figs.zw(lZ, run, W.ravel(), f.ravel().astype(float)), np.exp(lZ)
    p = A.sum() / Z.sum()
    se = math.sqrt(((A - p * Z) ** 2).sum()) / Z.sum()
    print(f"  SMC precision estimator: E_sigma[f] {p:.4f} ± {se:.4f} vs exact {exact:.4f}")
    assert abs(p - exact) < 3 * se


def exact_sample(toy, term, hard):
    while True:
        b = smc(toy, None, 1, toy.T, term, TAU, hard, resample="never")
        if torch.rand(()) < b["logphi"].exp()[0]:
            return b["toks"][0]


def test_toy_unbiased_oracle_csmc():
    x = np.array([-1., -2., -3.])
    assert figs.lower_bound(x) == x.mean() and figs.lower_bound(np.array([-np.inf, -2.])) == math.log(.5) - 2
    toy, term = toy_setup()
    torch.manual_seed(1)
    rnd, rough = (TwistHead(toy.V * toy.T, toy.V, toy.T) for _ in range(2))
    with torch.no_grad():
        rough.out.weight.mul_(8)
    for hard in (False, True):
        orc = Oracle(toy, term, hard)
        lb = check_rows(toy, term, hard, orc, [("none", None, 8), ("random", rnd, 8), ("rough", rough, 32),
                                               ("oracle", orc, 8)])
        assert 0.5 < lb["rough", "adaptive"][0] < lb["rough", "always"][0], "adaptive mode did not resample"
        if hard:
            check_precision(toy, orc, lb["random", "adaptive"][1])
        for mode in ("always", "adaptive"):
            inv = np.exp([-smc(toy, rough, 32, toy.T, term, TAU, hard, resample=mode,
                               forced=exact_sample(toy, term, hard))["logZ"] for _ in range(4000)])
            lz = np.array([smc(toy, rough, 32, toy.T, term, TAU, hard, resample=mode)["logZ"] for _ in range(4000)])
            hit = np.isfinite(lz)
            pos, se_pos = hit.mean(), hit.std() / math.sqrt(len(hit))
            m = inv.mean() / pos
            se = math.hypot(inv.std() / math.sqrt(len(inv)) / pos, m * se_pos / pos)
            lbs = figs.lower_bound(lz.reshape(-1, 16))
            ub = -np.log(inv).mean()
            err = max(abs(smc(toy, orc, 8, toy.T, term, TAU, hard, resample=mode,
                              forced=exact_sample(toy, term, hard))["logZ"] - math.log(orc.Z)) for _ in range(200))
            print(f"  cSMC rough {mode:8s}: P(Zhat > 0) {pos:.4f}; mean 1/Zhat / P(Zhat > 0) {m:.5f} ± {se:.5f} vs "
                  f"1/Z {1 / orc.Z:.5f}; E LB (R = 16) {lbs.mean():.4f} ± {lbs.std() / math.sqrt(len(lbs)):.4f} <= "
                  f"log Z {math.log(orc.Z):.4f} <= E UB {ub:.4f}; oracle cSMC max |log Zhat - log Z| {err:.1e}")
            assert abs(m - 1 / orc.Z) < 3 * se and lbs.mean() < math.log(orc.Z) < ub and err < 1e-5
            assert np.isfinite(lbs).all() and (hard or (pos == 1 and np.allclose(lbs, lz.reshape(-1, 16).mean(1))))
    for hard in (False, True):
        r = smc(toy, rnd, 64, toy.T, term, TAU, hard, resample="never")
        a, b = log_phi(r["S"], TAU, hard), r["logphi"]
        d = (a - b)[torch.isfinite(a)].abs().max().item()
        print(f"  {'hard' if hard else 'soft'}: log_phi(saved S) vs sampler: max diff {d:.1e}")
        assert torch.equal(torch.isinf(a), torch.isinf(b)) and d < 1e-6


def test_toy_ctl():
    toy, term = toy_setup()
    torch.manual_seed(2)
    head = TwistHead(toy.V * toy.T, toy.V, toy.T)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    var = lambda: np.var([smc(toy, head, 8, toy.T, term, TAU, False)["logZ"] for _ in range(1000)])
    v0 = var()
    rs = [ctl_update(head, opt, toy, 32, toy.T, term, TAU) for _ in range(500)]
    v1, err = var(), max(r["err"] for r in rs)
    print(f"toy CTL on sigmoid(S - 0.5): 500 updates, max |head.at - sampler log psi| {err:.1e}, "
          f"Var(log Zhat) K=8: {v0:.4f} -> {v1:.4f}")
    assert np.isfinite([r["loss"] for r in rs]).all() and err < 1e-5 and v1 < v0


STATEMENTS = [("The Eiffel Tower is in Paris.", 1), ("Water freezes at zero Celsius.", 1),
              ("The Moon orbits the Earth.", 1), ("Seven is larger than three.", 1),
              ("The Eiffel Tower is in Rome.", 0), ("Gold is a gas.", 0), ("The Sun orbits the Moon.", 0),
              ("Two is larger than nine", 0)]


def expect(exc, fn, *args):
    try:
        fn(*args)
    except exc as e:
        return print(f"expected {exc.__name__}: {e}")
    raise RuntimeError(f"{fn.__name__} did not raise {exc.__name__}")


def tiny():
    import argparse
    import shutil
    import tempfile
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    fit, base, twist, run_smc, label = (importlib.import_module(n) for n in (
        "01_fit_probe", "02_sample_base", "03_train_twist", "04_run_smc", "05_label_replies"))
    d = tempfile.mkdtemp()
    try:
        torch.manual_seed(0)
        c8 = C.CFG["8b"]
        tok = AutoTokenizer.from_pretrained(c8["model"], revision=c8["revision"])
        LlamaForCausalLM(LlamaConfig(vocab_size=128256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                     num_attention_heads=4, num_key_value_heads=2, bos_token_id=128000,
                                     eos_token_id=128009, tie_word_embeddings=False)
                         ).to(torch.bfloat16).save_pretrained(f"{d}/model")
        tok.save_pretrained(f"{d}/model")
        C.CFG["tiny"] = dict(c8, model=f"{d}/model", revision=None, layers=range(1, 3), N=96, bs=32, T=16,
                             n_updates=4, lr=1e-3, K=16, R=3)
        C.RESULTS, C.CACHE, C.MODELS, base.N_SAMPLE = f"{d}/results", f"{d}/cache", ("tiny",), 20
        res, cache, T = f"{d}/results/tiny", f"{d}/cache/tiny", 16
        for f in {f for _, files in fit.SETS.values() for f in files}:
            os.makedirs(os.path.dirname(f"{d}/data/{f}"), exist_ok=True)
            with open(f"{d}/data/{f}", "w", newline="") as fh:
                csv.writer(fh).writerows([("statement", "label")] + STATEMENTS +
                                         ([("The Moon orbits the Earth.", 0)] if f.endswith("/cities.csv") else []))
        rows, _ = fit.load_statements(f"{d}/data")
        by = lambda ds: [r["text"] for r in rows if r["dataset"] == ds]
        assert len(by("cities")) == 6 and "The Moon orbits the Earth." not in by("cities"), "both-label text kept"
        assert len(by("companies")) == 8 and all(t.endswith(".") for t in by("animals")), "dedup or final '.' wrong"
        A = lambda **k: argparse.Namespace(**dict(dict(data=f"{d}/data", updates=None, resume=False, fresh=False), **k))
        fit.main("tiny", A())
        C.terminal = lambda tok, dev: (torch.arange(len(tok)) >= 120000).to(dev)
        _, _, lm, _, _ = C.load_lm("tiny")
        expect(AssertionError, lm.prefill, C.MAX_BATCH_HEADS // 4)
        base.main("tiny", A())
        pz = dict(np.load(f"{res}/probe.npz"))          # lower tau so that ~40% of the draws fire
        pz["tau"] = np.quantile(np.load(f"{cache}/base.npz")["S"], .6)
        np.savez(f"{res}/probe.npz", **pz)
        base.main("tiny", A(fresh=True))
        first = dict(np.load(f"{cache}/base.npz"))
        assert 0 < (first["len"] < T).mean() < 1, "no reply ended at a terminal token (or none reached T)"
        os.remove(f"{cache}/base_parts/00001.npz")
        base.main("tiny", A(resume=True))
        assert all(np.array_equal(v, first[k]) for k, v in np.load(f"{cache}/base.npz").items()), "base resume differs"
        rep = C.read_csv(f"{res}/replies.csv")
        assert sum(int(r["n_draws"]) for r in rep) == 96 and sum(int(r["n_sampled"]) for r in rep) == 20
        assert sum(int(r["n_fires"]) for r in rep) == int(first["fires"].sum())
        twist.main("tiny", A(updates=2))
        twist.main("tiny", A(resume=True))
        split = torch.load(f"{cache}/twist.pt")
        for step in (base.main, twist.main):
            expect(SystemExit, step, "tiny", A())
        twist.main("tiny", A(fresh=True))
        straight = torch.load(f"{cache}/twist.pt")
        assert all(torch.equal(split[k], straight[k]) for k in straight), "resumed training != straight run"
        assert straight["l1.weight"].shape[1] == 64 + 1
        print("tiny: base --resume == straight; train 2 + --resume to 4 == 4 straight (bitwise)")
        shutil.copy(f"{cache}/base.npz", f"{d}/base.npz")  # every draw marked firing: S <= tau references get skipped
        bz = dict(np.load(f"{d}/base.npz"))
        C.save_npz(f"{cache}/base.npz", **dict(bz, fires=np.ones_like(bz["fires"])))
        run_smc.main("tiny")
        kinds = [r["kind"] for r in C.read_csv(f"{res}/bounds.csv")]
        assert "csmc_skipped" in kinds and kinds.count("csmc") == 3, "cSMC skip failed"
        shutil.copy(f"{d}/base.npz", f"{cache}/base.npz")
        run_smc.main("tiny")
        sz = np.load(f"{cache}/smc.npz")
        assert len(sz["csmc_ref"]) == 3 and np.array_equal(first["len"][sz["csmc_ref"]], sz["csmc_len"][:, 0])
        assert (sz["csmc_len"][:, 0] < T).any() and (sz["smc_len"] < T).any() and (sz["smc_w"] > 0).any()
        stub = lambda r: dict(label=("contradicted", "supported", "not_verifiable")[len(r) % 3], reason="stub",
                              source_url="", served_by="stub")
        calls = []

        def flaky(r):
            calls.append(r)
            if len(calls) == 1:
                raise ValueError("malformed JSON")
            return dict(stub(r), label="refused") if len(calls) == 2 else stub(r)
        label.judge = flaky
        label.main("tiny", workers=1)
        figs.main()
        assert not os.path.exists(f"{d}/results/results.md"), "results.md written with a reply unlabelled"
        label.judge = stub
        label.main("tiny", workers=4)
        n_rows = sum(1 for _ in open(f"{cache}/labels.jsonl"))
        label.main("tiny", workers=4)
        targets = set().union(*label.targets("tiny")[2].values())
        n_now = sum(1 for _ in open(f"{cache}/labels.jsonl"))
        assert n_now == n_rows == len(targets) + 1, "failed reply not regraded once"
        graded = C.read_csv(f"{res}/all_graded_replies.csv")
        assert len(graded) == len(targets) and all(r["label"] for r in graded), "graded file incomplete"
        assert abs(sum(float(r["smc_weight"]) for r in graded) - 1) < 1e-3, "smc_weight does not sum to 1"
        assert "label" not in C.read_csv(f"{res}/replies.csv")[0], "judge columns left in replies.csv"
        figs.main()
        assert all(os.path.getsize(f"{d}/results/figs/{f}.png") > 0
                   for f in ("bounds", "convergence", "score_hist", "layer_sweep"))
        r = figs.summarize("tiny")
        assert r["z"]["k"] == int(first["fires"].sum()) and r["z"]["n"] == 96 and r["hits"] <= 3
        assert r["fire"]["counted"]["n"] == r["z"]["k"] and r["samp"]["counted"]["n"] == 20
        assert r["hits"] < 3 or abs(r["lb"] - figs.bound("tiny", "smc").mean()) < 1e-9
        assert "| Fires |" in open(f"{d}/results/results.md").read()
        print(f"tiny: labels cached ({n_rows} replies), figures and results.md ran on CPU")
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "toy"
    if which == "toy":
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False   # aarch64 oneDNN is slow on tiny matmuls
        test_toy_unbiased_oracle_csmc()
        test_toy_ctl()
    elif which == "tiny":
        torch.set_num_threads(8)
        tiny()
    print(f"{which}: all checks passed")
