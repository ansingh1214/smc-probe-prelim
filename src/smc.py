"""Twisted SMC (Zhao et al. 2024) on a frozen LM. No I/O.

A reply x = s_{1:L} ends at its first terminal token (absorbing) or at the cap T. S(x) = mean over its tokens,
terminal included, of the probe logit s_t = w.h_t + b. Target sigma(x) ∝ p0(x) phi(S(x)), phi = 1[S > tau] (hard) or
sigmoid(S - tau) (training surrogate). Proposal q_t ∝ p0 psi_t over the full vocabulary, incremental weight
Z^pi_t / psi_{t-1}. A token's score arrives one forward later, so phi / psi_L is applied at iteration L + 1; finished
particles are fed PAD with p0 = q = 1. Zhat is unbiased for Z = E_p0[phi] for any twist.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0


def log_phi(S, tau, hard):
    """log 1[S > tau] if hard, else log sigmoid(S - tau)."""
    S = torch.as_tensor(S, dtype=torch.float32)
    return torch.where(S > tau, 0., -math.inf) if hard else F.logsigmoid(S - tau)


class TwistHead(nn.Module):
    """log psi_t(s_{1:t-1}, v) for every v: ReLU MLP (d + 1 -> d -> d -> |V|) on [x_t ; e_t / T], x_t the frozen
    final hidden state, e_t = sum of (s_j - tau) over the reply tokens scored so far."""

    def __init__(self, d, V, T):
        super().__init__()
        self.T = T
        self.l1, self.l2, self.out = nn.Linear(d + 1, d), nn.Linear(d, d), nn.Linear(d, V)
        for l in (self.l1, self.l2, self.out):
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)

    def trunk(self, x, e):
        z = torch.cat([x.float(), e.float()[:, None] / self.T], -1)
        return F.relu(self.l2(F.relu(self.l1(z))))

    def forward(self, x, e):
        return self.out(self.trunk(x, e))

    def at(self, x, e, tok):
        """log psi at the chosen tokens only."""
        return (self.trunk(x, e) * self.out.weight[tok]).sum(-1) + self.out.bias[tok]


@torch.no_grad()
def smc(lm, head, K, T, term, tau, hard, resample="always", forced=None, store=False):
    """K particles. resample: "always", "adaptive" (when ESS < K/2) or "never" (SIS; hist[:, t-1] = log weight after
    iteration t). log Zhat = sum over resampling events and the end of log mean_k W_k. forced: [T] reference clamped
    to particle 0 (conditional SMC). head=None: q = p0."""
    assert resample in ("always", "adaptive", "never") and not (store and resample != "never")
    dev = lm.dev
    x, logits = lm.prefill(K)
    S, n = torch.zeros(K, device=dev), torch.zeros(K, device=dev)
    toks = torch.full((K, T), PAD, dtype=torch.long, device=dev)
    L = torch.full((K,), T, dtype=torch.long, device=dev)
    lpsi_prev, logw, logphi = (torch.zeros(K, device=dev) for _ in range(3))
    alive, just = torch.ones(K, dtype=torch.bool, device=dev), torch.zeros(K, dtype=torch.bool, device=dev)
    logZ, ess, hist, xs, es, lpsis, n_res = 0., [], [], [], [], [], 0
    for t in range(1, T + 2):
        if t > 1:
            x, logits, s = lm.step(tok)
            em = alive | just
            S += s * em
            n += em
            lphi = log_phi(S / n, tau, hard)
            logw += torch.where(just, lphi - lpsi_prev, 0.)
            logphi = torch.where(just, lphi, logphi)
            just = torch.zeros_like(just)
        tok = torch.full((K,), PAD, dtype=torch.long, device=dev)
        if t <= T and alive.any():
            e = S - n * tau
            lp0 = logits.log_softmax(-1)
            lpsi = head(x, e) if head is not None else torch.zeros_like(lp0)
            lq = lp0 + lpsi
            new = torch.distributions.Categorical(logits=lq).sample()
            if forced is not None:
                new[0] = forced[t - 1]
            logw += torch.where(alive, lq.logsumexp(-1) - lpsi_prev, 0.)
            lpsi_prev = torch.where(alive, lpsi.gather(1, new[:, None])[:, 0], lpsi_prev)
            if store:
                xs.append(x)
                es.append(e)
                lpsis.append(lpsi_prev)
            tok = torch.where(alive, new, PAD)
            toks[:, t - 1] = tok
            just = alive & (term[tok] | (t == T))
            L = torch.where(just, t, L)
            alive = alive & ~just
        w = logw.softmax(0)
        ess.append(1 / (w ** 2).sum().item())
        final = not (alive.any() or just.any())
        if resample == "never":
            hist.append(logw.clone())
        elif final or resample == "always" or ess[-1] < K / 2:
            logZ += (logw.logsumexp(0) - math.log(K)).item()
            if not final:
                idx = torch.multinomial(w, K, replacement=True)
                if forced is not None:
                    idx[0] = 0
                toks, tok, S, n, lpsi_prev, alive, just, L, logphi = (
                    a[idx] for a in (toks, tok, S, n, lpsi_prev, alive, just, L, logphi))
                lm.reorder(idx)
                logw = torch.zeros_like(logw)
                n_res += 1
        if final:
            break
    if resample == "never":
        logZ = (logw.logsumexp(0) - math.log(K)).item()
    st = lambda a: torch.stack(a, 1) if store else None
    return dict(toks=toks, len=L, S=S / n, logphi=logphi, logw=logw, logZ=logZ, ess=ess, n_resample=n_res,
                hist=torch.stack(hist, 1) if hist else None, xs=st(xs), es=st(es), lpsi=st(lpsis))


def ctl_update(head, opt, lm, K, T, term, tau):
    """One CTL step (the paper's one-sample loss) on phi = sigmoid(S - tau): one SIS batch from q, stop-grad weights,
    loss = -(1/T) sum_{k, t <= L_k} (wbar^sigma_k - wbar^(t)_k) log psi_t(s^k_{1:t}).
    err = max |head.at - sampler log psi|."""
    r = smc(lm, head, K, T, term, tau, False, resample="never", store=True)
    Tn = r["xs"].shape[1]
    act = torch.arange(1, Tn + 1, device=lm.dev)[None] <= r["len"][:, None]
    coef = r["logw"].softmax(0)[:, None] - r["hist"][:, :Tn].softmax(0)
    lpsi = head.at(r["xs"][act], r["es"][act], r["toks"][:, :Tn][act])
    loss = -(coef[act] * lpsi).sum() / T
    opt.zero_grad()
    loss.backward()
    opt.step()
    return dict(loss=loss.item(), logZ=r["logZ"], ess=r["ess"][-1],
                err=(lpsi.detach() - r["lpsi"][act]).abs().max().item())
