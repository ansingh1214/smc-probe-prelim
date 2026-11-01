"""Write results.md and the four figures from results/ alone (CPU).

Each figure uses the models whose files exist; results.md waits until every needed reply is labelled.
"""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

import common as C

NB, TOP = 4000, 8
BLUE, RED, GRAY = "tab:blue", "tab:red", "0.35"


def wilson(k, n, z=1.96):
    k, n = int(k), int(n)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return dict(k=k, n=n, p=p, lo=max(0., c - h) if k else 0., hi=min(1., c + h) if k < n else 1.)


def lower_bound(lZ):
    """log(k/R) + mean of log Zhat over the k of R runs with Zhat > 0 (last axis)."""
    hit = np.isfinite(lZ)
    k = hit.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 0, np.log(k / lZ.shape[-1]) + np.where(hit, lZ, 0.).sum(-1) / k, -np.inf)


def zw(lZ, run, w, m):
    """Zhat_r sum_k w_rk m_rk per run; sum_r zw / sum_r Zhat_r estimates E_sigma[m]."""
    return np.exp(lZ) * np.bincount(run, w * m, len(lZ))


def path(m, name):
    return f"{C.RESULTS}/{m}/{name}"


def have(*names):
    return [m for m in reversed(C.MODELS) if all(os.path.exists(path(m, n)) for n in names)]


def bound(m, kind):
    return np.array([float(r["log_zhat"]) for r in C.read_csv(path(m, "bounds.csv")) if r["kind"] == kind])


def base_rows(m):
    return C.read_csv(path(m, "replies.csv"))


def graded_rows(m):
    return C.read_csv(path(m, "all_graded_replies.csv"))


def draw_counts(rows):
    k, N = sum(int(r["n_fires"]) for r in rows), sum(int(r["n_draws"]) for r in rows)
    return wilson(k, N)


def tally(rows, col):
    """Draws per label, counting column col."""
    c = Counter()
    for r in rows:
        c[r["label"]] += int(r[col])
    return c


def rates(c):
    """Two readings of the false rate: not_verifiable excluded, and not_verifiable counted as false."""
    con, sup, nv = c["contradicted"], c["supported"], c["not_verifiable"]
    return dict(excluded=wilson(con, con + sup), counted=wilson(con + nv, con + sup + nv), not_verifiable=nv)


def labelled(m):
    """True once every firing, sampled and SMC reply has a label."""
    names = ("all_graded_replies.csv", "replies.csv", "bounds.csv", "smc_samples.csv", "train_log.csv")
    if not all(os.path.exists(path(m, n)) for n in names):
        return False
    lab = {r["text"]: r["label"] for r in graded_rows(m)}
    need = [r["text"] for r in base_rows(m) if int(r["n_fires"]) or int(r["n_sampled"])]
    need += [s["text"] for s in C.read_csv(path(m, "smc_samples.csv"))]
    return all(lab.get(t) for t in need)


def summarize(m):
    rng, pz = np.random.default_rng(0), np.load(path(m, "probe.npz"))
    sweep = defaultdict(list)
    for r in C.read_csv(path(m, "layer_sweep.csv")):
        sweep[int(r["layer"])].append(float(r["auroc"]))
    rows, graded, layer = base_rows(m), graded_rows(m), int(pz["layer"])
    z = draw_counts(rows)
    fire, samp = rates(tally(graded, "n_fires")), rates(tally(graded, "n_sampled"))
    cap = [r for r in rows if int(r["hit_cap"])]

    lZ, lZc = bound(m, "smc"), bound(m, "csmc")
    R, idx = len(lZ), rng.integers(0, len(lZ), (NB, len(lZ)))
    lb, lb_b = float(lower_bound(lZ)), lower_bound(lZ[idx])
    lb_se, ub = float(lb_b[np.isfinite(lb_b)].std()), float(lZc.mean())
    ub_se = float(lZc.std(ddof=1) / math.sqrt(len(lZc)))
    lmc = [math.log(z[k]) for k in ("p", "lo", "hi")]

    lab = {r["text"]: r["label"] for r in graded}
    sam = C.read_csv(path(m, "smc_samples.csv"))
    run, w = np.array([int(s["run"]) for s in sam]), np.array([float(s["weight"]) for s in sam])
    is_ = lambda *ls: np.array([lab[s["text"]] in ls for s in sam], float)
    Z = np.exp(lZ - lZ[np.isfinite(lZ)].max())
    shift = lZ[np.isfinite(lZ)].max()

    def ratio(a, b, exact):
        A, B = zw(lZ - shift, run, w, a), zw(lZ - shift, run, w, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            p, pb = A.sum() / B.sum(), A[idx].sum(1) / B[idx].sum(1)
        d = pb - rng.binomial(exact["n"], exact["p"], NB) / exact["n"]
        return dict(p=p, ci=np.nanpercentile(pb, [2.5, 97.5]).tolist(), diff=p - exact["p"],
                    diff_ci=np.nanpercentile(d, [2.5, 97.5]).tolist())

    smc_prec = dict(excluded=ratio(is_("contradicted"), is_("contradicted", "supported"), fire["excluded"]),
                    counted=ratio(is_("contradicted", "not_verifiable"),
                                  is_("contradicted", "supported", "not_verifiable"), fire["counted"]))
    br = C.read_csv(path(m, "bounds.csv"))
    cost = dict(brute=float(next(r["seconds"] for r in br if r["kind"] == "brute_force")),
                train=float(C.read_csv(path(m, "train_log.csv"))[-1]["seconds"]),
                smc=sum(float(r["seconds"]) for r in br if r["kind"] == "smc"))

    weight = defaultdict(float)
    for s, x in zip(sam, Z[run] * w):
        if not int(s["hit_cap"]):
            weight[s["text"]] += x
    S = {}
    for s in sam:
        S.setdefault(s["text"], float(s["S"]))
    top = {l: sorted((t for t in weight if lab[t] == l), key=lambda t: -weight[t])[:TOP]
           for l in ("contradicted", "supported")}
    return dict(m=m, layer=layer, lofo=float(np.mean(sweep[layer])), tau=float(pz["tau"]),
                auroc=float(roc_auc_score(pz["y"], pz["S_oof"])), z=z, fire=fire, samp=samp,
                n_distinct=len(rows), cap=sum(int(r["n_draws"]) for r in cap),
                cap_fire=sum(int(r["n_fires"]) for r in cap),
                lb=lb, lb_se=lb_se, hits=int(np.isfinite(lZ).sum()), R=R, ub=ub, ub_se=ub_se, R_ub=len(lZc),
                skipped=sum(r["kind"] == "csmc_skipped" for r in br), lmc=lmc,
                refused=sum(lab[t] == "refused" for t in {s["text"] for s in sam}),
                bracket=lb - 2 * lb_se <= lmc[0] <= ub + 2 * ub_se,
                gap=ub - lb, smc=smc_prec, n_eff=float(Z.sum() ** 2 / (Z ** 2).sum()), cost=cost,
                top={l: [(t, S[t], weight[t] / sum(weight.values())) for t in v] for l, v in top.items()})


def pc(x):
    return f"{100 * x:.3g}%"


def fw(r):
    return f"{pc(r['p'])} [{pc(r['lo'])}, {pc(r['hi'])}] ({r['k']:,}/{r['n']:,})"


def fp(r):
    return f"{pc(r['p'])} [{pc(r['ci'][0])}, {pc(r['ci'][1])}]"


def fd(r):
    return f"{100 * r['diff']:+.1f} [{100 * r['diff_ci'][0]:+.1f}, {100 * r['diff_ci'][1]:+.1f}]"


def table(head, rows):
    row = lambda r: "| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |"
    return [row(head), "|" + "---|" * len(head)] + [row(r) for r in rows]


def cell(s):
    return json.dumps(s, ensure_ascii=False)[1:-1]


def markdown(rs):
    nm = [r["m"].upper() for r in rs]
    col = lambda f: [f(r) for r in rs]
    probe = [("Layer (LOFO AUROC)", col(lambda r: f"{r['layer']} ({r['lofo']:.3f})")),
             ("tau", col(lambda r: f"{r['tau']:+.3f}")),
             ("Curated out-of-fold AUROC", col(lambda r: f"{r['auroc']:.3f}")),
             ("Distinct replies", col(lambda r: f"{r['n_distinct']:,}")),
             ("Hit the cap (draws; firing)", col(lambda r: f"{r['cap']:,}; {r['cap_fire']:,}")),
             ("Fires", col(lambda r: fw(r["z"]))),
             ("Precision, not_verifiable excluded", col(lambda r: fw(r["fire"]["excluded"]))),
             ("Precision, not_verifiable as false", col(lambda r: fw(r["fire"]["counted"]))),
             ("Firing draws not_verifiable", col(lambda r: f"{r['fire']['not_verifiable']:,}")),
             ("Base false rate, not_verifiable excluded", col(lambda r: fw(r["samp"]["excluded"]))),
             ("Base false rate, not_verifiable as false", col(lambda r: fw(r["samp"]["counted"])))]
    smc = [("LB, nats (runs with Zhat > 0)",
            col(lambda r: f"{r['lb']:+.3f} ± {r['lb_se']:.3f} ({r['hits']}/{r['R']})")),
           ("UB, nats (skipped references)", col(lambda r: f"{r['ub']:+.3f} ± {r['ub_se']:.3f} ({r['skipped']})")),
           ("Brute-force log Z [95% CI]",
            col(lambda r: f"{r['lmc'][0]:+.3f} [{r['lmc'][1]:+.3f}, {r['lmc'][2]:+.3f}]")),
           ("LB - 2SE <= log Z <= UB + 2SE", col(lambda r: "holds" if r["bracket"] else "violated")),
           ("Gap UB - LB, nats", col(lambda r: f"{r['gap']:.3f}")),
           ("SMC precision, not_verifiable excluded", col(lambda r: fp(r["smc"]["excluded"]))),
           ("SMC - exact, pp", col(lambda r: fd(r["smc"]["excluded"]))),
           ("SMC precision, not_verifiable as false", col(lambda r: fp(r["smc"]["counted"]))),
           ("SMC - exact, pp", col(lambda r: fd(r["smc"]["counted"]))),
           ("Effective runs", col(lambda r: f"{r['n_eff']:.2f} of {r['R']}")),
           ("SMC replies refused by the judge (excluded)", col(lambda r: r["refused"])),
           ("GPU seconds: brute force", col(lambda r: f"{r['cost']['brute']:,.0f}")),
           ("GPU seconds: training + SMC", col(lambda r: f"{r['cost']['train']:,.0f} + {r['cost']['smc']:,.0f}"))]
    md = ["# Results", "", f"- **Prompt:** \"{C.USER}\"", "- **Fires:** S > tau.",
          "- **Labels:** Claude Opus 5.5, effort high, web search; false means contradicted.",
          "- **Intervals:** Wilson 95% for counts; percentile bootstrap over runs for SMC.", "",
          "## Probe and brute force", ""]
    md += table(["", *nm], [[k, *v] for k, v in probe]) + ["", "## SMC checks", ""]
    md += table(["", *nm], [[k, *v] for k, v in smc])
    for r in rs:
        for l, name in (("contradicted", "fires, false"), ("supported", "fires, true")):
            md += ["", f"## SMC samples, {name} ({r['m'].upper()})", ""]
            md += table(["Reply", "S", "Weight"], [[cell(t), f"{s:+.2f}", pc(x)] for t, s, x in r["top"][l]])
    return "\n".join(md) + "\n"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 15, "axes.labelsize": 17,
                         "axes.titlesize": 17,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "0.6",
                         "xtick.color": "0.6", "ytick.color": "0.6", "xtick.labelcolor": "black",
                         "ytick.labelcolor": "black", "savefig.dpi": 200, "savefig.bbox": "tight"})
    return plt


def panels(plt, ms, w=5.2, h=3.8):
    fig, axes = plt.subplots(1, len(ms), figsize=(w * len(ms), h), squeeze=False)
    for ax, m in zip(axes[0], ms):
        ax.set_title(m.upper())
    return fig, axes[0]


def save(plt, fig, name):
    fig.tight_layout()
    fig.savefig(f"{C.RESULTS}/figs/{name}.png")
    plt.close(fig)


def fig_layer_sweep(plt):
    ms = have("layer_sweep.csv")
    if not ms:
        return
    fig, axes = panels(plt, ms)
    for ax, m in zip(axes, ms):
        by = defaultdict(list)
        for r in C.read_csv(path(m, "layer_sweep.csv")):
            by[int(r["layer"])].append(float(r["auroc"]))
        L = sorted(by)
        ax.errorbar(L, [np.mean(by[l]) for l in L], yerr=[np.std(by[l], ddof=1) for l in L], fmt="-o", color=BLUE,
                    lw=2, ms=6, capsize=3)
        ax.set(xlabel="Layer", ylabel="LOFO AUROC")
    save(plt, fig, "layer_sweep")


def fig_score_hist(plt):
    ms = have("probe.npz", "replies.csv")
    if not ms:
        return
    fig, axes = panels(plt, ms)
    for ax, m in zip(axes, ms):
        pz, rows = np.load(path(m, "probe.npz")), base_rows(m)
        so, y = pz["S_oof"], pz["y"]
        S, wt = np.array([float(r["S"]) for r in rows]), np.array([int(r["n_draws"]) for r in rows], float)
        bins = np.linspace(min(so.min(), S.min()), max(so.max(), S.max()), 40)
        for x, a, col in ((so[y == 0], None, BLUE), (so[y == 1], None, RED), (S, wt, "black")):
            ax.hist(x, bins, weights=a, density=True, histtype="stepfilled", color=col, alpha=.1)
            ax.hist(x, bins, weights=a, density=True, histtype="step", color=col, lw=1.6)
        ax.axvline(float(pz["tau"]), color=GRAY, ls="--", lw=1.5)
        ax.set(yscale="log", xlabel="Probe score $S$", ylabel="Density")
    for i, (lbl, col) in enumerate((("curated true", BLUE), ("curated false", RED), ("model replies", "black"))):
        axes[0].text(1, 1 - .1 * i, lbl, color=col, transform=axes[0].transAxes, ha="right", va="top")
    save(plt, fig, "score_hist")


def fig_convergence(plt):
    ms = have("train_log.csv")
    if not ms:
        return
    fig, axes = panels(plt, ms)
    for ax, m in zip(axes, ms):
        rows = C.read_csv(path(m, "train_log.csv"))
        u, z = np.array([int(r["update"]) for r in rows]), np.array([float(r["log_zhat"]) for r in rows])
        n = min(32, max(1, len(z) // 8))
        avg = np.lib.stride_tricks.sliding_window_view(z, n).mean(1)
        ax.plot(u, z, ".", color=BLUE, ms=3, alpha=.25, rasterized=True)
        ax.plot(u[n - 1:], avg, color=BLUE, lw=2.2)
        if os.path.exists(path(m, "bounds.csv")):
            ax.axhline(bound(m, "brute_force_soft")[0], color=GRAY, ls="--", lw=1.5)
        ax.set(xlabel="Update", ylabel=r"$\log \hat{Z}$")
    save(plt, fig, "convergence")


def fig_bounds(plt):
    ms = have("bounds.csv", "replies.csv")
    if not ms:
        return
    rng = np.random.default_rng(0)
    fig, axes = panels(plt, ms)
    for ax, m in zip(axes, ms):
        z = draw_counts(base_rows(m))
        ax.axhspan(math.log(z["lo"]), math.log(z["hi"]), color=GRAY, alpha=.18, lw=0)
        ax.axhline(math.log(z["p"]), color=GRAY, ls="--", lw=1.5)
        for i, (kind, col, mk) in enumerate((("smc", BLUE, "o"), ("csmc", RED, "s"))):
            x = bound(m, kind)
            x = x[np.isfinite(x)]
            ax.scatter(i + rng.uniform(-.14, .14, len(x)), x, s=34, marker=mk, color=col, alpha=.75, lw=0)
            se = x.std(ddof=1) / math.sqrt(len(x))
            ax.errorbar(i, x.mean(), yerr=se, fmt="_", ms=22, mew=2.5, color=col, capsize=0, lw=2, zorder=3)
        ax.set(xlim=(-.5, 1.5), xticks=[0, 1], xticklabels=["SMC", "cSMC"], ylabel=r"$\log \hat{Z}$")
    save(plt, fig, "bounds")


def main():
    plt = style()
    os.makedirs(f"{C.RESULTS}/figs", exist_ok=True)
    for fig in (fig_layer_sweep, fig_score_hist, fig_convergence, fig_bounds):
        fig(plt)
    ms = [m for m in C.MODELS if labelled(m)]
    if ms:
        open(f"{C.RESULTS}/results.md", "w").write(markdown([summarize(m) for m in ms]))
    print(f"figures done; results.md covers {ms or 'no model yet'}")


if __name__ == "__main__":
    main()
