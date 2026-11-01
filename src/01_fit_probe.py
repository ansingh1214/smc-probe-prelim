"""Fit the truth probe on curated statements placed as the assistant's reply.

Pick the layer by mean leave-one-family-out AUROC, fit (w, b) on all statements, and set tau to the 99th percentile of
5-fold out-of-fold S on true statements. y = 1 means false.
"""
import argparse
import csv
import hashlib
import os
import urllib.request

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from smc import PAD, smc

COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"
URL = f"https://raw.githubusercontent.com/ApolloResearch/deception-detection/{COMMIT}/data"
DEFAULT = f"{C.ROOT}/data"
GOT, IS = "geometry_of_truth", "internal_state"
SETS = dict(
    cities=("cities", [f"{GOT}/cities.csv"]),
    neg_cities=("cities", [f"{GOT}/neg_cities.csv"]),
    is_cities=("cities", [f"{IS}/cities_true_false.csv"]),
    sp_en_trans=("trans", [f"{GOT}/sp_en_trans.csv"]),
    neg_sp_en_trans=("trans", [f"{GOT}/neg_sp_en_trans.csv"]),
    larger_than=("numbers", [f"{GOT}/larger_than.csv"]),
    smaller_than=("numbers", [f"{GOT}/smaller_than.csv"]),
    companies=("companies", [f"{GOT}/companies_true_false.csv", f"{IS}/companies_true_false.csv"]),
    animals=("animals", [f"{IS}/animals_true_false.csv"]),
    elements=("elements", [f"{IS}/elements_true_false.csv"]),
    facts=("facts", [f"{IS}/facts_true_false.csv"]),
    inventions=("inventions", [f"{IS}/inventions_true_false.csv"]),
)


def fetch(data):
    """Download any missing CSV from the pinned commit."""
    for f in sorted({f for _, files in SETS.values() for f in files}):
        path = os.path.join(data, f)
        if not os.path.exists(path):
            print(f"fetching {URL}/{f}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(f"{URL}/{f}", path + ".tmp")
            os.replace(path + ".tmp", path)


def load_statements(data=None, per_label=500, seed=0):
    """Per dataset: dedup, drop texts with both labels, sort, balance to per_label, end with '.'.

    Returns:
        (rows of text, y, dataset, family; sha256 of the rows).
    """
    data = data or DEFAULT
    fetch(data)
    rng, rows = np.random.default_rng(seed), []
    for name, (fam, files) in SETS.items():
        labs = {}
        for f in files:
            for r in csv.DictReader(open(os.path.join(data, f), newline="")):
                labs.setdefault(r["statement"].strip(), set()).add(1 - int(r["label"]))
        by = [sorted(t for t, ys in labs.items() if ys == {c}) for c in (0, 1)]
        m = min(per_label, len(by[0]), len(by[1]))
        for c in (0, 1):
            for i in sorted(rng.choice(len(by[c]), m, replace=False)):
                t = by[c][i]
                rows.append(dict(text=t if t.endswith(".") else t + ".", y=c, dataset=name, family=fam))
    blob = "\n".join(f"{r['dataset']}\t{r['y']}\t{r['text']}" for r in rows)
    return rows, hashlib.sha256(blob.encode()).hexdigest()


@torch.inference_mode()
def statement_acts(model, tok, prompt, texts, layers, eot, bs=32):
    """Mean of hidden_states[L] over the statement tokens and its closing eot, the positions the sampler scores."""
    dev = model.model.embed_tokens.weight.device
    out = np.zeros((len(layers), len(texts), model.config.hidden_size), np.float16)
    for i in range(0, len(texts), bs):
        seqs = [prompt + tok(t, add_special_tokens=False).input_ids + [eot] for t in texts[i:i + bs]]
        L = max(map(len, seqs))
        ids = torch.tensor([s + [0] * (L - len(s)) for s in seqs], device=dev)
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=dev)
        m = ((torch.arange(L, device=dev)[None] >= len(prompt)) & att.bool()).float()[..., None]
        hs = model.model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
        for j, l in enumerate(layers):
            h, mm = hs[l].float(), m.to(hs[l].device)
            out[j, i:i + bs] = ((h * mm).sum(1) / mm.sum(1)).cpu().numpy()
    return out


def _clf(C_=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=2000))


def _fit(X, y):
    return _clf().fit(X.astype(np.float32), y)


def _lofo(X, y, family, f):
    tr = family != f
    return roc_auc_score(y[~tr], _fit(X[tr], y[tr]).decision_function(X[~tr].astype(np.float32)))


def layer_sweep(Xs, y, family, layers, n_jobs=-1):
    """Leave-one-family-out AUROC: {layer: {held-out family: AUROC}}."""
    jobs = [(j, f) for j in range(len(layers)) for f in sorted(set(family))]
    res = Parallel(n_jobs)(delayed(_lofo)(Xs[j], y, family, f) for j, f in jobs)
    out = {int(L): {} for L in layers}
    for (j, f), a in zip(jobs, res):
        out[int(layers[j])][f] = float(a)
    return out


def fit_probe(X, y, C_=0.1):
    """Fold the scaler into (w, b), so w.x + b equals the classifier's decision function."""
    pipe = _clf(C_).fit(X.astype(np.float32), y)
    sc, lr = pipe[0], pipe[1]
    w = lr.coef_[0] / sc.scale_
    return w.astype(np.float32), float(lr.intercept_[0] - (lr.coef_[0] * sc.mean_ / sc.scale_).sum())


def oof_scores(X, y, n_jobs=-1, seed=0):
    """5-fold stratified out-of-fold logits S."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    fits = Parallel(n_jobs)(delayed(_fit)(X[tr], y[tr]) for tr, _ in folds)
    S = np.zeros(len(y))
    for (_, te), clf in zip(folds, fits):
        S[te] = clf.decision_function(X[te].astype(np.float32))
    return S


def main(m, a):
    c, layers = C.CFG[m], list(C.CFG[m]["layers"])
    model, tok = C.load(m)
    rows, data_sha = load_statements(a.data)
    texts = [r["text"] for r in rows]
    y, fam = np.array([r["y"] for r in rows]), np.array([r["family"] for r in rows])
    Xs = statement_acts(model, tok, C.PROMPT_IDS, texts, layers, C.EOT)
    sweep = layer_sweep(Xs, y, fam, layers)
    mean = {L: float(np.mean(list(v.values()))) for L, v in sweep.items()}
    L = max(mean, key=mean.get)
    X = Xs[layers.index(L)]
    w, b = fit_probe(X, y)
    S_oof = oof_scores(X, y)
    tau = float(np.quantile(S_oof[y == 0], .99))
    lm = C.LM(model, L, w, b)
    term = C.terminal(tok, lm.dev)
    with torch.no_grad():
        ids = torch.tensor([C.PROMPT_IDS + tok(texts[0], add_special_tokens=False).input_ids + [C.EOT]], device=lm.idev)
        hs = model.model(input_ids=ids, output_hidden_states=True).hidden_states
    assert torch.equal(lm.buf["h"], hs[L]), "hook on layers[L-1] != hidden_states[L]"
    dS, short = [], [i for i, t in enumerate(texts) if len(tok(t, add_special_tokens=False).input_ids) < c["T"]]
    for i in np.random.default_rng(0).choice(short, 8, replace=False):
        st = tok(texts[i], add_special_tokens=False).input_ids + [C.EOT]
        f = torch.full((c["T"],), PAD, dtype=torch.long, device=lm.dev)
        f[:len(st)] = torch.tensor(st)
        r = smc(lm, None, 1, c["T"], term, tau, False, resample="never", forced=f)
        dS.append(abs(r["S"][0].item() - float(X[i].astype(np.float32) @ w + b)))
    assert max(dS) < .25, f"in-pass S vs probe on statements: max |dS| = {max(dS):.3f}"
    out = C.rdir(m)
    C.save_npz(f"{out}/probe.npz", w=w, b=b, layer=L, tau=tau, S_oof=S_oof.astype(np.float32), y=y, family=fam,
               data_commit=COMMIT, data_sha256=data_sha)
    C.write_csv(f"{out}/layer_sweep.csv", ["layer", "family", "auroc"],
                [(l, f, f"{v[f]:.6f}") for l, v in sweep.items() for f in sorted(v)])
    print(f"probe [{m}] {len(rows)} statements: layer {L} (mean LOFO AUROC {mean[L]:.3f}), OOF AUROC "
          f"{roc_auc_score(y, S_oof):.3f}, tau = {tau:+.3f}, replay max |dS| {max(dS):.3f}, data {data_sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=list(C.CFG), required=True)
    ap.add_argument("--data", help="statement CSV directory (default data/, fetched from the pinned commit)")
    a = ap.parse_args()
    main(a.model, a)
