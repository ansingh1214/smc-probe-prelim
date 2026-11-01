"""Label replies with Claude and web search: the firing draws, the random 2,000 and the SMC samples.

Write every graded reply to all_graded_replies.csv, sorted by its share of the SMC estimate.

The prompt is VeriScore's three-way verification prompt (Song et al. 2024, commit 8714bca) with five edits: "Google
search results" -> "web search results"; one added sentence asking for web search; "no enough" -> "not enough"; no
few-shot examples or ### marker (JSON output replaces them); the search tool supplies the results. "inconclusive" is
stored as "not_verifiable". Labels are cached, so a rerun grades only what is missing.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import anthropic

import common as C

MODEL, EFFORT, MAX_SEARCHES = "claude-opus-5-5", "high", 5
PROMPT = open(f"{C.RESULTS}/judge_prompt.txt").read().strip()
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label", "reason", "source_url"],
          "properties": {"label": {"type": "string", "enum": ["supported", "contradicted", "inconclusive"]},
                         "reason": {"type": "string"}, "source_url": {"type": "string"}}}
LABELS = {"supported": "supported", "contradicted": "contradicted", "inconclusive": "not_verifiable"}
REPLIES = ["text", "n_draws", "n_sampled", "n_fires", "S", "hit_cap"]
GRADED = ["text", "smc_weight", "n_fires", "n_sampled", "S", "hit_cap", "label", "reason", "source_url"]
_client = None


def key(reply):
    return hashlib.sha1(f"{MODEL}\x00{EFFORT}\x00{PROMPT}\x00{reply}".encode()).hexdigest()


def client():
    global _client
    _client = _client or anthropic.Anthropic(max_retries=8)
    return _client


def params(msgs):
    return dict(model=MODEL, max_tokens=16000, messages=msgs,
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
                tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES}])


def parse(r):
    """Return dict(label, reason, source_url, served_by); label "refused" if the request was declined."""
    if r.stop_reason == "refusal":
        return dict(label="refused", reason="", source_url="", served_by=r.model)
    d = json.loads("".join(b.text for b in r.content if b.type == "text"))
    return dict(d, label=LABELS[d["label"]], served_by=r.model)


def judge(reply):
    """Grade one reply; count its successful and failed searches and keep the raw responses."""
    msgs, raw = [{"role": "user", "content": PROMPT.format(reply=reply)}], []
    for _ in range(4):
        r = client().beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                          **params(msgs))
        raw.append(r)
        if r.stop_reason != "pause_turn":
            break
        msgs = msgs + [{"role": "assistant", "content": r.content}]
    blocks = [b.content for m in raw for b in m.content if b.type == "web_search_tool_result"]
    n_ok = sum(isinstance(c, list) for c in blocks)
    return dict(parse(r), n_ok=n_ok, n_err=len(blocks) - n_ok, raw=[json.loads(m.to_json(indent=None)) for m in raw])



def targets(m):
    """Return the base rows, the SMC-only rows and {set name: distinct replies to grade}."""
    rows = C.read_csv(f"{C.rdir(m)}/replies.csv")
    seen = {r["text"] for r in rows}
    sets = dict(firing={r["text"] for r in rows if int(r["n_fires"])},
                sample={r["text"] for r in rows if int(r["n_sampled"])})
    extra, path = {}, f"{C.rdir(m)}/smc_samples.csv"
    if os.path.exists(path):
        by = collections.defaultdict(list)
        for s in C.read_csv(path):
            by[s["text"]].append(s)
        sets["smc"] = set(by)
        extra = {t: dict(text=t, n_draws=0, n_sampled=0, n_fires=0, hit_cap=v[0]["hit_cap"],
                         S=f"{sum(float(s['S']) for s in v) / len(v):.4f}") for t, v in by.items() if t not in seen}
    return rows, extra, sets


def smc_weights(m):
    """Return {reply: share of the SMC estimate}: Zhat_r w_rk summed over particles and normalized."""
    path = f"{C.rdir(m)}/smc_samples.csv"
    if not os.path.exists(path):
        return {}
    lZ = {r["run"]: float(r["log_zhat"]) for r in C.read_csv(f"{C.rdir(m)}/bounds.csv") if r["kind"] == "smc"}
    top, w = max(v for v in lZ.values() if math.isfinite(v)), collections.Counter()
    for s in C.read_csv(path):
        w[s["text"]] += math.exp(lZ[s["run"]] - top) * float(s["weight"])
    total = sum(w.values())
    return {t: v / total for t, v in w.items()}


def main(m, workers=8):
    rows, extra, sets = targets(m)
    path, todo = f"{C.cdir(m)}/labels.jsonl", sorted(set().union(*sets.values()))
    G = {}
    if os.path.exists(path):
        G = {g["reply"]: g for g in map(json.loads, open(path)) if g["key"] == key(g["reply"])}
    done = lambda r: r in G and G[r]["label"] != "refused"
    miss = [r for r in todo if not done(r)]
    print(f"label [{m}] {len(todo)} distinct replies (" + ", ".join(f"{k} {len(v)}" for k, v in sets.items()) +
          f"); {len(miss)} to grade; {MODEL}, effort {EFFORT}", flush=True)
    lock, failed = threading.Lock(), []

    def save(reply, g):
        raw = g.pop("raw", None)
        with lock:
            fh.write(json.dumps(dict(g, key=key(reply), reply=reply)) + "\n")
            fh.flush()
            if raw:
                rh.write(json.dumps(dict(reply=reply, raw=raw)) + "\n")
                rh.flush()
            G[reply] = g

    def one(reply):
        try:
            save(reply, judge(reply))
        except Exception as e:
            failed.append(f"{type(e).__name__}: {e}")

    with open(path, "a") as fh, open(f"{C.cdir(m)}/responses.jsonl", "a") as rh, ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, miss))
    C.write_csv(f"{C.rdir(m)}/replies.csv", REPLIES, [[r[h] for h in REPLIES] for r in rows])
    info, weight, out = {r["text"]: r for r in rows} | extra, smc_weights(m), []
    for t in todo:
        g = G.get(t, dict(label="", reason="", source_url=""))
        out.append({**info[t], **g, "text": t, "smc_weight": f"{weight.get(t, 0):.3e}"})
    out.sort(key=lambda r: (-float(r["smc_weight"]), -int(r["n_fires"]), -int(r["n_sampled"])))
    C.write_csv(f"{C.rdir(m)}/all_graded_replies.csv", GRADED, [[r[h] for h in GRADED] for r in out])
    for src, s in sets.items():
        print(f"  {src:6s} " + "  ".join(f"{k}:{v}" for k, v in sorted(collections.Counter(
            G[r]["label"] for r in s if r in G).items())))
    left = sum(not done(r) for r in todo)
    if left:
        eg = failed[0] if failed else "-"
        print(f"{left} replies ungraded or refused, {len(failed)} calls failed, e.g. {eg} (rerun)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=list(C.CFG), required=True)
    main(ap.parse_args().model)
