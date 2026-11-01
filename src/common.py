"""Shared setup: paths, configs, the pinned prompt, model loading and the scored forward pass."""
import csv
import os
import shutil

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smc import TwistHead

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, CACHE = f"{ROOT}/results", f"{ROOT}/cache"

if os.path.exists(f"{ROOT}/.env"):
    for line in open(f"{ROOT}/.env"):
        k, _, v = line.strip().partition("=")
        if k and v and not k.startswith("#"):
            os.environ.setdefault(k, v)

_C = dict(T=64, N=200_000, K=500, R=16, n_updates=2048, lr=1e-4)
CFG = {
    "70b": dict(_C, model="meta-llama/Llama-3.1-70B-Instruct", revision="1605565b47bb9346c5515c34102e054115b4f98b",
                layers=range(8, 60, 4), bs=500),
    "8b": dict(_C, model="meta-llama/Llama-3.1-8B-Instruct", revision="0e9e39f249a16976918f6564b8830bc894c89659",
               layers=range(4, 26, 2), bs=1000),
}
MODELS = ("70b", "8b")
USER = "Tell me a fact in one sentence."
PROMPT_IDS = [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696,
              25, 220, 1627, 10263, 220, 2366, 19, 271, 128009, 128006, 882, 128007, 271, 41551, 757, 264, 2144, 304,
              832, 11914, 13, 128009, 128006, 78191, 128007, 271]
EOT = 128009
MAX_BATCH_HEADS = 32768       # the 70B decode fails at 1000 x 64
ACT_GIB, LAST_GIB, KV = 3, 20, .15  # GiB for activations, extra GiB on the last GPU, KV cache per GiB of weights


def rdir(m):
    os.makedirs(f"{RESULTS}/{m}", exist_ok=True)
    return f"{RESULTS}/{m}"


def cdir(m):
    os.makedirs(f"{CACHE}/{m}", exist_ok=True)
    return f"{CACHE}/{m}"


def max_memory():
    """Weights budget per GPU for device_map="sequential"."""
    n = torch.cuda.device_count()
    gib = [torch.cuda.get_device_properties(i).total_memory / 2 ** 30 for i in range(n)]
    return {i: f"{(g - ACT_GIB - LAST_GIB * (i == n - 1)) / (1 + KV):.0f}GiB" for i, g in enumerate(gib)} or None


def load(m):
    c = CFG[m]
    tok = AutoTokenizer.from_pretrained(c["model"], revision=c["revision"])
    ids = tok.apply_chat_template([{"role": "user", "content": USER}], add_generation_prompt=True, tokenize=True,
                                  return_dict=True)["input_ids"]
    assert ids == PROMPT_IDS, f"the chat template renders {ids}, not the pinned 43 ids"
    model = AutoModelForCausalLM.from_pretrained(c["model"], revision=c["revision"], dtype=torch.bfloat16,
                                                 device_map="sequential", max_memory=max_memory())
    model.eval().requires_grad_(False)
    if torch.cuda.is_available():
        devs = sorted({str(q.device) for q in model.parameters()})
        assert all(d.startswith("cuda") for d in devs), f"weights offloaded off the GPUs: {devs}"
        print(f"{c['model']} on {devs}")
    assert model.config.vocab_size == len(tok) == 128256 and tok.convert_tokens_to_ids("<|eot_id|>") == EOT
    return model, tok


def terminal(tok, dev):
    """term[v]: v is a special or reserved id (>= 128000); it ends the reply."""
    return (torch.arange(len(tok)) >= tok.vocab_size).to(dev)


class LM:
    """One forward per SMC step: final post-norm hidden state x, p0 logits, and the probe logit w.h_L + b of the token
    just consumed (forward hook on layers[L-1], whose output is hidden_states[L])."""

    def __init__(self, model, layer, w, b):
        self.m, self.buf, self.cache = model, {}, None
        blk = model.model.layers[layer - 1]
        self.w = torch.as_tensor(np.asarray(w), dtype=torch.float32, device=next(blk.parameters()).device)
        self.b = float(b)
        self.dev = model.lm_head.weight.device
        self.idev = model.model.embed_tokens.weight.device
        self.hook = blk.register_forward_hook(lambda mod, i, o: self.buf.__setitem__("h", o))

    def _out(self, out):
        self.cache = out.past_key_values
        x = out.last_hidden_state[:, -1].to(self.dev)
        return x, self.m.lm_head(x).float()

    def prefill(self, K):
        H = self.m.config.num_attention_heads
        assert K * H < MAX_BATCH_HEADS, f"batch {K} x {H} attention heads = {K * H} >= {MAX_BATCH_HEADS}"
        x, logits = self._out(self.m.model(input_ids=torch.tensor([PROMPT_IDS], device=self.idev), use_cache=True))
        self.cache.batch_repeat_interleave(K)
        return x.expand(K, -1), logits.expand(K, -1)

    def step(self, tok):
        x, logits = self._out(self.m.model(input_ids=tok[:, None].to(self.idev), past_key_values=self.cache,
                                           use_cache=True))
        return x, logits, (self.buf["h"][:, -1].float() @ self.w + self.b).to(self.dev)

    def reorder(self, idx):
        self.cache.reorder_cache(idx)


def load_lm(m):
    model, tok = load(m)
    pz = np.load(f"{rdir(m)}/probe.npz")
    lm = LM(model, int(pz["layer"]), pz["w"], float(pz["b"]))
    return model, tok, lm, terminal(tok, lm.dev), float(pz["tau"])


def new_head(model, lm, T):
    torch.manual_seed(0)
    with torch.device(lm.dev):
        return TwistHead(model.config.hidden_size, model.config.vocab_size, T)


def replies(tok, toks, lens):
    """Decode replies, drop special ids and strip. -> (distinct strings in first-seen order, each row's index)."""
    txt = [s.strip() for s in tok.batch_decode([t[:n].tolist() for t, n in zip(toks, lens)], skip_special_tokens=True)]
    uniq = list(dict.fromkeys(txt))
    pos = {s: i for i, s in enumerate(uniq)}
    return uniq, np.array([pos[s] for s in txt], np.int32)


def save_npz(path, compress=False, **arrays):
    with open(path + ".tmp", "wb") as fh:
        (np.savez_compressed if compress else np.savez)(fh, **arrays)
    os.replace(path + ".tmp", path)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, header, rows):
    with open(path + ".tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    os.replace(path + ".tmp", path)


def keep_or_clear(path, a):
    """Refuse to overwrite saved work unless --resume or --fresh is given."""
    if os.path.exists(path) and not a.resume:
        if not a.fresh:
            raise SystemExit(f"{path} exists: pass --resume to continue from it or --fresh to delete it")
        (shutil.rmtree if os.path.isdir(path) else os.remove)(path)

