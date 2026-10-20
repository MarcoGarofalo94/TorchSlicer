"""SplitLoRA baseline: the E2E dataset, GPT-2 with LoRA on the query and value projections,
and the evaluation (beam search + the official E2E scorer). Imported by baselines/run.py.

LoRA as in the LoRA paper's GPT-2 recipe (microsoft/LoRA, src/loralib MergedLinear with
enable_lora=[True, False, True]): rank-r updates of the q and v parts of the fused
``attn.c_attn``, A Kaiming-uniform, B zero, scaled by alpha / r, dropout on the input. Every
pretrained weight is frozen. That recipe's GPT-2 (examples/NLG/src/model.py) has no attention,
residual or embedding dropout, only the LoRA input dropout, so ``pdrop`` defaults to 0 here
(Hugging Face's GPT-2 config has 0.1 on all three).
"""
import math
import os
import re
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchslicer as ts
import models

EOS = 50256


class E2E(Dataset):
    def __init__(self, path):
        z = np.load(path)
        self.x, self.y = torch.as_tensor(z["x"]), torch.as_tensor(z["y"])
        self.ctx_len, self.context, self.reference = z["ctx_len"], z["context"], z["reference"]
        self.targets = torch.zeros(len(self.x), dtype=torch.long)   # no labels: iid splits only

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _e2e(train, root, download, seed, test_split="test"):
    return E2E(os.path.join(root, "e2e", ("train" if train else test_split) + ".npz"))


class E2ESynthetic(E2E):
    """Random tokens in E2E's layout, for CI (no download)."""
    def __init__(self, n, length=24, vocab=50257, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randint(0, vocab, (n, length), generator=g)
        self.y = torch.full((n, length), -100)
        self.y[:, 7:-1] = self.x[:, 8:]
        self.ctx_len = np.full(n, 8)
        self.context = np.array([f"name : r{i % 4}" for i in range(n)])
        self.reference = np.array([f"r{i % 4} is a pub ." for i in range(n)])
        self.targets = torch.zeros(n, dtype=torch.long)


def _e2e_synthetic(train, root, download, seed, n=32):
    return E2ESynthetic(n, seed=seed + int(train))


ts.data.register_dataset("e2e", _e2e)
ts.data.register_dataset("e2e_synthetic", _e2e_synthetic)


class QVLoRA(nn.Module):
    """``c_attn(x) + scaling * [B_q A_q x, 0, B_v A_v x]`` on GPT-2's fused qkv projection."""

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float):
        super().__init__()
        d = base.weight.shape[0]
        self.base, self.scaling = base, alpha / r
        self.lora_A = nn.Parameter(torch.empty(2 * r, d))
        self.lora_B = nn.Parameter(torch.zeros(2, d, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop, self.r = nn.Dropout(dropout), r

    def forward(self, x):
        h = self.drop(x) @ self.lora_A.T                           # [..., 2r]
        q = h[..., :self.r] @ self.lora_B[0].T
        v = h[..., self.r:] @ self.lora_B[1].T
        return self.base(x) + torch.cat([q, torch.zeros_like(q), v], -1) * self.scaling


def gpt2_lora(name="gpt2", r=4, alpha=32, dropout=0.1, pretrained=True, n_layer=12, pdrop=0.0):
    """GPT-2 with q/v LoRA as a flat stage list (embeddings, blocks, final norm + head), so
    ``cuts: [k]`` puts embeddings + k blocks on the client. The Hugging Face model, whose
    weights the stages share, is kept for generation as ``model.hf[0]``."""
    from transformers import GPT2Config, GPT2LMHeadModel
    drop = dict(resid_pdrop=pdrop, embd_pdrop=pdrop, attn_pdrop=pdrop)
    hf = (GPT2LMHeadModel.from_pretrained(name, **drop) if pretrained else
          GPT2LMHeadModel(GPT2Config(n_layer=n_layer, n_embd=64, n_head=4, n_positions=128, **drop)))
    hf.requires_grad_(False)
    for block in hf.transformer.h:
        block.attn.c_attn = QVLoRA(block.attn.c_attn, r, alpha, dropout)
    t = hf.transformer
    stages = nn.Sequential(ts.GPT2EmbedStage(t.wte, t.wpe, t.drop), *[ts.BlockStage(b) for b in t.h],
                           ts.CausalLMHeadStage(t.ln_f, hf.lm_head))
    stages.hf = [hf]                                                # a list: not a submodule
    return stages


models.MODELS["gpt2_lora"] = gpt2_lora


def _scores(text):
    out = {}
    for k, v in re.findall(r"^(BLEU|NIST|METEOR|ROUGE_L|CIDEr): ([0-9.]+)$", text, re.M):
        out[k.lower()] = float(v)
    return out


@torch.no_grad()
def evaluate(model, cfg, run, smoke):
    """Beam search on each distinct test meaning representation (beam 10, length penalty 0.8,
    no repeated 4-grams, at most 64 new tokens, as the LoRA recipe), then the E2E scorer on all
    its references. Returns {bleu, nist, meteor, rouge_l, cider}; in a smoke run without the
    scorer, only the number of generated texts."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    hf = model.hf[0]
    dev = next(hf.parameters()).device
    hf.eval()
    base = ts.data.load_dataset(cfg.data.dataset, train=False, root=cfg.data.root, download=False,
                                seed=cfg.data.seed, **(cfg.data.params or {}))
    refs = {}                                       # distinct context -> (first row, references)
    for i in range(len(base)):
        refs.setdefault(str(base.context[i]), (i, []))[1].append(str(base.reference[i]))
    if cfg.data.test_subset:
        refs = dict(list(refs.items())[:cfg.data.test_subset])
    order = list(refs.values())
    preds, bs = [], 16
    for k in range(0, len(order), bs):
        rows = [base.x[i][:int(base.ctx_len[i])].tolist() for i, _ in order[k:k + bs]]
        L = max(map(len, rows))
        ids = torch.tensor([[EOS] * (L - len(r)) + r for r in rows], device=dev)   # left padding
        mask = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in rows], device=dev)
        out = hf.generate(input_ids=ids, attention_mask=mask, num_beams=10, length_penalty=0.8,
                          no_repeat_ngram_size=4, max_new_tokens=64, eos_token_id=[EOS, 628],
                          pad_token_id=EOS, do_sample=False)
        for seq in out[:, L:].tolist():
            text = tok.decode(seq).split("<|endoftext|>")[0].split("\n\n")[0].strip()
            preds.append(text)
    if cfg.data.dataset == "e2e_synthetic":         # CI: random text, nothing to score
        return {"generated": len(preds)}
    if os.environ.get("E2E_PREDS"):                 # keep the generated texts for inspection
        with open(os.environ["E2E_PREDS"], "w") as f:
            f.write("\n".join(preds) + "\n")
    scorer = os.path.abspath(os.environ.get("E2E_METRICS", os.path.join(cfg.data.root, "e2e-metrics")))
    if not os.path.isfile(os.path.join(scorer, "measure_scores.py")):
        if smoke:
            return {"generated": len(preds)}
        raise FileNotFoundError(f"E2E scorer not found at {scorer}: run prepare.py (or set E2E_METRICS)")
    with tempfile.TemporaryDirectory() as d:
        ref_f, hyp_f = os.path.join(d, "ref.txt"), os.path.join(d, "hyp.txt")
        with open(ref_f, "w") as f:
            f.write("\n\n".join("\n".join(r) for _, r in order) + "\n")
        with open(hyp_f, "w") as f:
            f.write("\n".join(preds) + "\n")
        py = os.environ.get("E2E_PYTHON", "python")
        res = subprocess.run([py, os.path.join(scorer, "measure_scores.py"), "-p", ref_f, hyp_f],
                             capture_output=True, text=True, cwd=d,
                             env={**os.environ, "PYTHONPATH": scorer})
    scores = _scores(res.stdout)
    if not scores:
        raise RuntimeError(f"E2E scorer failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return {**scores, "generated": len(preds)}
