"""E2E NLG for the SplitLoRA baseline: data/e2e/{train,valid,test}.npz (GPT-2 tokens) and the
official E2E scorer in data/e2e-metrics.

The text files are the ones the LoRA paper's GPT-2 E2E recipe uses (microsoft/LoRA,
examples/NLG/data/e2e, one "meaning representation||reference" per line), pinned to a commit.
Each example is encoded as in that recipe (src/gpt2_encode.py, --add_bos --add_eos):
context tokens + <|endoftext|>, then " " + reference + <|endoftext|>. The loss is on the
reference tokens only (labels = -100 elsewhere). The test file keeps every reference: the
scorer takes all references of a meaning representation at once.

    .venv/bin/python baselines/splitlora_e2e/prepare.py      # needs network once; git for the scorer
"""
import os
import subprocess
import sys
import urllib.request

import numpy as np

LORA = "https://raw.githubusercontent.com/microsoft/LoRA/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLG/data/e2e/"
SCORER = ("https://github.com/tuetschek/e2e-metrics", "1b9c4d8")
EOS = 50256


def encode(lines, tok):
    ctx, out = [], []
    for line in lines:
        c, r = line.rstrip("\n").split("||")
        ctx.append(tok(c)["input_ids"] + [EOS])
        out.append(tok(" " + r.strip("\n"))["input_ids"] + [EOS])
    return ctx, out


def main(root="data"):
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    d = os.path.join(root, "e2e")
    os.makedirs(d, exist_ok=True)
    for split in ("train", "valid", "test"):
        raw = os.path.join(d, f"{split}.txt")
        if not os.path.exists(raw):
            urllib.request.urlretrieve(LORA + f"{split}.txt", raw)
        lines = open(raw, encoding="utf8").read().splitlines()
        ctx, ref = encode(lines, tok)
        L = max(len(c) + len(r) for c, r in zip(ctx, ref))
        x = np.full((len(ctx), L), EOS, dtype=np.int64)
        y = np.full((len(ctx), L), -100, dtype=np.int64)
        for i, (c, r) in enumerate(zip(ctx, ref)):
            seq = c + r
            x[i, :len(seq)] = seq
            y[i, len(c) - 1:len(seq) - 1] = r          # next-token targets on the reference
        texts = [ln.split("||") for ln in lines]
        np.savez(os.path.join(d, f"{split}.npz"), x=x, y=y, ctx_len=np.array([len(c) for c in ctx]),
                 context=np.array([t[0] for t in texts]), reference=np.array([t[1].strip() for t in texts]))
        print(split, len(lines), "examples, max length", L)
    scorer = os.path.join(root, "e2e-metrics")
    if not os.path.isdir(scorer):
        subprocess.run(["git", "clone", "-q", SCORER[0], scorer], check=True)
        subprocess.run(["git", "-C", scorer, "checkout", "-q", SCORER[1]], check=True)
    print("scorer:", scorer, "(needs Java 8+ on PATH for METEOR, and matplotlib, scikit-image, future)")


if __name__ == "__main__":
    main(*sys.argv[1:])
