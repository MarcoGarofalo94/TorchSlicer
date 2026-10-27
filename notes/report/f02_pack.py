"""F2 check: pack= for models torch.fx can't trace (HF LLMs, MoE), tied weights.

Small random Hugging Face models (built from a config, no download), CPU, dropout off.
1. Why pack: ts.slice(GPT-2) raises TraceError and points to pack=.
2. The pack: stages chained == model(ids).logits (GPT-2, Llama with RoPE, Mixtral MoE);
   ts.trace(model, pack=...).summary(...) with sizes, the tied embedding <-> LM head counted once (*).
3. Cut: strategies over stages, named cuts by stage index, the tied group across slice 0 and the
   last slice, refusals.
4. Exactness: Adam steps, sliced vs unsplit (losses, final weights, the weights still tied).
5. Same on a cluster: centralized run on loopback workers; the tied weight is a replica on two
   workers, and the model ends trained like unsplit (training.sync_back: all).
6. MoE: MoEBlockStage should backpropagate the router load-balancing loss. Is there one to collect?

Run: .venv/bin/python notes/report/f02_pack.py
"""
import copy
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

HERE = Path(globals().get("__file__", Path.cwd() / "_")).resolve().parent   # a notebook has no __file__
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "examples" / "pack").is_dir())
sys.path.insert(0, str(ROOT))                                   # the repo root, for examples.pack
import torchslicer as ts
from torchslicer.core.tied import find_tied_parameters
from examples.pack.gpt2 import pack_gpt2
from examples.pack.llama import pack_llama
from examples.pack.deepseek_moe import pack_moe

logging.getLogger("torchslicer").setLevel(logging.WARNING)
transformers.logging.set_verbosity_error()
torch.manual_seed(0)
VOCAB, T = 200, 16
GPT2 = transformers.GPT2LMHeadModel(transformers.GPT2Config(
    vocab_size=VOCAB, n_positions=32, n_embd=32, n_layer=4, n_head=2,
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))
LLAMA = transformers.LlamaForCausalLM(transformers.LlamaConfig(
    vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True))
MIXTRAL = transformers.MixtralForCausalLM(transformers.MixtralConfig(
    vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2))
IDS = torch.randint(0, VOCAB, (8, T + 1))
X, Y = IDS[:, :-1], IDS[:, 1:]                 # next-token prediction
STEPS, LR = 3, 1e-2
OPT = {"name": "Adam", "params": {"lr": LR}}   # Adam: one wrong update of a tied weight shows up
LOSS = {"name": "CrossEntropyLoss"}            # the head stage returns logits [B, V, T]


def fresh(base):
    return copy.deepcopy(base).train()


def same_weights(a, b, atol=1e-5):
    return all(torch.allclose(p, q, atol=atol) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def train_unsplit(base):
    m = fresh(base)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    losses = []
    for _ in range(STEPS):
        opt.zero_grad()
        loss = F.cross_entropy(m(X).logits.transpose(1, 2), Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return m, sum(losses) / STEPS


# ── 1. Why pack ─────────────────────────────────────────────────────────────
section("1. ts.slice(GPT2LMHeadModel, n=2): torch.fx can't trace it")
try:
    ts.slice(fresh(GPT2), n=2)
    raise AssertionError("GPT-2 traced: expected TraceError")
except ts.TraceError as e:
    lines = str(e).splitlines()
    print(lines[0][:150])
    print(*[l for l in lines if "pack" in l][:2], sep="\n")

# ── 2. The pack ─────────────────────────────────────────────────────────────
section("2. Stages chained == model(ids).logits; GPT-2 and Llama tie embedding <-> LM head")
for name, base, pack in [("gpt2", GPT2, pack_gpt2), ("llama (RoPE)", LLAMA, pack_llama),
                         ("mixtral (MoE)", MIXTRAL, pack_moe)]:
    m = fresh(base).eval()
    stages = pack(m)
    with torch.no_grad():
        diff = (nn.Sequential(*stages)(X) - m(X).logits.transpose(1, 2)).abs().max().item()
    tied = m.lm_head.weight is m.get_input_embeddings().weight
    print(f"{name:<14} {len(stages)} stages {[type(s).__name__ for s in stages][:3]}...  "
          f"max |diff| {diff:.1e}  tied: {tied}")
    assert diff < 1e-5

section("2b. ts.trace(gpt2, pack=pack_gpt2).summary(params=True, example_input=x)")
g = ts.trace(fresh(GPT2), pack=pack_gpt2)
print(g.summary(params=True, example_input=X))

# ── 3. Cut ──────────────────────────────────────────────────────────────────
section("3. Cut over stages: strategies, stage-index cuts, the tied group")
for what, s in {
    "uniform n=3": ts.slice(fresh(GPT2), n=3, pack=pack_gpt2),
    "params n=3": ts.slice(fresh(GPT2), strategy="params", n=3, pack=pack_gpt2),
    "cuts=[1, 3]": ts.slice(fresh(GPT2), cuts=[1, 3], pack=pack_gpt2),
    "ts.slice(g, n=3)": ts.slice(g, n=3),
}.items():
    parts = [p.layer_indices for p in s.partitions]
    groups = find_tied_parameters([sl.layers for sl in s.slices()])
    print(f"{what:<18} stages per slice {parts}  tied groups: "
          f"{[(gr.owner, gr.replicas) for gr in groups]}  (owner, replicas)")

s = ts.slice(g, cuts=[1, 3])
print("\n" + s.summary(params=True, example_input=X))

refusals = {
    "ts.slice(gpt2, n=2, pack=pack_gpt2, expand=['transformer'])":
        lambda: ts.slice(fresh(GPT2), n=2, pack=pack_gpt2, expand=["transformer"]),
    "ts.slice(gpt2, cuts=['transformer.h.1'], pack=pack_gpt2)  (names, not stage indices)":
        lambda: ts.slice(fresh(GPT2), cuts=["transformer.h.1"], pack=pack_gpt2),
    "ts.slice(gpt2, cuts=[5], pack=pack_gpt2)  (after the last stage)":
        lambda: ts.slice(fresh(GPT2), cuts=[5], pack=pack_gpt2),
}
for what, call in refusals.items():
    try:
        call()
        print(f"{what}\n    -> NOT refused")
    except (ValueError, KeyError, IndexError) as e:
        print(f"{what}\n    -> {type(e).__name__}: {str(e).splitlines()[0][:150]}")

# ── 4. Exactness ────────────────────────────────────────────────────────────
section(f"4. Train {STEPS} Adam steps: sliced vs unsplit (losses, final weights, still tied)")
print(f"{'model / cutting':<30} {'loss':>10} {'unsplit':>10} {'|diff|':>8}  {'weights':>8}  {'tied':>5}")
for mname, base, pack in [("gpt2", GPT2, pack_gpt2), ("llama", LLAMA, pack_llama)]:
    ref, ref_mean = train_unsplit(base)
    for cut_name, kw in {"uniform n=3": {"n": 3}, "cuts=[0] (head far away)": {"cuts": [0]},
                         "GPipe n=3, 4 micro": {"n": 3, "n_micro": 4}}.items():
        n_micro = kw.pop("n_micro", None)
        mm = fresh(base)
        s = ts.slice(mm, pack=pack, **kw)
        extra = {"n_micro_batches": n_micro} if n_micro else {}
        hist = s.train([(X, Y)] * STEPS, OPT, LOSS, epochs=1, **extra)
        ok, tied = same_weights(mm, ref), mm.lm_head.weight is mm.get_input_embeddings().weight
        print(f"{mname + ' ' + cut_name:<30} {hist[0]['loss']:>10.6f} {ref_mean:>10.6f} "
              f"{abs(hist[0]['loss'] - ref_mean):>8.1e}  {'equal' if ok else 'DIFFER':>8}  {str(tied):>5}")
        assert ok and tied and abs(hist[0]["loss"] - ref_mean) < 1e-5

# ── 5. Same on a cluster ────────────────────────────────────────────────────
section("5. Centralized GPT-2 on 3 loopback gRPC workers, tied weight on worker 0 and worker 2")
ref, ref_mean = train_unsplit(GPT2)
cfg = ts.RunConfig()
cfg.training.epochs = 1
cfg.training.optimizer, cfg.training.criterion = OPT, LOSS
cfg.discovery.n_workers = 3
cfg.deploy.topology = "centralized"
cfg.network.device = "cpu"
cfg.logging.enabled = False
cfg.training.sync_back = "all"
mm = fresh(GPT2)
hist = ts.run(lambda: mm, lambda: [(X, Y)] * STEPS, run_config=cfg, verbose=False, pack=pack_gpt2)
ok, tied = same_weights(mm, ref), mm.lm_head.weight is mm.transformer.wte.weight
print(f"loss {hist[0]['loss']:.6f} (unsplit {ref_mean:.6f})  model trained like unsplit: {ok}  still tied: {tied}")
assert ok and tied

# ── 6. MoE aux loss ─────────────────────────────────────────────────────────
section("6. Mixtral: the router load-balancing loss")
mm = fresh(MIXTRAL)
stages = pack_moe(mm)
h = stages[0](X)
for st in stages[1:-1]:
    h = st(h)
collected = [st.pop_aux_loss() for st in stages[1:-1]]
hf = copy.deepcopy(MIXTRAL).train()
hf.config.output_router_logits = True
out = hf(X)
print(f"HF, output_router_logits=True: aux_loss = {out.aux_loss.item():.4f} "
      f"(x router_aux_loss_coef {hf.config.router_aux_loss_coef} added to the LM loss)")
print(f"MoEBlockStage collected per block: {collected}")
print(f"block forward returns: {type(mm.model.layers[0](stages[0](X), position_embeddings=stages[1].rotary_emb(stages[0](X), torch.arange(T)[None]))).__name__}"
      f" (transformers {transformers.__version__})")
if all(a is None for a in collected):
    print("FINDING: no aux loss reaches MoEBlockStage: the router is trained by the LM loss only.")

ref, ref_mean = train_unsplit(MIXTRAL)       # HF without output_router_logits: no aux loss either
mm = fresh(MIXTRAL)
hist = ts.slice(mm, n=3, pack=pack_moe).train([(X, Y)] * STEPS, OPT, LOSS, epochs=1)
ok = same_weights(mm, ref)
print(f"sliced Mixtral vs unsplit HF without aux loss: loss {hist[0]['loss']:.6f} vs {ref_mean:.6f}, "
      f"weights {'equal' if ok else 'DIFFER'}")
assert ok
print("\nall checks passed")
