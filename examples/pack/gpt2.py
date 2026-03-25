"""
Pack function for GPT-2 family models.

Applies to: GPT2LMHeadModel, DistilGPT2, GPT2-medium/large/xl,
            and any model with a ``model.transformer.h`` block list.

Usage
-----
    from transformers import GPT2LMHeadModel
    import torchslicer as ts
    from examples.pack.gpt2 import pack_gpt2

    model  = GPT2LMHeadModel.from_pretrained("gpt2")
    sliced = ts.slice(model, n=4, pack=pack_gpt2)
    sliced.train(loader, optimizer, criterion, epochs=3)

Notes
-----
- GPT-2 computes token + position embeddings in stage 0; they are fused here
  because GPT-2's wpe lookup is tightly coupled to the sequence length.
- The final layer norm (``transformer.ln_f``) travels with the lm_head in the
  last stage so that the last worker applies it without a cross-partition edge.
- For LoRA fine-tuning call ``ts.peft_unwrap(model)`` before this function and
  explicitly unfreeze ``lm_head`` if weight-tying was broken by PEFT:
      for p in model.lm_head.parameters(): p.requires_grad_(True)
"""

import torch.nn as nn
import torchslicer as ts


def pack_gpt2(model: nn.Module) -> list[nn.Module]:
    """Return a flat stage list for any GPT-2 family model."""
    transformer = model.transformer
    embed  = ts.GPT2EmbedStage(transformer.wte, transformer.wpe, transformer.drop)
    blocks = [ts.BlockStage(b) for b in transformer.h]
    head   = ts.CausalLMHeadStage(transformer.ln_f, model.lm_head)
    return [embed] + blocks + [head]
