"""
Pack function for LLaMA / Mistral / Qwen2 / DeepSeek-dense family.

Applies to any model with the structure:
    model.model.embed_tokens   — token embedding
    model.model.layers         — list of transformer blocks
    model.model.norm           — final RMSNorm
    model.lm_head              — output projection

Confirmed working with:
    - LLaMA 2 / 3 (meta-llama/*)
    - Mistral 7B / Mixtral (mistralai/*)  [dense blocks only; for MoE use pack_deepseek_moe]
    - Qwen2 / Qwen2.5 (Qwen/*)
    - DeepSeek-dense (deepseek-ai/* without MoE)

Usage
-----
    from transformers import AutoModelForCausalLM
    import torchslicer as ts
    from examples.pack.llama import pack_llama

    model  = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    sliced = ts.slice(model, n=4, pack=pack_llama)
    sliced.train(loader, optimizer, criterion, epochs=3)

Notes
-----
- RoPE (rotary position embeddings) are computed per-block when a
  ``model.model.rotary_emb`` module is present (transformers ≥ 4.43).
  Pass it to each BlockStage so position embeddings are computed at
  forward time on whichever device runs the block.
- For LoRA fine-tuning call ``ts.peft_unwrap(model)`` before this function.
"""

import torch.nn as nn
import torchslicer as ts


def pack_llama(model: nn.Module) -> list[nn.Module]:
    """Return a flat stage list for LLaMA / Mistral / Qwen2 family models."""
    core       = model.model
    rotary_emb = getattr(core, 'rotary_emb', None)

    embed  = ts.SimpleEmbedStage(core.embed_tokens)
    blocks = [ts.BlockStage(layer, rotary_emb=rotary_emb) for layer in core.layers]
    head   = ts.CausalLMHeadStage(core.norm, model.lm_head)
    return [embed] + blocks + [head]
