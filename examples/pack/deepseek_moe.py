"""
Pack function for DeepSeek-V2 / V3 MoE and Mixtral.

Applies to models where some (or all) transformer blocks return
``(hidden_states, aux_loss)`` — the router load-balancing loss.

Confirmed working with:
    - DeepSeek-V2 / V3 (deepseek-ai/*)
    - Mixtral-8x7B / 8x22B (mistralai/Mixtral-*)

Usage
-----
    from transformers import AutoModelForCausalLM
    import torchslicer as ts
    from examples.pack.deepseek_moe import pack_moe

    model  = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
    sliced = ts.slice(model, n=4, pack=pack_moe)
    sliced.train(loader, optimizer, criterion, epochs=3)

Notes
-----
- Each MoE block's aux loss is accumulated during forward and backpropagated
  independently on the worker that holds it.  This trains router weights without
  any gradient crossing partition boundaries.
- ``aux_loss_weight`` (default 0.01) matches the load-balancing coefficient
  used in the original DeepSeek and Mixtral training.  Override per-block if
  your model uses a different coefficient.
- Dense blocks (if any) are wrapped in ``BlockStage``; MoE blocks are wrapped
  in ``MoEBlockStage``.  Detection is done by inspecting child module names.
"""

import torch.nn as nn
import torchslicer as ts


def _is_moe_block(block: nn.Module) -> bool:
    """Heuristic: return True if this block contains MoE routing sub-modules."""
    for _, module in block.named_modules():
        name = type(module).__name__.lower()
        if any(kw in name for kw in ('moe', 'expert', 'router', 'sparsemoe')):
            return True
    child_names = {n for n, _ in block.named_children()}
    return bool(child_names & {'experts', 'router', 'gate'})


def pack_moe(model: nn.Module, aux_loss_weight: float = 0.01) -> list[nn.Module]:
    """Return a flat stage list for DeepSeek-V2/V3 or Mixtral MoE models.

    Parameters
    ----------
    model :
        A ``PreTrainedModel`` with ``model.model.embed_tokens``,
        ``model.model.layers``, ``model.model.norm``, and ``model.lm_head``.
    aux_loss_weight :
        Multiplier applied to each block's router aux loss before backprop.
    """
    core       = model.model
    rotary_emb = getattr(core, 'rotary_emb', None)

    embed  = ts.SimpleEmbedStage(core.embed_tokens)
    blocks = []
    for layer in core.layers:
        if _is_moe_block(layer):
            blocks.append(ts.MoEBlockStage(layer,
                                           aux_loss_weight=aux_loss_weight,
                                           rotary_emb=rotary_emb))
        else:
            blocks.append(ts.BlockStage(layer, rotary_emb=rotary_emb))
    head = ts.CausalLMHeadStage(core.norm, model.lm_head)
    return [embed] + blocks + [head]
