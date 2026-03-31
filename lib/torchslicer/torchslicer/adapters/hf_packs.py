"""
Built-in pack functions for popular HuggingFace model families.

These are the same functions that previously lived in ``examples/pack/`` but
are now first-class library code so they work out of the box with
``ts.from_pretrained()`` and ``ts.hf_pack()``.

Auto-detection
--------------
``hf_pack(model_type)`` resolves a pack function from a model's
``config.model_type`` field (the string HuggingFace stores in config.json).

    pack_fn = ts.hf_pack("gpt2")          # GPT-2 family
    pack_fn = ts.hf_pack("bert")          # BERT / seq-cls
    pack_fn = ts.hf_pack("distilbert")    # DistilBERT seq-cls
    pack_fn = ts.hf_pack("llama")         # LLaMA / Mistral / Qwen2

Manual use
----------
Import the pack function directly for full control::

    from torchslicer.adapters.hf_packs import pack_gpt2, pack_llama
    sliced = ts.slice(model, n=4, pack=pack_gpt2)
"""

import torch
import torch.nn as nn

from ..core.stages import (
    BlockStage,
    CausalLMHeadStage,
    GPT2EmbedStage,
    MoEBlockStage,
    SimpleEmbedStage,
)


# ── GPT-2 ─────────────────────────────────────────────────────────────────────

def pack_gpt2(model: nn.Module) -> list[nn.Module]:
    """GPT-2 / DistilGPT-2 / GPT-2-medium/large/xl.

    Requires ``model.transformer.h`` block list.
    For LoRA, call ``ts.peft_unwrap(model)`` first.
    """
    transformer = model.transformer
    embed  = GPT2EmbedStage(transformer.wte, transformer.wpe, transformer.drop)
    blocks = [BlockStage(b) for b in transformer.h]
    head   = CausalLMHeadStage(transformer.ln_f, model.lm_head)
    return [embed] + blocks + [head]


# ── BERT ──────────────────────────────────────────────────────────────────────

def pack_bert_seq_cls(model: nn.Module) -> list[nn.Module]:
    """BertForSequenceClassification.

    Pooler + dropout + classifier are bundled into a single head stage.
    """
    bert   = model.bert
    embed  = SimpleEmbedStage(bert.embeddings)
    blocks = [BlockStage(b) for b in bert.encoder.layer]
    head   = BlockStage(nn.Sequential(bert.pooler, model.dropout, model.classifier))
    return [embed] + blocks + [head]


def pack_bert_mlm(model: nn.Module) -> list[nn.Module]:
    """BertForMaskedLM."""
    bert   = model.bert
    embed  = SimpleEmbedStage(bert.embeddings)
    blocks = [BlockStage(b) for b in bert.encoder.layer]
    head   = BlockStage(model.cls)
    return [embed] + blocks + [head]


# ── DistilBERT ────────────────────────────────────────────────────────────────

class _DistilBertClsHead(nn.Module):
    """CLS-token extraction + classification head for DistilBERT.

    Input:  ``float [B, T, H]`` (last hidden states)
    Output: logits ``[B, num_labels]``
    """

    def __init__(self, pre_classifier: nn.Module, dropout: nn.Module,
                 classifier: nn.Module):
        super().__init__()
        self.pre_classifier = pre_classifier
        self.dropout        = dropout
        self.classifier     = classifier

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls = hidden_states[:, 0]
        cls = torch.relu(self.pre_classifier(cls))
        cls = self.dropout(cls)
        return self.classifier(cls)


def pack_distilbert_seq_cls(model: nn.Module) -> list[nn.Module]:
    """DistilBertForSequenceClassification."""
    db     = model.distilbert
    embed  = SimpleEmbedStage(db.embeddings)
    blocks = [BlockStage(layer) for layer in db.transformer.layer]
    head   = BlockStage(
        _DistilBertClsHead(model.pre_classifier, model.dropout, model.classifier)
    )
    return [embed] + blocks + [head]


# ── LLaMA / Mistral / Qwen2 / DeepSeek-dense ─────────────────────────────────

def pack_llama(model: nn.Module) -> list[nn.Module]:
    """LLaMA 2/3, Mistral, Qwen2, DeepSeek-dense.

    Any model with ``model.model.embed_tokens``, ``model.model.layers``,
    ``model.model.norm``, and ``model.lm_head``.
    For LoRA, call ``ts.peft_unwrap(model)`` first.
    """
    core       = model.model
    rotary_emb = getattr(core, 'rotary_emb', None)
    embed  = SimpleEmbedStage(core.embed_tokens)
    blocks = [BlockStage(layer, rotary_emb=rotary_emb) for layer in core.layers]
    head   = CausalLMHeadStage(core.norm, model.lm_head)
    return [embed] + blocks + [head]


# ── DeepSeek-V2/V3 / Mixtral MoE ─────────────────────────────────────────────

def _is_moe_block(block: nn.Module) -> bool:
    for _, module in block.named_modules():
        name = type(module).__name__.lower()
        if any(kw in name for kw in ('moe', 'expert', 'router', 'sparsemoe')):
            return True
    child_names = {n for n, _ in block.named_children()}
    return bool(child_names & {'experts', 'router', 'gate'})


def pack_moe(model: nn.Module, aux_loss_weight: float = 0.01) -> list[nn.Module]:
    """DeepSeek-V2/V3 or Mixtral MoE models.

    Dense blocks are wrapped in ``BlockStage``; MoE blocks in ``MoEBlockStage``.
    """
    core       = model.model
    rotary_emb = getattr(core, 'rotary_emb', None)
    embed  = SimpleEmbedStage(core.embed_tokens)
    blocks = []
    for layer in core.layers:
        if _is_moe_block(layer):
            blocks.append(MoEBlockStage(layer, aux_loss_weight=aux_loss_weight,
                                        rotary_emb=rotary_emb))
        else:
            blocks.append(BlockStage(layer, rotary_emb=rotary_emb))
    head = CausalLMHeadStage(core.norm, model.lm_head)
    return [embed] + blocks + [head]


# ── Registry ──────────────────────────────────────────────────────────────────

# Maps HuggingFace config.model_type → default pack function.
# Keys are lowercase strings matching HF's model_type field.
_PACK_REGISTRY: dict[str, callable] = {
    # GPT-2 family
    "gpt2":         pack_gpt2,
    # BERT family — seq-cls by default; use pack_bert_mlm explicitly for MLM
    "bert":         pack_bert_seq_cls,
    "roberta":      pack_bert_seq_cls,
    # DistilBERT
    "distilbert":   pack_distilbert_seq_cls,
    # LLaMA / Mistral / Qwen2 / dense DeepSeek
    "llama":        pack_llama,
    "mistral":      pack_llama,
    "qwen2":        pack_llama,
    "deepseek":     pack_llama,
    # MoE variants — prefer pack_moe when these are detected as MoE architectures
    "mixtral":      pack_moe,
}


def hf_pack(model_type: str):
    """Return the pack function for a HuggingFace *model_type* string.

    The ``model_type`` value matches what HuggingFace stores in
    ``model.config.model_type`` (e.g. ``"gpt2"``, ``"bert"``, ``"llama"``).

    Raises ``KeyError`` if the architecture is not in the built-in registry.
    To add a custom architecture::

        from torchslicer.adapters.hf_packs import _PACK_REGISTRY
        _PACK_REGISTRY["my_arch"] = my_pack_fn

    Or pass ``pack=my_pack_fn`` directly to ``ts.slice()`` / ``ts.from_pretrained()``.
    """
    key = model_type.lower()
    if key not in _PACK_REGISTRY:
        available = list(_PACK_REGISTRY)
        raise KeyError(
            f"No built-in pack function for model_type='{model_type}'. "
            f"Available: {available}. "
            "Pass pack=your_pack_fn to ts.slice() or ts.from_pretrained()."
        )
    return _PACK_REGISTRY[key]
