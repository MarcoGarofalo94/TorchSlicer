"""
HuggingFace model adapter for TorchSlicer.

Wraps a HuggingFace transformer into a flat nn.Sequential of opaque stages so
that ts.slice() sees no cross-partition skip connections.  Each stage has the
signature ``forward(tensor) -> tensor``, compatible with SplitLayer and all
executors (local, distributed, P2P).

Built-in architectures
----------------------
- GPT-2 family      (GPT2LMHeadModel)
- BERT family       (BertForSequenceClassification, BertForMaskedLM)
- LLaMA/Mistral/Qwen  (any model with model.embed_tokens + model.layers + model.norm)
- DeepSeek-V2/V3 MoE  (same structure; MoE blocks detected and wrapped in MoEBlockStage)
- Generic fallback  (any model with an Embedding child, a ModuleList of blocks, a Linear head)

Extension
---------
Register custom architectures without forking the library::

    import torchslicer as ts

    ts.register_hf_architecture(
        name="llava",
        detect=lambda model: hasattr(model, "vision_tower"),
        extract=lambda model, task: [
            MyLLaVAFusionStage(model),
            *[ts.BlockStage(b) for b in model.language_model.model.layers],
            ts.CausalLMHeadStage(model.language_model.model.norm,
                                  model.language_model.lm_head),
        ],
    )
    sliced = ts.slice(ts.wrap_hf(model), strategy="uniform", n=4)

Multi-modal inputs
------------------
Subclass ``AuxInputStage`` for stages that need extra tensors (pixel_values etc.)::

    class LLaVAFusionStage(ts.AuxInputStage):
        def forward(self, input_ids, *, pixel_values=None, **_):
            ...  # fuse image features into text embeddings
            return fused  # [B, T, H] — normal activation from here on

The coordinator DataLoader should yield:
    ({"input_ids": ..., "pixel_values": ...}, labels)
or:
    ((input_ids_tensor, {"pixel_values": ...}), labels)

The executor routes the non-input_ids entries as ``aux_inputs`` to worker 0 only.

MoE models (DeepSeek-V2/V3, Mixtral)
--------------------------------------
Use ``MoEBlockStage`` when ``block(hidden_states)`` returns ``(h, aux_loss)``.
The aux loss is accumulated during forward and backpropagated independently on
each worker after the activation gradient backward — this correctly trains router
weights without crossing partition boundaries::

    stages = [
        ts.SimpleEmbedStage(model.embed_tokens),
        *[ts.MoEBlockStage(b, aux_loss_weight=0.01) for b in model.layers],
        ts.CausalLMHeadStage(model.norm, model.lm_head),
    ]

Attention masks
---------------
Phase 1: fixed-length sequences, no padding mask.  Decoder models apply their
built-in causal mask internally; encoder models compute full attention (equivalent
to all-ones mask).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


# ── architecture registry ─────────────────────────────────────────────────────

_ARCH_REGISTRY: list[tuple[str, Callable, Callable]] = []


def register_hf_architecture(
    name: str,
    detect: Callable[[nn.Module], bool],
    extract: Callable[[nn.Module, str], list[nn.Module]],
) -> None:
    """Register a custom HF architecture extractor.

    Registered architectures are checked *before* the built-in detectors
    (GPT-2, BERT, LLaMA/Qwen, generic fallback).  Later registrations take
    priority over earlier ones.

    Parameters
    ----------
    name :
        Human-readable label used in error messages.
    detect :
        ``fn(model) -> bool`` — return True if this extractor owns the model.
    extract :
        ``fn(model, task) -> list[nn.Module]`` — return the flat stage list.

    Example
    -------
    ::

        import torchslicer as ts
        from transformers import LlavaForConditionalGeneration

        def _detect_llava(model):
            return hasattr(model, "vision_tower") and hasattr(model, "language_model")

        def _extract_llava(model, task):
            from myproject.stages import LLaVAFusionStage
            lm = model.language_model
            return [
                LLaVAFusionStage(model.vision_tower,
                                 model.multi_modal_projector,
                                 lm.model.embed_tokens),
                *[ts.BlockStage(b) for b in lm.model.layers],
                ts.CausalLMHeadStage(lm.model.norm, lm.lm_head),
            ]

        ts.register_hf_architecture("llava", _detect_llava, _extract_llava)
        sliced = ts.slice(ts.wrap_hf(model), strategy="uniform", n=4)
    """
    _ARCH_REGISTRY.append((name, detect, extract))


# ── AuxInputStage — base class for multimodal first stages ────────────────────

class AuxInputStage(nn.Module):
    """Base class for pipeline stages that need extra named tensors.

    Subclass this for any model where stage 0 requires more than the main
    activation — typically a vision encoder / projector that fuses image
    features into the text embedding sequence.

    Contract
    --------
    - ``accepts_aux_inputs = True`` signals ``SplitLayer`` and executors that
      this stage expects keyword arguments in addition to the main tensor.
    - ``forward(main, **aux)`` must accept any extra tensors by name and return
      a single float tensor ``[B, T, H]``.
    - All subsequent stages receive normal single-tensor activations.

    Example
    -------
    ::

        class LLaVAFusionStage(ts.AuxInputStage):
            def __init__(self, vision_tower, projector, embed_tokens, image_token_id):
                super().__init__()
                self.vision_tower   = vision_tower
                self.projector      = projector
                self.embed_tokens   = embed_tokens
                self.image_token_id = image_token_id

            def forward(self, input_ids, *, pixel_values=None, **_):
                text_embeds = self.embed_tokens(input_ids)
                if pixel_values is None:
                    return text_embeds                      # text-only fallback
                img_feats = self.projector(
                    self.vision_tower(pixel_values).last_hidden_state)
                mask = (input_ids == self.image_token_id)
                text_embeds[mask] = img_feats.reshape(-1, img_feats.size(-1))
                return text_embeds                          # [B, T, H]
    """

    accepts_aux_inputs: bool = True

    def forward(self, main: torch.Tensor, **aux: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ── internal stage modules ────────────────────────────────────────────────────

class _GPT2EmbedStage(nn.Module):
    """GPT-2 token + position embedding.  Input: int64 [B, T].  Output: [B, T, H]."""

    def __init__(self, wte: nn.Embedding, wpe: nn.Embedding, drop: nn.Module):
        super().__init__()
        self.wte  = wte
        self.wpe  = wpe
        self.drop = drop

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.size(1)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.drop(self.wte(input_ids) + self.wpe(pos))


class _SimpleEmbedStage(nn.Module):
    """Embedding stage that delegates entirely to a single nn.Module.

    Works for BertEmbeddings (handles positions internally) and for simple
    ``embed_tokens`` tables (LLaMA, T5, Qwen, etc.).

    Input: int64 [B, T].  Output: [B, T, H].
    """

    def __init__(self, embed: nn.Module):
        super().__init__()
        self.embed = embed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class _BlockStage(nn.Module):
    """Single opaque transformer block.

    Calls ``block(hidden_states)`` with no attention mask (Phase 1).
    HuggingFace blocks return ``(hidden_states, ...)`` tuples in older
    versions and bare tensors in newer ones — both are handled.

    For LLaMA/Mistral/Qwen models on transformers ≥ 4.43, the parent model
    precomputes RoPE embeddings externally.  Pass ``rotary_emb`` from
    ``model.model.rotary_emb`` to have this stage compute them per-block.

    Input/output: float [B, T, H].
    """

    def __init__(self, block: nn.Module, rotary_emb: nn.Module = None):
        super().__init__()
        self.block      = block
        self.rotary_emb = rotary_emb  # None → old-style (block computes its own RoPE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self.rotary_emb is not None:
            T            = hidden_states.size(1)
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0)
            kwargs["position_embeddings"] = self.rotary_emb(hidden_states, position_ids)
        out = self.block(hidden_states, **kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out


class _MoEBlockStage(nn.Module):
    """Transformer block that returns ``(hidden_states, aux_loss)`` — MoE pattern.

    Used for DeepSeek-V2/V3, Mixtral, and any model whose blocks return a
    router load-balancing loss alongside the hidden states.

    The aux loss is accumulated during the forward pass.  After the activation
    backward completes, the executor calls ``layer.pop_moe_aux_loss()`` on the
    ``SplitLayer`` which in turn calls ``pop_aux_loss()`` here.  The executor
    then calls ``.backward()`` on the result.  This trains the router weights
    on each worker independently without crossing partition boundaries.

    Parameters
    ----------
    block :
        The HuggingFace transformer block module.
    aux_loss_weight :
        Scalar multiplier applied to aux_loss before backprop (default 0.01).
        Matches the typical DeepSeek/Mixtral load-balancing coefficient.
    rotary_emb :
        Optional RoPE module from ``model.model.rotary_emb``.  Required for
        transformers ≥ 4.43 where position embeddings are computed externally.
    """

    def __init__(self, block: nn.Module, aux_loss_weight: float = 0.01,
                 rotary_emb: nn.Module = None):
        super().__init__()
        self.block           = block
        self.aux_loss_weight = aux_loss_weight
        self.rotary_emb      = rotary_emb
        self._aux_loss: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self.rotary_emb is not None:
            T            = hidden_states.size(1)
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0)
            kwargs["position_embeddings"] = self.rotary_emb(hidden_states, position_ids)
        out = self.block(hidden_states, **kwargs)
        if isinstance(out, (tuple, list)):
            h = out[0]
            # Detect scalar aux loss in position 1
            if (len(out) > 1
                    and isinstance(out[1], torch.Tensor)
                    and out[1].ndim == 0):
                raw = out[1] * self.aux_loss_weight
                self._aux_loss = (
                    self._aux_loss + raw if self._aux_loss is not None else raw
                )
            return h
        return out

    def pop_aux_loss(self) -> Optional[torch.Tensor]:
        """Return accumulated aux loss and clear the buffer."""
        loss, self._aux_loss = self._aux_loss, None
        return loss


class _CausalLMHeadStage(nn.Module):
    """Final norm + LM projection head.

    Input:  float [B, T, H]
    Output: logits [B, vocab_size, T]  (C-before-spatial so that
            ``nn.CrossEntropyLoss(logits, targets)`` works with targets [B, T]).

    ``norm`` may be None for models where the norm is already inside the last block.
    """

    def __init__(self, norm: Optional[nn.Module], lm_head: nn.Module):
        super().__init__()
        self.norm    = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        # lm_head: [B, T, H] → [B, T, V]  →  transpose to [B, V, T]
        return self.lm_head(hidden_states).transpose(1, 2)


class _BERTSeqClassHead(nn.Module):
    """BERT pooler + dropout + classifier for sequence classification."""

    def __init__(self, pooler: nn.Module, dropout: nn.Module, classifier: nn.Module):
        super().__init__()
        self.pooler     = pooler
        self.dropout    = dropout
        self.classifier = classifier

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(hidden_states)
        return self.classifier(self.dropout(pooled))


class _BERTMLMHead(nn.Module):
    """BERT masked-LM prediction head (cls.predictions)."""

    def __init__(self, predictions: nn.Module):
        super().__init__()
        self.predictions = predictions

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predictions(hidden_states)


# Public aliases (underscore-free names for user-facing stage construction)
BlockStage       = _BlockStage
MoEBlockStage    = _MoEBlockStage
CausalLMHeadStage = _CausalLMHeadStage
SimpleEmbedStage  = _SimpleEmbedStage


# ── architecture-specific stage extractors ───────────────────────────────────

def _stages_gpt2(model, task: str) -> list[nn.Module]:
    transformer = model.transformer
    embed  = _GPT2EmbedStage(transformer.wte, transformer.wpe, transformer.drop)
    blocks = [_BlockStage(b) for b in transformer.h]
    if task != "causal_lm":
        raise ValueError(
            f"GPT-2 adapter only supports task='causal_lm', got {task!r}")
    head = _CausalLMHeadStage(transformer.ln_f, model.lm_head)
    return [embed] + blocks + [head]


def _stages_bert(model, task: str) -> list[nn.Module]:
    bert   = model.bert
    embed  = _SimpleEmbedStage(bert.embeddings)
    blocks = [_BlockStage(b) for b in bert.encoder.layer]
    if task == "seq_classification":
        head = _BERTSeqClassHead(bert.pooler, model.dropout, model.classifier)
    elif task == "masked_lm":
        head = _BERTMLMHead(model.cls.predictions)
    else:
        raise ValueError(
            f"BERT adapter supports task='seq_classification' or 'masked_lm', "
            f"got {task!r}")
    return [embed] + blocks + [head]


def _stages_llama(model, task: str) -> list[nn.Module]:
    """LLaMA / Mistral / Qwen / DeepSeek-style: model.embed_tokens + model.layers + model.norm.

    Automatically detects MoE blocks (return scalar aux_loss) and wraps them
    in ``_MoEBlockStage`` so the router load-balancing loss is handled correctly.

    For transformers ≥ 4.43, RoPE position embeddings are computed externally
    (``model.model.rotary_emb``).  When present, the RoPE module is passed to
    each block stage so it can compute ``position_embeddings`` per forward call.
    """
    core      = model.model
    embed     = _SimpleEmbedStage(core.embed_tokens)
    rotary_emb = getattr(core, 'rotary_emb', None)

    blocks = []
    for b in core.layers:
        if _block_is_moe(b):
            blocks.append(_MoEBlockStage(b, rotary_emb=rotary_emb))
        else:
            blocks.append(_BlockStage(b, rotary_emb=rotary_emb))

    if task != "causal_lm":
        raise ValueError(
            f"LLaMA adapter only supports task='causal_lm', got {task!r}")
    head = _CausalLMHeadStage(core.norm, model.lm_head)
    return [embed] + blocks + [head]


def _block_is_moe(block: nn.Module) -> bool:
    """Heuristic: detect MoE blocks by inspecting child names or output structure.

    Checks for MoE-specific attribute names common across DeepSeek-V2,
    Mixtral, and similar architectures — avoids an expensive forward probe.
    """
    child_names = {name for name, _ in block.named_children()}
    moe_indicators = {"mlp", "experts", "gate", "router", "moe", "shared_experts"}
    # If the block has both "experts" or "router" AND "mlp", it's likely MoE
    has_experts = bool(child_names & {"experts", "router", "gate"})
    # Also check for mixtral/deepseek-specific sub-modules
    for name, module in block.named_modules():
        mname = type(module).__name__.lower()
        if any(ind in mname for ind in ("moe", "expert", "router", "sparsemoe")):
            return True
    return has_experts


def _stages_generic(model, task: str) -> list[nn.Module]:
    """Fallback: inspect named children to find embed → ModuleList → head pattern."""
    children = dict(model.named_children())

    # Find the ModuleList of transformer blocks
    block_list = None
    block_key  = None
    for k, v in children.items():
        if isinstance(v, nn.ModuleList) and len(v) > 0:
            block_list = v
            block_key  = k
            break

    if block_list is None:
        for k, v in children.items():
            sub = dict(v.named_children()) if hasattr(v, 'named_children') else {}
            for sk, sv in sub.items():
                if isinstance(sv, nn.ModuleList) and len(sv) > 0:
                    block_list = sv
                    block_key  = f"{k}.{sk}"
                    break
            if block_list is not None:
                break

    if block_list is None:
        raise ValueError(
            f"wrap_hf: could not auto-detect transformer block list in "
            f"{type(model).__name__}.  Supported built-ins: GPT-2, BERT, "
            f"LLaMA/Mistral/Qwen/DeepSeek.  "
            f"Register a custom extractor with ts.register_hf_architecture().")

    embed_module = None
    for k, v in children.items():
        if k == block_key:
            continue
        if isinstance(v, nn.Embedding):
            embed_module = v
            break

    if embed_module is None:
        raise ValueError(
            f"wrap_hf: could not find an nn.Embedding child in "
            f"{type(model).__name__}.  "
            f"Register a custom extractor with ts.register_hf_architecture().")

    head_module = None
    past_blocks = False
    for k, v in children.items():
        if k == block_key:
            past_blocks = True
            continue
        if past_blocks and isinstance(v, nn.Linear):
            head_module = v
            break

    if head_module is None:
        raise ValueError(
            f"wrap_hf: could not find a Linear head after block list in "
            f"{type(model).__name__}.  "
            f"Register a custom extractor with ts.register_hf_architecture().")

    embed  = _SimpleEmbedStage(embed_module)
    blocks = [_BlockStage(b) for b in block_list]
    head   = _CausalLMHeadStage(norm=None, lm_head=head_module)
    return [embed] + blocks + [head]


def _extract_stages(model, task: str) -> list[nn.Module]:
    # User-registered architectures are checked first (last registered wins)
    for name, detect, extract in reversed(_ARCH_REGISTRY):
        try:
            if detect(model):
                return extract(model, task)
        except Exception as e:
            raise ValueError(
                f"wrap_hf: registered extractor {name!r} raised an error: {e}") from e

    # Built-in detectors
    # GPT-2 family: model.transformer.h
    if hasattr(model, 'transformer') and hasattr(getattr(model, 'transformer', None), 'h'):
        return _stages_gpt2(model, task)

    # BERT family: model.bert.encoder.layer
    if hasattr(model, 'bert') and hasattr(getattr(model, 'bert', None), 'encoder'):
        return _stages_bert(model, task)

    # LLaMA / Mistral / Qwen / DeepSeek family: model.model.layers + model.model.embed_tokens
    inner = getattr(model, 'model', None)
    if inner is not None and hasattr(inner, 'layers') and hasattr(inner, 'embed_tokens'):
        return _stages_llama(model, task)

    return _stages_generic(model, task)


# ── public API ────────────────────────────────────────────────────────────────

class HFAdapter(nn.Sequential):
    """
    A flat ``nn.Sequential`` of opaque pipeline stages wrapping a HuggingFace
    transformer model.  Safe to pass directly to ``ts.slice()``.

    Each stage has signature ``forward(tensor) -> tensor``:

    * Stage 0 (embed): ``int64 [B, T]  →  float [B, T, H]``
    * Stages 1..N-1 (blocks): ``float [B, T, H]  →  float [B, T, H]``
    * Stage N (head): ``float [B, T, H]  →  logits``

    Because residual connections are encapsulated inside each block stage,
    ``ModelGraph.from_sequential()`` produces a purely linear DAG with no
    cross-partition edges — ``BaseSplitter.validate()`` always passes.

    Multi-modal stages (``AuxInputStage``) receive extra keyword tensors via
    the ``**aux`` argument to ``SplitLayer.forward()``; subsequent stages see
    only the fused float activation.
    """

    def __init__(self, model: nn.Module, task: str = "causal_lm"):
        stages = _extract_stages(model, task)
        super().__init__(*stages)
        self.task = task
        self._hf_model_class = type(model).__name__

    def __repr__(self) -> str:
        n_blocks = len(self) - 2
        return (
            f"HFAdapter({self._hf_model_class}, task={self.task!r}, "
            f"stages={len(self)} [{n_blocks} transformer blocks])"
        )


def wrap_hf(model: nn.Module, task: str = "causal_lm") -> HFAdapter:
    """
    Wrap a HuggingFace model so it is compatible with ``ts.slice()``.

    Parameters
    ----------
    model :
        A HuggingFace ``PreTrainedModel`` (or any ``nn.Module`` with the
        conventional embed → blocks → head structure).
    task :
        ``"causal_lm"``          — autoregressive LM (GPT-2, LLaMA, Qwen …)
        ``"seq_classification"`` — sequence classification (BERT …)
        ``"masked_lm"``          — masked LM (BERT …)

    Returns
    -------
    HFAdapter
        An ``nn.Sequential`` subclass ready for ``ts.slice()``.

    Notes
    -----
    - For custom architectures (LLaVA, InternVL, Florence-2 …) register an
      extractor first with ``ts.register_hf_architecture()``.
    - For LoRA models call ``ts.peft_unwrap(model)`` before ``wrap_hf``.
    - Qwen2/2.5 and DeepSeek-dense are detected automatically via the LLaMA path.
    - DeepSeek-V2/V3 MoE blocks are wrapped in ``MoEBlockStage`` automatically.
    """
    return HFAdapter(model, task=task)
