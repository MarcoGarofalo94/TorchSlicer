"""
Pipeline stage primitives for use in pack functions.

These are building blocks for ``pack(model) -> list[nn.Module]`` functions
passed to ``ts.slice()``.  They handle the protocol concerns that the executor
and ``SplitLayer`` need to know about (aux inputs, MoE aux loss).

Example usage in a pack function::

    import torchslicer as ts

    def pack_qwen(model):
        core = model.model
        rotary_emb = getattr(core, 'rotary_emb', None)
        return [
            ts.SimpleEmbedStage(core.embed_tokens),
            *[ts.BlockStage(layer, rotary_emb=rotary_emb) for layer in core.layers],
            ts.CausalLMHeadStage(core.norm, model.lm_head),
        ]

    sliced = ts.slice(model, n=4, pack=pack_qwen)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class GPT2EmbedStage(nn.Module):
    """Token + position embedding stage for GPT-2 family models.

    GPT-2 computes embeddings as ``wte(input_ids) + wpe(positions)``, which
    requires two separate ``nn.Embedding`` tables and cannot be expressed as a
    single ``SimpleEmbedStage``.

    Input:  ``int64 [B, T]``
    Output: ``float [B, T, H]``

    Parameters
    ----------
    wte : token embedding table (``transformer.wte``)
    wpe : position embedding table (``transformer.wpe``)
    drop : dropout applied after the sum (``transformer.drop``)
    """

    def __init__(self, wte: nn.Embedding, wpe: nn.Embedding, drop: nn.Module):
        super().__init__()
        self.wte  = wte
        self.wpe  = wpe
        self.drop = drop

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        T   = input_ids.size(1)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.drop(self.wte(input_ids) + self.wpe(pos))


class AuxInputStage(ABC, nn.Module):
    """Base class for pipeline stages that need extra named tensors.

    Subclass this for any model where stage 0 requires more than the main
    activation — typically a vision encoder / projector that fuses image
    features into the text embedding sequence.

    Contract
    --------
    - ``accepts_aux_inputs = True`` signals ``SplitLayer`` and executors that
      this stage expects keyword arguments in addition to the main tensor.
    - ``forward(main, **aux)`` must accept any extra tensors by name and return
      a single float tensor.
    - All subsequent stages receive normal single-tensor activations.

    Example
    -------
    ::

        class VisionFusionStage(ts.AuxInputStage):
            def __init__(self, vision_encoder, projector, embed_tokens, image_token_id):
                super().__init__()
                self.vision_encoder = vision_encoder
                self.projector      = projector
                self.embed_tokens   = embed_tokens
                self.image_token_id = image_token_id

            def forward(self, input_ids, *, pixel_values=None, **_):
                text_embeds = self.embed_tokens(input_ids)
                if pixel_values is None:
                    return text_embeds
                img_feats = self.projector(
                    self.vision_encoder(pixel_values).last_hidden_state)
                mask = (input_ids == self.image_token_id)
                text_embeds[mask] = img_feats.reshape(-1, img_feats.size(-1))
                return text_embeds
    """

    accepts_aux_inputs: bool = True

    @abstractmethod
    def forward(self, main: torch.Tensor, **aux: torch.Tensor) -> torch.Tensor: ...


class SimpleEmbedStage(nn.Module):
    """Wraps a single embedding module as a pipeline stage.

    Works for any module that accepts ``int64 [B, T]`` and returns
    ``float [B, T, H]`` — e.g. ``embed_tokens``, ``BertEmbeddings``.
    """

    def __init__(self, embed: nn.Module):
        super().__init__()
        self.embed = embed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class BlockStage(nn.Module):
    """Wraps a single transformer block (or any nn.Module) as an opaque stage.

    Calls ``block(hidden_states)`` and unwraps tuple/list outputs, returning
    only the first element.  This hides intra-block skip connections from the
    splitter so they never cross partition boundaries.

    Parameters
    ----------
    block :
        Any ``nn.Module`` with signature ``forward(tensor) -> tensor | tuple``.
    rotary_emb :
        Optional RoPE module (e.g. ``model.model.rotary_emb`` on LLaMA/Qwen).
        Required for transformers ≥ 4.43 where position embeddings are
        computed externally.  When provided, ``position_embeddings=(cos, sin)``
        is passed as a keyword argument to the block.
    """

    def __init__(self, block: nn.Module, rotary_emb: nn.Module = None):
        super().__init__()
        self.block      = block
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self.rotary_emb is not None:
            T            = hidden_states.size(1)
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0)
            kwargs["position_embeddings"] = self.rotary_emb(hidden_states, position_ids)
        out = self.block(hidden_states, **kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out


class MoEBlockStage(nn.Module):
    """Wraps a Mixture-of-Experts block that returns ``(hidden_states, aux_loss)``.

    The aux loss (router load-balancing loss) is accumulated during forward and
    backpropagated independently on each worker after the activation gradient
    backward — this trains router weights correctly without crossing partition
    boundaries.

    Used for DeepSeek-V2/V3, Mixtral, and any block whose forward returns a
    scalar aux loss alongside the hidden states.

    Parameters
    ----------
    block :
        The MoE transformer block.
    aux_loss_weight :
        Scalar multiplier on the aux loss before backprop (default ``0.01``).
    rotary_emb :
        Optional RoPE module (same as ``BlockStage``).

    Example
    -------
    ::

        def pack_deepseek(model):
            core = model.model
            rotary_emb = getattr(core, 'rotary_emb', None)
            return [
                ts.SimpleEmbedStage(core.embed_tokens),
                *[ts.MoEBlockStage(layer, rotary_emb=rotary_emb)
                  for layer in core.layers],
                ts.CausalLMHeadStage(core.norm, model.lm_head),
            ]
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


class CausalLMHeadStage(nn.Module):
    """Final norm + LM projection head for causal language models.

    Input:  ``float [B, T, H]``
    Output: logits ``[B, vocab_size, T]`` — C-before-spatial so that
            ``nn.CrossEntropyLoss(logits, targets)`` works with ``targets [B, T]``.

    Parameters
    ----------
    norm :
        Layer norm before the head.  Pass ``None`` if the norm is already
        applied inside the last block.
    lm_head :
        Linear projection ``H → vocab_size``.
    """

    def __init__(self, norm: Optional[nn.Module], lm_head: nn.Module):
        super().__init__()
        self.norm    = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states).transpose(1, 2)
