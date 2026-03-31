"""
HuggingFace integration helpers.

``hf_pack(model_type)``  — look up the built-in pack function for an HF architecture.
``from_pretrained(...)`` — load + slice an HF model in one call (defined in ts.__init__).
"""

from .hf_packs import hf_pack, _PACK_REGISTRY  # noqa: F401 — re-exported

__all__ = ["hf_pack"]
