"""Privacy (M12): attacks that observe a cut, training-time defences, leakage metrics.

``sliced.train(..., privacy=[ts.privacy.NormLabelAttack(cut=0), ts.privacy.NoPeek(alpha=0.5)])``
or the config's ``privacy`` section; the epoch history gets ``leak_<attack>_<metric>``.
Tutorial: docs/tutorials/privacy.md.
"""
from .base import Attack, Defence, Observation, register_attack, register_defence, resolve_privacy
from .attacks import (DirectionLabelAttack, DistanceCorrelation, InversionAttack, NormLabelAttack,
                      default_decoder)
from .defences import NoPeek
from .metrics import auc, distance_correlation, psnr, ssim

__all__ = ["Attack", "Defence", "Observation", "register_attack", "register_defence", "resolve_privacy",
           "NormLabelAttack", "DirectionLabelAttack", "DistanceCorrelation", "InversionAttack",
           "default_decoder", "NoPeek", "auc", "distance_correlation", "psnr", "ssim"]
