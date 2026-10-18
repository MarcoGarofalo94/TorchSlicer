"""Privacy (M12): attacks that observe a cut, training-time defences, leakage metrics.

``sliced.train(..., privacy=[ts.privacy.NormLabelAttack(cut=0), ts.privacy.NoPeek(alpha=0.5)])``
or the config's ``privacy`` section; the epoch history gets ``leak_<attack>_<metric>``.
Tutorial: docs/tutorials/privacy.md.
"""
from .base import Attack, Defence, Observation, register_attack, register_defence, resolve_privacy
from .attacks import (DirectionLabelAttack, DistanceCorrelation, InversionAttack, NormLabelAttack,
                      default_decoder)
from .defences import NoPeek
from .fsha import FSHA, default_discriminator, default_pilot
from .unsplit import UnSplit, total_variation
from .llm import EmbeddingRecovery, token_f1
from .dp import DPAccountant, gaussian_epsilon, gaussian_mu, gdp_delta, gdp_epsilon
from .metrics import auc, distance_correlation, psnr, ssim

__all__ = ["Attack", "Defence", "Observation", "register_attack", "register_defence", "resolve_privacy",
           "NormLabelAttack", "DirectionLabelAttack", "DistanceCorrelation", "InversionAttack",
           "default_decoder", "NoPeek", "FSHA", "default_pilot", "default_discriminator", "UnSplit",
           "total_variation", "EmbeddingRecovery", "token_f1", "DPAccountant", "gaussian_mu",
           "gaussian_epsilon", "gdp_delta", "gdp_epsilon", "auc", "distance_correlation", "psnr", "ssim"]
