"""Split-learning settings (M4): roles/placement, cut hooks, SL metrics, multi-client sessions."""

from .hooks import (
    CutHook, CutContext, Float16, BFloat16, Quantize, TopK, CircularConv, GaussianNoise, ClipNoise, Recorder, Chain,
    register_hook, resolve_cut_hooks, hooks_from_config, pack_payload, unpack_payload, payload_nbytes,
)
from .roles import Placement, SETTINGS, resolve_setting
from .metrics import CommMeter
from .clients import ClientSession, SplitLayerClientState
