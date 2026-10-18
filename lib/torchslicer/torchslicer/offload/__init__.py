"""Out-of-core training on one node: slices paged between a slow tier and the device (M5)."""
from .pager import SlicePager
from .executor import OffloadExecutor, measure_bandwidth, choose_optimizer_placement

__all__ = ["SlicePager", "OffloadExecutor", "measure_bandwidth", "choose_optimizer_placement"]
