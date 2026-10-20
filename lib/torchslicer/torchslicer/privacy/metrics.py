"""Leakage metrics: label-inference AUC, reconstruction PSNR/SSIM, distance correlation.

All take tensors and return Python floats, except :func:`distance_correlation`, which is
differentiable (the NoPeek loss uses it).
"""

import math

import torch
import torch.nn.functional as F


def auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    """Area under the ROC curve of ``scores`` for the boolean ``positive`` (Mann-Whitney U,
    ties count one half). ``nan`` when one class is absent."""
    s = scores.detach().reshape(-1).double().cpu()
    y = positive.detach().reshape(-1).bool().cpu()
    n_pos = int(y.sum())
    n_neg = y.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
    end = counts.cumsum(0).double()
    avg_rank = end - (counts.double() - 1) / 2          # 1-based mean rank of each tie group
    rank_sum = avg_rank[inv][y].sum().item()
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def distance_correlation(x: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Sample distance correlation between the rows of ``x`` and ``z`` (each flattened to
    ``(N, -1)``), in [0, 1]; differentiable. The biased (V-statistic) estimator of
    Székely et al., the one NoPeek minimizes; it depends on the batch size, so compare
    values at equal batch sizes."""
    x = x.reshape(x.shape[0], -1).float()
    z = z.reshape(z.shape[0], -1).float()

    def centred(t):
        d = torch.cdist(t, t)
        return d - d.mean(0, keepdim=True) - d.mean(1, keepdim=True) + d.mean()

    a, b = centred(x), centred(z)
    dcov2 = (a * b).mean()
    dvar = (a * a).mean() * (b * b).mean()
    return (dcov2.clamp(min=0) / dvar.clamp(min=eps).sqrt()).clamp(min=eps).sqrt()


def _range(target: torch.Tensor) -> float:
    r = (target.max() - target.min()).item()
    return r if r > 0 else 1.0


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    """Mean per-sample PSNR in dB. ``data_range`` defaults to the target batch's
    max - min (inputs are usually normalized, so [0, 1] would be wrong)."""
    peak = data_range or _range(target)
    mse = (pred.detach().float() - target.float()).pow(2).reshape(target.shape[0], -1).mean(1)
    return (10 * torch.log10(peak ** 2 / mse.clamp(min=1e-12))).mean().item()


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None,
         window: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM (Wang et al. 2004: Gaussian window, K1 = 0.01, K2 = 0.03) of ``(N, C, H, W)``
    images, per channel; the window shrinks to fit small images. ``nan`` for other shapes."""
    if target.dim() != 4:
        return float("nan")
    L = data_range or _range(target)
    x, y = pred.detach().float(), target.float()
    w = min(window, x.shape[-1], x.shape[-2])
    w -= (w + 1) % 2                                     # odd
    g = torch.exp(-((torch.arange(w, device=x.device) - w // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(x.shape[1], 1, w, w).contiguous()

    def blur(t):
        return F.conv2d(t, k, groups=x.shape[1])

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return s.mean().item()


def quantile(values: list, q: float) -> float:
    """``q`` quantile of the finite ``values`` (linear interpolation), ``nan`` if none."""
    v = sorted(x for x in values if not math.isnan(x))
    if not v:
        return float("nan")
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)
