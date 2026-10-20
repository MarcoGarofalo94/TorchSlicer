"""Differential-privacy accounting for :class:`~torchslicer.sl.hooks.ClipNoise` on a cut.

``ClipNoise(clip=C, noise_multiplier=z)`` clips each sample's row of the cut tensor to L2
norm ``C`` and adds N(0, (zC)^2) noise: per sample, the Gaussian mechanism with L2
sensitivity ``2C`` (replacing the sample moves its clipped row by at most ``2C``), i.e.
``mu = 2/z`` Gaussian DP (Dong, Roth, Su, "Gaussian Differential Privacy",
https://arxiv.org/abs/1905.02383, Thm 2.7). ``k`` releases of a sample's row compose to
``mu * sqrt(k)``-GDP (Cor. 3.3), and ``mu``-GDP is ``(eps, delta(eps))``-DP for
``delta(eps) = Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2)`` (Cor. 2.13; the same curve
as the analytic Gaussian mechanism of Balle & Wang, ICML 2018,
https://arxiv.org/abs/1805.06530, Thm 8). This is tight for the Gaussian mechanism.

What it covers: what one sample's *own* row reveals to the receiver (its label to the
client through the returned gradient, its input to the server through the activation),
composed over epochs, assuming a sample appears at most once per epoch. What it does not:
the sample's influence on *other* rows through weights trained on it, or through
cross-sample layers (BatchNorm in training mode) on the sender's side of the cut. It is not
an end-to-end DP guarantee for the trained model.
"""

import math

from ..sl.hooks import Chain, ClipNoise
from .base import Attack, register_attack


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def gaussian_mu(noise_multiplier: float, releases: int = 1, sensitivity: float = 2.0) -> float:
    """GDP ``mu`` of ``releases`` Gaussian releases with noise std ``noise_multiplier * C``
    and L2 sensitivity ``sensitivity * C`` (2 = replace one sample)."""
    if releases <= 0:
        return 0.0
    if noise_multiplier <= 0:
        return math.inf
    return sensitivity / noise_multiplier * math.sqrt(releases)


def gdp_delta(mu: float, eps: float) -> float:
    """``delta(eps)`` of ``mu``-GDP."""
    if mu == 0:
        return 0.0
    tail = _phi(-eps / mu - mu / 2)
    return _phi(-eps / mu + mu / 2) - (0.0 if tail == 0 else math.exp(eps + math.log(tail)))


def gdp_epsilon(mu: float, delta: float) -> float:
    """The smallest ``eps`` with ``mu``-GDP => ``(eps, delta)``-DP (bisection)."""
    if mu == 0:
        return 0.0
    if math.isinf(mu):
        return math.inf
    if gdp_delta(mu, 0.0) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while gdp_delta(mu, hi) > delta:
        lo, hi = hi, 2 * hi
        if hi > 1e6:
            return math.inf
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gdp_delta(mu, mid) > delta else (lo, mid)
    return hi


def gaussian_epsilon(noise_multiplier: float, releases: int, delta: float,
                     sensitivity: float = 2.0) -> float:
    """``eps`` at ``delta`` for ``releases`` clipped Gaussian releases of one sample."""
    return gdp_epsilon(gaussian_mu(noise_multiplier, releases, sensitivity), delta)


class DPAccountant(Attack):
    """Per-sample ``(eps, delta)`` of the ``ClipNoise`` hook on ``cut``, after each epoch.

    ``direction``: ``"backward"`` accounts the returned gradients (label privacy against
    the client side), ``"forward"`` the activations (input privacy against the server);
    the hook must noise that direction and be the first of a ``Chain`` (hooks after it are
    post-processing). Metrics: ``epsilon`` at ``delta``, ``mu`` (GDP) and ``releases`` (=
    epochs with a release so far). See the module docstring for what this covers.
    """
    name = "dp"

    def __init__(self, cut: int = 0, delta: float = 1e-5, direction: str = "backward",
                 key: str | None = None):
        super().__init__(cut, key)
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")
        self.direction, self.delta = direction, float(delta)
        self.noise_multiplier = None
        self._epochs, self._seen = 0, False

    def bind_hook(self, hook) -> None:
        first = hook.hooks[0] if isinstance(hook, Chain) and hook.hooks else hook
        covers = isinstance(first, ClipNoise) and (first.grad if self.direction == "backward" else first.forward)
        if not covers:
            raise ValueError(f"DPAccountant on cut {self.cut} ({self.direction}) needs a ClipNoise hook on that "
                             f"cut noising the {'gradient (grad=True)' if self.direction == 'backward' else 'activation'}"
                             f", first in its chain; found {type(hook).__name__ if hook else None}")
        self.noise_multiplier = first.noise_multiplier

    def reset(self):
        if self._seen:
            self._epochs += 1
        self._seen = False

    def observe(self, obs):
        self._seen = True

    def metrics(self):
        k = self._epochs + int(self._seen)
        mu = gaussian_mu(self.noise_multiplier, k)
        return {"epsilon": round(gdp_epsilon(mu, self.delta), 6), "mu": round(mu, 6), "releases": k}


register_attack("dp", DPAccountant)
