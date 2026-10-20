"""Built-in training-time defences."""

import torch

from .base import Defence, Observation, register_defence
from .metrics import distance_correlation


class NoPeek(Defence):
    """NoPeek (Vepakomma et al., https://arxiv.org/abs/2008.09161): the client's loss gains
    ``alpha * dcor(x, z)``, the distance correlation between its raw inputs ``x`` and the
    activation ``z`` it sends at ``cut``. The client adds that term's gradient w.r.t.
    ``z`` to the gradient the server returns, so the client slices train on
    ``task loss + alpha * dcor`` (exactly: the server's gradient does not depend on it).
    The term is computed per (micro-)batch; under GPipe it is scaled by ``1 / M`` like the
    task loss. The reported ``loss`` stays the task loss."""
    name = "nopeek"

    def __init__(self, cut: int = 0, alpha: float = 0.1):
        super().__init__(cut)
        self.alpha = float(alpha)

    def grad_term(self, z: torch.Tensor, obs: Observation):
        if self.alpha == 0:
            return None
        with torch.enable_grad():
            zd = z.detach().float().requires_grad_(True)
            d = distance_correlation(obs.inputs.detach(), zd)
            (g,) = torch.autograd.grad(self.alpha * d, zd)
        return g


register_defence("nopeek", NoPeek)
