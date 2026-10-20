"""UnSplit: model inversion and stealing by an honest-but-curious server.

Erdogan, Kupcu, Cicek, "UnSplit: Data-Oblivious Model Inversion, Model Stealing, and Label
Inference Attacks Against Split Learning", WPES 2022, https://arxiv.org/abs/2108.09033
(§3.2, Algorithm 1). The server knows the client's architecture but not its weights, has no
data and cannot query the client. From the smashed data ``z = f1(theta1, x)`` it receives,
it searches inputs and parameters by coordinate descent:

    x~ = argmin_x~  MSE(f1~(theta1~, x~), z) + lambda * TV(x~)
    theta1~ = argmin_theta1~  MSE(f1~(theta1~, x~), z)

alternating the two (Adam, lr 1e-3, lambda 0.1 for shallow splits in the paper). The clone
``f1~`` is kept across batches, so it also steals the client model.
"""

import math

import torch

from .base import Attack, register_attack
from .metrics import psnr, ssim


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Mean isotropic total variation of images ``(N, C, H, W)``:
    ``sqrt(|x[i+1,j] - x[i,j]|^2 + |x[i,j+1] - x[i,j]|^2)`` averaged over pixels (the
    paper sums; a mean keeps ``lambda`` independent of the image size)."""
    dh = x[..., 1:, :-1] - x[..., :-1, :-1]
    dw = x[..., :-1, 1:] - x[..., :-1, :-1]
    return (dh.pow(2) + dw.pow(2) + 1e-8).sqrt().mean()


class UnSplit(Attack):
    """Invert the activations at ``cut`` knowing only the client slices' architecture.

    On every ``every``-th observed batch it inverts its first ``n_invert`` samples:
    ``rounds`` of (``input_steps`` Adam steps on ``x~``, then ``model_steps`` on the clone),
    learning rate ``lr``, TV weight ``tv`` (images only). ``x~`` starts at zero, with the
    shape of the inputs (part of the architecture). The clone (a re-initialized copy of
    the client slices, in training mode like the client) persists across batches.
    Metrics: ``mse``, ``psnr``, ``ssim`` of ``x~`` against the true inputs, and
    ``clone_mse``: how far the clone's output on the *true* inputs is from ``z`` (model
    stealing).
    """
    name = "unsplit"
    needs_client = True

    def __init__(self, cut: int = 0, n_invert: int = 8, rounds: int = 20, input_steps: int = 10,
                 model_steps: int = 10, lr: float = 1e-3, tv: float = 0.1, every: int = 10,
                 seed: int = 0, key: str | None = None):
        super().__init__(cut, key)
        self.n_invert, self.rounds = int(n_invert), int(rounds)
        self.input_steps, self.model_steps = int(input_steps), int(model_steps)
        self.lr, self.tv, self.every, self.seed = float(lr), float(tv), max(1, int(every)), seed
        self.clone = None
        self._seen = 0
        self.reset()

    def reset(self):
        self.mses, self.psnrs, self.ssims, self.clone_mses = [], [], [], []

    def observe(self, obs):
        self._seen += 1
        if (self._seen - 1) % self.every:
            return
        if self.clone is None:
            torch.manual_seed(self.seed)          # the executor runs attacks in a forked RNG
            self.clone = self.client(fresh=True).requires_grad_(True).train()
            self._opt_m = torch.optim.Adam(self.clone.parameters(), lr=self.lr)
        z = obs.tensor[:self.n_invert].float()
        x = obs.inputs[:self.n_invert]
        x_hat = torch.zeros(x.shape, device=z.device, requires_grad=True)
        opt_x = torch.optim.Adam([x_hat], lr=self.lr)
        image = x_hat.dim() == 4
        for _ in range(self.rounds):
            for _ in range(self.input_steps):
                loss = (self.clone(x_hat) - z).pow(2).mean()
                if image and self.tv:
                    loss = loss + self.tv * total_variation(x_hat)
                opt_x.zero_grad()
                loss.backward()
                opt_x.step()
            x_fixed = x_hat.detach()
            for _ in range(self.model_steps):
                loss = (self.clone(x_fixed) - z).pow(2).mean()
                self._opt_m.zero_grad()
                loss.backward()
                self._opt_m.step()
        with torch.no_grad():
            x_hat, x = x_hat.detach(), x.float()
            self.mses.append((x_hat - x).pow(2).mean().item())
            self.psnrs.append(psnr(x_hat, x))
            self.ssims.append(ssim(x_hat, x))
            self.clone_mses.append((self.clone(x) - z).pow(2).mean().item())

    def metrics(self):
        if not self.mses:
            return {}
        mean = lambda v: sum(v) / len(v)
        out = {"mse": round(mean(self.mses), 6), "psnr": round(mean(self.psnrs), 4),
               "clone_mse": round(mean(self.clone_mses), 6)}
        s = [v for v in self.ssims if not math.isnan(v)]
        if s:
            out["ssim"] = round(mean(s), 4)
        return out


register_attack("unsplit", UnSplit)
