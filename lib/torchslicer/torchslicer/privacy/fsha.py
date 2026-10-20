"""Feature-space hijacking attack (FSHA): a malicious server.

Pasquini, Ateniese, Bernaschi, "Unleashing the Tiger: Inference Attacks on Split Learning",
CCS 2021, https://arxiv.org/abs/2012.02670 (§3.2-3.3, Table A.1/B.2). The server holds
public inputs ``X_pub`` from the clients' domain. At every training iteration it

1. trains a pilot encoder ``f~`` and its inverse ``f~^-1`` as an auto-encoder on ``X_pub``
   (Eq. 1, MSE);
2. trains a discriminator ``D`` to tell ``f~(X_pub)`` from the client's smashed data
   ``f(X_priv)`` (Wasserstein loss with a gradient penalty, as in the paper's
   implementation);
3. instead of the task gradient, sends the client the gradient of the adversarial loss
   ``-D(f(X_priv))`` (Eq. 3), which steers ``f`` into ``f~``'s feature space.

It then reconstructs the private inputs as ``f~^-1(f(X_priv))``. The attack changes
training on purpose: the client's slices stop learning the task (the clients would see it
in the validation error).
"""

import math

import torch
import torch.nn as nn

from .attacks import aux_inputs, default_decoder
from .base import Attack, register_attack
from .metrics import psnr, ssim


def default_pilot(x_shape, z_shape, width: int = 64) -> nn.Module:
    """A shallow *linear* encoder from one input to one activation (the paper's pilot
    network is linear convolutions): strided convolutions down to the activation's
    spatial size when both are images, else one linear map."""
    if len(x_shape) == 3 and len(z_shape) == 3:
        layers, size, c = [], x_shape[1], x_shape[0]
        while math.ceil(size / 2) >= z_shape[1] and size > z_shape[1]:
            layers.append(nn.Conv2d(c, width, 3, stride=2, padding=1))
            c, size = width, math.ceil(size / 2)
        if not layers:
            layers.append(nn.Conv2d(c, width, 3, padding=1))
        layers += [nn.AdaptiveAvgPool2d(tuple(z_shape[1:])), nn.Conv2d(width, z_shape[0], 3, padding=1)]
        return nn.Sequential(*layers)
    return nn.Sequential(nn.Flatten(), nn.Linear(math.prod(x_shape), math.prod(z_shape)),
                         nn.Unflatten(1, tuple(z_shape)))


def default_discriminator(z_shape, width: int = 128) -> nn.Module:
    """A critic on one activation: convolutional for (C, H, W), else an MLP; one score."""
    if len(z_shape) == 3:
        return nn.Sequential(
            nn.Conv2d(z_shape[0], width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 1))
    return nn.Sequential(nn.Flatten(), nn.Linear(math.prod(z_shape), 2 * width), nn.ReLU(),
                         nn.Linear(2 * width, 2 * width), nn.ReLU(), nn.Linear(2 * width, 1))


class FSHA(Attack):
    """Hijack the client slices' feature space at ``cut`` and reconstruct their inputs.

    ``aux``: the server's public inputs ``X_pub`` (a tensor or a loader; default: the
    run's evaluation set), the first ``n_aux`` used. Every observed batch takes one step on
    the pilot/inverse (``lr_pilot``) and on the critic (``lr_disc``, gradient penalty
    ``gp_weight``) with a ``X_pub`` batch of the same size, then forges the gradient sent
    back. ``pilot``, ``decoder`` (= ``f~^-1``), ``discriminator``: your own modules
    (defaults :func:`default_pilot`, :func:`~torchslicer.privacy.attacks.default_decoder`,
    :func:`default_discriminator`); the default pilot ends in a ReLU when the first smashed
    batch is non-negative, which the server sees (a linear target space could not be
    matched by a client that ends in a ReLU). ``forge=False`` only observes (an honest server
    running the same networks, as a control). Metrics: ``mse``, ``psnr``, ``ssim`` of
    ``f~^-1(z)`` against the true inputs, averaged over the epoch.
    """
    name = "fsha"
    forges = True

    def __init__(self, cut: int = 0, aux=None, n_aux: int = 1024, lr_pilot: float = 1e-4,
                 lr_disc: float = 1e-4, gp_weight: float = 500.0, width: int = 64,
                 pilot: nn.Module | None = None, decoder: nn.Module | None = None,
                 discriminator: nn.Module | None = None, forge: bool = True, seed: int = 0,
                 key: str | None = None):
        super().__init__(cut, key)
        self.aux, self.n_aux = aux, int(n_aux)
        self.lr_pilot, self.lr_disc, self.gp_weight = float(lr_pilot), float(lr_disc), float(gp_weight)
        self.width, self.seed, self.forge_grad = int(width), seed, bool(forge)
        self.pilot, self.decoder, self.discriminator = pilot, decoder, discriminator
        self._opt_ae = None
        self._pending = {}
        self.reset()

    def set_aux(self, aux) -> None:
        if self.aux is None:
            self.aux = aux

    def reset(self):
        self.mses, self.psnrs, self.ssims = [], [], []
        self._pending.clear()

    def _build(self, z, device):
        self.aux = aux_inputs(self.aux, self.n_aux, device, "FSHA")
        torch.manual_seed(self.seed)              # the executor runs attacks in a forked RNG
        self._gen = torch.Generator(device=device).manual_seed(self.seed)
        x_shape, z_shape = tuple(self.aux.shape[1:]), tuple(z.shape[1:])
        if self.pilot is None:
            self.pilot = default_pilot(x_shape, z_shape, self.width)
            if bool((z >= 0).all()):                  # the smashed data are non-negative (a ReLU
                self.pilot.append(nn.ReLU())          # at the cut): so must the target space be
        self.pilot = self.pilot.to(device)
        self.decoder = (self.decoder or default_decoder(z_shape, x_shape)).to(device)
        self.discriminator = (self.discriminator or default_discriminator(z_shape, 2 * self.width)).to(device)
        self._opt_ae = torch.optim.Adam([*self.pilot.parameters(), *self.decoder.parameters()], lr=self.lr_pilot)
        self._opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)

    def _penalty(self, real, fake):
        eps = torch.rand((real.shape[0],) + (1,) * (real.dim() - 1), generator=self._gen, device=real.device)
        mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
        g, = torch.autograd.grad(self.discriminator(mix).sum(), mix, create_graph=True)
        return (g.reshape(g.shape[0], -1).norm(dim=1) - 1).pow(2).mean()

    def observe(self, obs):
        z = obs.tensor.float()
        if self._opt_ae is None:
            self._build(z, z.device)
        idx = torch.randint(self.aux.shape[0], (z.shape[0],), generator=self._gen, device=z.device)
        xp = self.aux[idx].float()
        # 1. pilot + inverse: an auto-encoder on the public data (Eq. 1)
        ae = (self.decoder(self.pilot(xp)) - xp).pow(2).mean()
        self._opt_ae.zero_grad()
        ae.backward()
        self._opt_ae.step()
        # 2. critic: high on the pilot's feature space, low on the client's (WGAN-GP)
        real = self.pilot(xp).detach()
        d = self.discriminator
        loss_d = d(z).mean() - d(real).mean() + self.gp_weight * self._penalty(real, z)
        self._opt_d.zero_grad()
        loss_d.backward()
        self._opt_d.step()
        # 3. the gradient sent instead of the task's: make the client look like the pilot (Eq. 3)
        if self.forge_grad:
            zr = z.detach().requires_grad_(True)
            g, = torch.autograd.grad(-d(zr).mean(), zr)
            self._pending[obs.ctx.batch_id] = g.to(obs.tensor.dtype)
        with torch.no_grad():
            x_hat = self.decoder(z)
            x = obs.inputs.float()
            self.mses.append((x_hat - x).pow(2).mean().item())
            self.psnrs.append(psnr(x_hat, x))
            self.ssims.append(ssim(x_hat, x))

    def forge(self, obs):
        return self._pending.pop(obs.ctx.batch_id, None)

    def metrics(self):
        if not self.mses:
            return {}
        out = {"mse": round(sum(self.mses) / len(self.mses), 6),
               "psnr": round(sum(self.psnrs) / len(self.psnrs), 4)}
        s = [v for v in self.ssims if not math.isnan(v)]
        if s:
            out["ssim"] = round(sum(s) / len(s), 4)
        return out


register_attack("fsha", FSHA)
