"""Built-in attacks.

Label inference from returned gradients (Li et al., "Label Leakage and Protection in
Two-party Split Learning", ICLR 2022, https://arxiv.org/abs/2102.08504): the party below
the cut scores each sample from its cut gradient; the leak AUC of that score against the
true labels is computed per batch, summarized per epoch as the mean and, as in the paper,
the 95 % quantile over batches.

- :class:`NormLabelAttack`: score = ``||g||``. Positives (the minority, less confident
  class) get larger gradients.
- :class:`DirectionLabelAttack`: score = ``cos(g, g+)`` with ``g+`` the gradient of one
  positive sample of the batch, picked at random (the paper's oracle evaluation): same-class
  gradients point the same way.

Input reconstruction from cut activations:

- :class:`InversionAttack`: a learning-based black-box attack in the spirit of the inverse
  network of He, Zhang, Lee ("Model Inversion Attacks Against Collaborative Inference",
  ACSAC 2019, https://doi.org/10.1145/3359789.3359824): the server queries the client
  slices with auxiliary inputs from the data distribution, trains a decoder from their
  activations back to the inputs, online, while training runs, and decodes each activation
  it receives. Scored by PSNR and SSIM against the true inputs.
- :class:`DistanceCorrelation`: a model-free leakage measure, the distance correlation
  between raw inputs and the cut activations (the quantity NoPeek minimizes).
"""

import math

import torch
import torch.nn as nn

from .base import Attack, Observation, register_attack
from .metrics import auc, distance_correlation, psnr, quantile, ssim


def _per_sample(obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
    g, y = obs.tensor, obs.labels
    if y is None or y.dim() != 1 or y.is_floating_point() or g.shape[0] != y.shape[0]:
        raise ValueError("label attacks need one class label per sample: labels of shape (N,) "
                         f"for a cut gradient of shape (N, ...); got labels "
                         f"{None if y is None else tuple(y.shape)}, gradient {tuple(g.shape)}")
    return g.reshape(g.shape[0], -1).float(), y


class _LabelAttack(Attack):
    direction = "backward"

    def __init__(self, cut: int = 0, positive: int = 1, key: str | None = None):
        super().__init__(cut, key)
        self.positive = int(positive)
        self.reset()

    def reset(self):
        self.aucs = []

    def metrics(self):
        if not self.aucs:
            return {}
        finite = [a for a in self.aucs if not math.isnan(a)]
        return {"auc": round(sum(finite) / len(finite), 6) if finite else float("nan"),
                "auc_q95": round(quantile(self.aucs, 0.95), 6),
                "batches": len(finite)}


class NormLabelAttack(_LabelAttack):
    """Label leak AUC of the gradient norm at ``cut`` (Li et al. 2022, norm attack).
    ``positive``: the class scored as positive (the minority one)."""
    name = "label_norm"

    def observe(self, obs):
        g, y = _per_sample(obs)
        self.aucs.append(auc(g.norm(dim=1), y == self.positive))


class DirectionLabelAttack(_LabelAttack):
    """Label leak AUC of ``cos(g, g+)`` at ``cut`` (Li et al. 2022, cosine attack), with
    ``g+`` a random positive's gradient in each batch (left out of the scoring)."""
    name = "label_direction"

    def __init__(self, cut: int = 0, positive: int = 1, seed: int = 0, key: str | None = None):
        super().__init__(cut, positive, key)
        self._gen = torch.Generator().manual_seed(seed)

    def observe(self, obs):
        g, y = _per_sample(obs)
        pos = (y == self.positive).nonzero().reshape(-1).cpu()
        if pos.numel() == 0:
            self.aucs.append(float("nan"))
            return
        ref = int(pos[torch.randint(pos.numel(), (1,), generator=self._gen)])
        keep = torch.arange(g.shape[0]) != ref
        cos = torch.nn.functional.cosine_similarity(g, g[ref:ref + 1], dim=1)
        self.aucs.append(auc(cos.cpu()[keep], (y == self.positive).cpu()[keep]))


class DistanceCorrelation(Attack):
    """Mean distance correlation between the raw inputs and what the server receives at
    ``cut`` (1 = a function of the inputs up to rotation and scale, 0 = independent).
    Batch-size dependent: compare runs with equal batch sizes."""
    name = "dcor"

    def __init__(self, cut: int = 0, key: str | None = None):
        super().__init__(cut, key)
        self.reset()

    def reset(self):
        self.values = []

    def observe(self, obs):
        with torch.no_grad():
            self.values.append(distance_correlation(obs.inputs, obs.tensor).item())

    def metrics(self):
        return {"value": round(sum(self.values) / len(self.values), 6)} if self.values else {}


def aux_inputs(aux, n: int, device, who: str = "this attack") -> torch.Tensor:
    """The first ``n`` inputs of ``aux`` (a tensor, or a loader of ``(x, y)`` / ``x``
    batches) as one tensor on ``device``."""
    if not isinstance(aux, torch.Tensor):
        if aux is None:
            raise ValueError(f"{who} needs auxiliary inputs: pass aux= or give the run an "
                             "evaluation set (eval_data / data.dataset)")
        xs, k = [], 0
        for batch in aux:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            xs.append(x)
            k += x.shape[0]
            if k >= n:
                break
        aux = torch.cat(xs)
    return aux[:n].to(device)


def default_decoder(z_shape, x_shape) -> nn.Module:
    """A small decoder from one activation (``z_shape``, no batch dim) to one input:
    convolutional when both are images (C, H, W), else a 2-layer MLP."""
    if len(z_shape) == 3 and len(x_shape) == 3:
        return nn.Sequential(
            nn.Conv2d(z_shape[0], 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(size=tuple(x_shape[1:]), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, x_shape[0], 3, padding=1))
    dz, dx = math.prod(z_shape), math.prod(x_shape)
    return nn.Sequential(nn.Flatten(), nn.Linear(dz, 512), nn.ReLU(), nn.Linear(512, dx),
                         nn.Unflatten(1, tuple(x_shape)))


class InversionAttack(Attack):
    """Reconstruct the client's inputs from the activations the server receives at ``cut``.

    The attacker holds ``aux``: inputs from the data distribution, disjoint from the
    clients' (a tensor, or a loader of ``(x, y)``; default: the run's evaluation set,
    ``eval_data`` / ``data.dataset``). For every ``every``-th batch it observes, it takes
    ``steps`` Adam steps (``lr``, minibatches of ``batch_size``) on a decoder fitted to
    ``(query(x_aux), x_aux)`` with the client's *current* weights, then decodes the
    observed activation. Metrics: mean PSNR / SSIM (``data_range`` = the true batch's
    max - min) over the epoch's decoded batches. ``decoder``: your own module from an
    activation batch to an input batch (default :func:`default_decoder`).
    """
    name = "inversion"
    needs_query = True

    def __init__(self, cut: int = 0, aux=None, n_aux: int = 512, steps: int = 2,
                 batch_size: int = 64, lr: float = 1e-3, every: int = 1, decoder: nn.Module | None = None,
                 seed: int = 0, key: str | None = None):
        super().__init__(cut, key)
        self.aux, self.n_aux, self.steps = aux, int(n_aux), int(steps)
        self.batch_size, self.lr, self.every = int(batch_size), float(lr), max(1, int(every))
        self.decoder, self.seed = decoder, seed
        self._opt = None
        self._seen = 0
        self.reset()

    def set_aux(self, aux) -> None:
        """Default auxiliary data (the executor passes its evaluation set)."""
        if self.aux is None:
            self.aux = aux

    def _aux_tensor(self, device) -> torch.Tensor:
        self.aux = aux_inputs(self.aux, self.n_aux, device, type(self).__name__)
        return self.aux

    def reset(self):
        self.psnrs, self.ssims = [], []

    def _fit(self, device):
        aux = self._aux_tensor(device)
        if self._opt is None:
            torch.manual_seed(self.seed)          # the executor runs attacks in a forked RNG
            self._gen = torch.Generator(device=device).manual_seed(self.seed)
            z = self.query(aux[:2])
            if self.decoder is None:
                self.decoder = default_decoder(tuple(z.shape[1:]), tuple(aux.shape[1:]))
            self.decoder = self.decoder.to(device)
            self._opt = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        self.decoder.train()
        for _ in range(self.steps):
            idx = torch.randint(aux.shape[0], (min(self.batch_size, aux.shape[0]),),
                                generator=self._gen, device=device)
            x = aux[idx]
            z = self.query(x)
            loss = (self.decoder(z.float()) - x.float()).pow(2).mean()
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

    def observe(self, obs):
        self._seen += 1
        if (self._seen - 1) % self.every:
            return
        z, x = obs.tensor, obs.inputs
        self._fit(z.device)
        self.decoder.eval()
        with torch.no_grad():
            x_hat = self.decoder(z.float())
        self.psnrs.append(psnr(x_hat, x))
        self.ssims.append(ssim(x_hat, x))

    def metrics(self):
        if not self.psnrs:
            return {}
        out = {"psnr": round(sum(self.psnrs) / len(self.psnrs), 4)}
        s = [v for v in self.ssims if not math.isnan(v)]
        if s:
            out["ssim"] = round(sum(s) / len(s), 4)
        return out


register_attack("label_norm", NormLabelAttack)
register_attack("label_direction", DirectionLabelAttack)
register_attack("dcor", DistanceCorrelation)
register_attack("inversion", InversionAttack)
