"""ts.Algorithm — pluggable SL algorithm interface (M8).

Design: the executor owns the training loop; an Algorithm overrides only the
hooks it needs. Defaults reproduce vanilla SplitNN bit-for-bit.

Analogous to Flower's Strategy: one class, same instance runs in every
executor (local, centralized, P2P), researchers override what changes.
"""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class BackwardPolicy(Enum):
    FULL = "full"
    LOCAL_LOSS = "local_loss"


@dataclass
class AlgorithmContext:
    round: int      # = epoch
    batch_id: int
    n_slices: int


class Algorithm:
    """
    SL algorithm interface.

    Override only the hooks you need. Defaults give vanilla SplitNN.
    The same instance is passed to LocalExecutor, DistributedExecutor
    and P2PExecutor; the executor calls these hooks.

    Lifecycle::

        algorithm.configure(n_slices, criterion, placement)  # once at setup
        for each epoch:
            for each batch:
                policy = algorithm.backward_policy(round)
                if policy == LOCAL_LOSS:
                    loss = algorithm.client_loss(cut_out, labels, ctx)
                    # executor runs backward on client slices only
                    algorithm.step_aux(ctx)
                else:
                    custom = algorithm.loss(out, labels, ctx)  # None → use criterion
                    # executor runs full backward chain
            # epoch end
            algorithm.on_round_end(client_states, metrics, round)
            algorithm.aggregate(client_states, server_states, round)
    """

    def configure(self, n_slices: int, criterion: nn.Module, placement) -> None:
        """Called once at setup. Store what you need from the executor."""
        self._n_slices = n_slices
        self._criterion = criterion
        self._placement = placement

    # ── per-batch ──────────────────────────────────────────────────────────────

    def backward_policy(self, round: int) -> BackwardPolicy:
        """Which backward pass to run. Default: full chain."""
        return BackwardPolicy.FULL

    def loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        ctx: AlgorithmContext,
    ) -> "torch.Tensor | None":
        """Loss at the last slice (FULL policy). Return None → use executor's criterion."""
        return None

    def client_loss(
        self,
        cut_output: torch.Tensor,
        labels: torch.Tensor,
        ctx: AlgorithmContext,
    ) -> torch.Tensor:
        """Loss at the client cut (LOCAL_LOSS policy). Must be overridden."""
        raise NotImplementedError(
            f"{type(self).__name__} uses LOCAL_LOSS but did not override client_loss()")

    def client_cut(self) -> "int | None":
        """LOCAL_LOSS: index of the slice whose output goes to client_loss().
        None = last slice in placement.client_slices (or 0 if no SL setting)."""
        return None

    def step_aux(self, ctx: AlgorithmContext) -> None:
        """Called after LOCAL_LOSS backward, before slice optimizer steps.
        Step auxiliary optimizers (e.g., aux-head) here."""

    # ── per-round ──────────────────────────────────────────────────────────────

    def select_clients(self, available: list, round: int) -> list:
        """Which clients participate this round. Default: all."""
        return available

    def aggregate(
        self,
        client_states: list,
        server_states: list,
        round: int,
    ) -> "tuple[dict | None, dict | None]":
        """Aggregate states after a round. Return (client_state, server_state).
        None means no change. client_state is broadcast to all clients."""
        return None, None

    def needs_per_client_server_state(self) -> bool:
        """SplitFed V1: executor allocates a separate server-slice state per client."""
        return False

    def on_round_end(
        self,
        client_states: list,
        metrics: dict,
        round: int,
    ) -> "list | None":
        """End of a round (epoch) with all client states. Return updated states
        or None to keep them. Default: no-op."""
        return None


def overridden(algorithm: Algorithm, name: str) -> bool:
    """Whether ``algorithm``'s class replaces ``Algorithm.<name>``."""
    return getattr(type(algorithm), name, None) is not getattr(Algorithm, name)


def check_supported(algorithm: Algorithm, executor: str, unsupported: dict) -> None:
    """Raise if ``algorithm`` relies on a hook ``executor`` doesn't call, instead of
    training something else silently. ``unsupported``: {hook name: what it does}."""
    used = [f"{name} ({what})" for name, what in unsupported.items()
            if (name == "needs_per_client_server_state" and algorithm.needs_per_client_server_state())
            or (name != "needs_per_client_server_state" and overridden(algorithm, name))]
    if used:
        raise ValueError(f"{type(algorithm).__name__} uses hooks the {executor} doesn't run yet: "
                         + "; ".join(used))

