"""Cost-balancing splitters: pick cuts among ``valid_cuts()`` to minimise the largest
partition cost, for a per-node cost (parameters, FLOPs, measured time)."""
import copy
import statistics
import time
from abc import abstractmethod

import torch

from .base import BaseSplitter


def balanced_cuts(costs: list, valid_cuts: list, n_partitions: int) -> list:
    """Choose ``n_partitions - 1`` cuts from ``valid_cuts`` (cut after node k) so that the
    largest partition cost is minimal; ties go to the most even split (sum of squares)."""
    blocks, start = [], 0                 # atomic runs of nodes between valid cut points
    for k in [*valid_cuts, len(costs) - 1]:
        blocks.append((k, sum(costs[start:k + 1])))
        start = k + 1
    m = len(blocks)
    if n_partitions > m:
        raise ValueError(
            f"can't make {n_partitions} partitions: the graph has only {m - 1} valid cut points. "
            f"Use fewer partitions, or expand= to trace into submodules for finer cuts."
        )
    if n_partitions == 1:
        return []
    prefix = [0.0]
    for _, c in blocks:
        prefix.append(prefix[-1] + c)
    seg = lambda i, j: prefix[j] - prefix[i]          # cost of blocks i..j-1
    INF = (float("inf"), float("inf"))
    # best[p][j]: (max, sum of squares) for the first j blocks in p partitions
    best = [[INF] * (m + 1) for _ in range(n_partitions + 1)]
    back = [[0] * (m + 1) for _ in range(n_partitions + 1)]
    best[0][0] = (0.0, 0.0)
    for p in range(1, n_partitions + 1):
        for j in range(p, m - (n_partitions - p) + 1):
            for i in range(p - 1, j):
                prev = best[p - 1][i]
                if prev is INF:
                    continue
                c = seg(i, j)
                cand = (max(prev[0], c), prev[1] + c * c)
                if cand < best[p][j]:
                    best[p][j], back[p][j] = cand, i
    cuts, j = [], m
    for p in range(n_partitions, 0, -1):
        i = back[p][j]
        if p > 1:
            cuts.append(blocks[i - 1][0])
        j = i
    return sorted(cuts)


class CostSplitter(BaseSplitter):
    """Base for splitters that balance a per-node cost. Subclasses implement ``node_costs``."""

    @abstractmethod
    def node_costs(self, model_graph) -> list:
        ...

    def split(self, model_graph, n_partitions: int) -> list:
        costs = [float(c) for c in self.node_costs(model_graph)]
        self.last_costs = costs
        cuts = balanced_cuts(costs, model_graph.valid_cuts(), n_partitions)
        return self.partitions_from_cuts(model_graph, cuts)


class ParamSplitter(CostSplitter):
    """Balance the number of parameters (≈ weight + optimizer memory) per partition.

    A parameter shared by several nodes (tied weights) is counted at its first use.
    ``trainable_only=True`` counts only parameters with ``requires_grad`` (e.g. LoRA).
    """

    def __init__(self, trainable_only: bool = False):
        self.trainable_only = trainable_only

    def node_costs(self, model_graph) -> list:
        seen, costs = set(), []
        for node in model_graph.nodes:
            total = 0
            for p in node.module.parameters():
                if id(p) in seen or (self.trainable_only and not p.requires_grad):
                    continue
                seen.add(id(p))
                total += p.numel()
            costs.append(total)
        return costs


class _InputSplitter(CostSplitter):
    """A cost splitter that runs the model on an example input (a copy of it, so the
    real model's weights, buffers and RNG state are left untouched)."""

    def __init__(self, example_input=None, aux: dict | None = None):
        self.example_input = example_input
        self.aux = aux

    def _copy_graph(self, model_graph, device=None):
        if self.example_input is None:
            raise ValueError(
                f"{type(self).__name__} needs an example input batch: "
                f"ts.slice(model, strategy='{getattr(self, 'name', '')}', n=..., example_input=x)"
            )
        g = copy.copy(model_graph)
        g.nodes = [copy.copy(n) for n in model_graph.nodes]
        modules = copy.deepcopy([n.module for n in model_graph.nodes])  # one memo: sharing kept
        for n, m in zip(g.nodes, modules):
            n.module = m.to(device) if device is not None else m
        return g


class FlopsSplitter(_InputSplitter):
    """Balance forward FLOPs per partition, counted with ``torch.utils.flop_counter``
    on ``example_input`` (backward FLOPs are roughly proportional)."""

    name = "flops"

    def node_costs(self, model_graph) -> list:
        from torch.utils.flop_counter import FlopCounterMode
        g = self._copy_graph(model_graph)
        x = self.example_input
        costs, outputs = [], []
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for node in g.nodes:
                args = [outputs[p] for p in node.predecessors] if node.predecessors else [x]
                kwargs = self.aux if (not node.predecessors and self.aux
                                      and getattr(node.module, "accepts_aux_inputs", False)) else {}
                counter = FlopCounterMode(display=False)
                with counter:
                    outputs.append(node.module(*args, **kwargs))
                costs.append(counter.get_total_flops())
        return costs


class ProfiledSplitter(_InputSplitter):
    """Balance measured forward + backward time per partition.

    Each node runs in isolation on ``device`` (default: CUDA if available) with its
    real input shapes; the median of ``repeats`` timed runs after ``warmup`` is used.
    Measure on the hardware the slices will run on; with heterogeneous workers,
    subclass and scale the costs per worker.
    """

    name = "profiled"

    def __init__(self, example_input=None, aux: dict | None = None, device=None,
                 warmup: int = 2, repeats: int = 5):
        super().__init__(example_input, aux)
        self.device = device
        self.warmup = warmup
        self.repeats = repeats

    def node_costs(self, model_graph) -> list:
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        g = self._copy_graph(model_graph, device)
        to_dev = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t
        x = to_dev(self.example_input)
        aux = {k: to_dev(v) for k, v in (self.aux or {}).items()}
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        fork_devices = [device.index or 0] if device.type == "cuda" else []

        costs, outputs = [], []
        with torch.random.fork_rng(devices=fork_devices):
            for node in g.nodes:
                raw = [outputs[p] for p in node.predecessors] if node.predecessors else [x]
                kwargs = aux if (not node.predecessors and aux
                                 and getattr(node.module, "accepts_aux_inputs", False)) else {}
                times = []
                for r in range(self.warmup + self.repeats):
                    # clone(): a non-leaf copy, so in-place first ops (ReLU(inplace=True)) work
                    args = [a.detach().requires_grad_(True).clone() if isinstance(a, torch.Tensor)
                            and a.is_floating_point() else a for a in raw]
                    sync()
                    t0 = time.perf_counter()
                    out = node.module(*args, **kwargs)
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        out.backward(torch.ones_like(out))
                    sync()
                    if r >= self.warmup:
                        times.append(time.perf_counter() - t0)
                    node.module.zero_grad(set_to_none=True)
                costs.append(statistics.median(times))
                outputs.append(out.detach() if isinstance(out, torch.Tensor) else out)
        return costs
