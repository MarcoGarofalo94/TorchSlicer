"""Cost-balancing splitters: pick cuts among ``valid_cuts()`` to minimise the largest
partition cost, for a per-node cost (parameters, FLOPs, measured time)."""
import copy
import statistics
import time
from abc import abstractmethod

import torch

from .base import BaseSplitter


def balanced_cuts(costs: list, valid_cuts: list, n_partitions: int, *,
                  speeds: list | None = None, mem: list | None = None,
                  caps: list | None = None, cut_cost: list | None = None,
                  link_cost: list | None = None) -> list:
    """Choose ``n_partitions - 1`` cuts from ``valid_cuts`` (cut after node k) so that the
    largest partition cost is minimal; ties go to the most even split (sum of squares).

    Partitions are ordered devices. Optional heterogeneous terms (all default to off,
    which gives exactly the homogeneous result):

    - ``speeds[p]``: partition p's cost is divided by it (``cost / speed``);
    - ``mem[k]`` / ``caps[p]``: the summed ``mem`` of partition p must be ``<= caps[p]``
      (``None`` or ``inf`` = no cap);
    - ``cut_cost[k]`` x ``link_cost[p]``: added to partition p when it ends with a cut
      after node k (e.g. activation bytes x seconds per byte of link p → p+1).
    """
    blocks, start = [], 0                 # atomic runs of nodes between valid cut points
    for k in [*valid_cuts, len(costs) - 1]:
        blocks.append((k, sum(costs[start:k + 1]), sum(mem[start:k + 1]) if mem else 0.0))
        start = k + 1
    m = len(blocks)
    if n_partitions > m:
        raise ValueError(
            f"can't make {n_partitions} partitions: the graph has only {m - 1} valid cut points. "
            f"Use fewer partitions, or expand= to trace into submodules for finer cuts."
        )
    if n_partitions == 1 and not caps:
        return []
    prefix, mprefix = [0.0], [0.0]
    for _, c, mb in blocks:
        prefix.append(prefix[-1] + c)
        mprefix.append(mprefix[-1] + mb)
    seg = lambda i, j: prefix[j] - prefix[i]          # cost of blocks i..j-1
    INF = (float("inf"), float("inf"))
    # best[p][j]: (max, sum of squares) for the first j blocks in p partitions
    best = [[INF] * (m + 1) for _ in range(n_partitions + 1)]
    back = [[0] * (m + 1) for _ in range(n_partitions + 1)]
    best[0][0] = (0.0, 0.0)
    for p in range(1, n_partitions + 1):
        cap = caps[p - 1] if caps and caps[p - 1] is not None else None
        for j in range(p, m - (n_partitions - p) + 1):
            for i in range(p - 1, j):
                prev = best[p - 1][i]
                if prev is INF:
                    continue
                if cap is not None and mprefix[j] - mprefix[i] > cap:
                    continue
                c = seg(i, j)
                if speeds:
                    c = c / speeds[p - 1]
                if cut_cost and p < n_partitions:
                    c = c + cut_cost[blocks[j - 1][0]] * link_cost[p - 1]
                cand = (max(prev[0], c), prev[1] + c * c)
                if cand < best[p][j]:
                    best[p][j], back[p][j] = cand, i
    if best[n_partitions][m] is INF:
        raise ValueError(
            f"no split into {n_partitions} partitions fits the workers' memory caps "
            f"(MB: {[round(c / 2**20) if c is not None else None for c in caps]}); "
            f"the estimated memory of the whole model is {round(mprefix[-1] / 2**20)} MB")
    cuts, j = [], m
    for p in range(n_partitions, 0, -1):
        i = back[p][j]
        if p > 1:
            cuts.append(blocks[i - 1][0])
        j = i
    return sorted(cuts)


class CostSplitter(BaseSplitter):
    """Base for splitters that balance a per-node cost. Subclasses implement ``node_costs``.

    Heterogeneous workers: with ``nodes=`` (the executor passes the discovered workers'
    ``NodeInfo``, in slice order) the split minimises the slowest device's time:

    - **speed**: partition p's cost is divided by node p's ``compute_gflops``, if the
      speeds differ by more than ``speed_tolerance`` (a noisy benchmark on identical
      workers must not move the cuts);
    - **memory**: a partition's estimated memory (``memory_factor`` x trainable param
      bytes + frozen param bytes: weights, grads, two Adam states; activations are not
      counted) must fit node p's ``memory_cap_mb()``;
    - **links** (``use_links``, cost strategies measured in time only: ``flops``,
      ``profiled``): each cut adds activation + gradient bytes / the measured bandwidth of
      the link to the next node (``planning.measure_links``).
    """

    cost_unit = "params"      # "params" (no time unit) | "flops" | "seconds"
    speed_tolerance = 1.25
    memory_factor = 4.0
    use_links = True

    @abstractmethod
    def node_costs(self, model_graph) -> list:
        ...

    def split(self, model_graph, n_partitions: int, nodes: list | None = None) -> list:
        costs = [float(c) for c in self.node_costs(model_graph)]
        self.last_costs = costs
        terms = self.device_terms(model_graph, nodes) if nodes else {}
        self.last_terms = terms
        cuts = balanced_cuts(costs, model_graph.valid_cuts(), n_partitions, **terms)
        return self.partitions_from_cuts(model_graph, cuts)

    # ── heterogeneous terms ───────────────────────────────────────────────────

    def device_terms(self, model_graph, nodes: list) -> dict:
        """``balanced_cuts`` keyword arguments for these nodes (empty = homogeneous)."""
        terms = {}
        caps = [nd.memory_cap_mb() * 2**20 if nd.memory_cap_mb() else None for nd in nodes]
        if any(c is not None for c in caps):
            terms["caps"] = caps
            terms["mem"] = self.node_memory(model_graph)

        gflops = [float(nd.compute_gflops) for nd in nodes]
        known = all(g > 0 for g in gflops)
        hetero = known and max(gflops) / min(gflops) > self.speed_tolerance
        links = [nd.link_bandwidth_mbps for nd in nodes[:-1]]
        out_bytes = getattr(self, "last_out_bytes", None)
        timed = self.cost_unit in ("flops", "seconds") and known
        if timed and self.use_links and out_bytes and links and all(b > 0 for b in links):
            if not hetero:   # same speed for all, but in seconds so link time adds up
                gflops = [sum(gflops) / len(gflops)] * len(gflops)
            terms["speeds"] = self.speeds_from_gflops(gflops)
            terms["cut_cost"] = [2.0 * b for b in out_bytes]          # activation + its gradient
            terms["link_cost"] = [8.0 / (b * 1e6) for b in links]    # seconds per byte
        elif hetero:
            terms["speeds"] = self.speeds_from_gflops(gflops)
        return terms

    def speeds_from_gflops(self, gflops: list) -> list:
        """Divisors turning a node cost into time on each device (relative for ``params``)."""
        return list(gflops)

    def node_memory(self, model_graph) -> list:
        """Estimated training memory per node, bytes (shared parameters counted once):
        ``memory_factor`` x trainable parameter bytes (weights, gradients, optimizer
        state) + frozen parameters + buffers, plus the activations saved for backward when
        the strategy has an ``example_input`` (measured by ``activation_bytes``)."""
        acts = None
        if getattr(self, "example_input", None) is not None:
            acts = activation_bytes(model_graph, self.example_input, getattr(self, "aux", None))
        seen, out = set(), []
        for node in model_graph.nodes:
            total = 0.0
            for p in node.module.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                nbytes = p.numel() * p.element_size()
                total += nbytes * (self.memory_factor if p.requires_grad else 1.0)
            for b in node.module.buffers():
                if id(b) not in seen:
                    seen.add(id(b))
                    total += b.numel() * b.element_size()
            out.append(total + (acts[len(out)] if acts else 0))
        return out


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
    cost_unit = "flops"

    def speeds_from_gflops(self, gflops: list) -> list:
        return [g * 1e9 / 3.0 for g in gflops]     # forward FLOPs x3 ≈ forward + backward

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
        self.last_out_bytes = [_nbytes(o) for o in outputs]
        return costs


class ProfiledSplitter(_InputSplitter):
    """Balance measured forward + backward time per partition.

    Each node runs in isolation on ``device`` (default: CUDA if available) with its
    real input shapes; the median of ``repeats`` timed runs after ``warmup`` is used.
    Measure on the hardware the slices will run on; with heterogeneous workers,
    subclass and scale the costs per worker.
    """

    name = "profiled"
    cost_unit = "seconds"

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
        self.last_out_bytes = [_nbytes(o) for o in outputs]
        self.profile_device = device
        return costs

    def speeds_from_gflops(self, gflops: list) -> list:
        # costs are seconds on the profiling device: scale by each node's relative speed
        from ..executors.worker import measure_compute_gflops
        ref = measure_compute_gflops(getattr(self, "profile_device", None) or "cpu")
        return [g / ref for g in gflops]


def _nbytes(t) -> int:
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def activation_bytes(model_graph, example_input, aux: dict | None = None) -> list:
    """Bytes each node saves for backward on ``example_input`` (parameters excluded).

    Runs the real modules node by node, keeping one node's graph at a time, on the
    device they are on, without changing them: the RNG state is forked and buffers
    (BatchNorm running stats) are restored afterwards.
    """
    params = {p.data_ptr() for n in model_graph.nodes for p in n.module.parameters()}
    saved = [(b, b.detach().clone()) for n in model_graph.nodes for b in n.module.buffers()]
    counts, outputs = [], []
    current = [0]

    def pack(t):
        if t.data_ptr() not in params:
            current[0] += _nbytes(t)
        return t

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        for node in model_graph.nodes:
            args = [outputs[p] for p in node.predecessors] if node.predecessors else [example_input]
            # a fresh leaf, cloned (as SplitLayer.forward does) so in-place first ops
            # (ResNet's ReLU(inplace=True)) don't hit "a leaf Variable ... in-place operation"
            args = [a.detach().requires_grad_(True).clone() if isinstance(a, torch.Tensor)
                    and a.is_floating_point() else a for a in args]
            kwargs = aux if (not node.predecessors and aux
                             and getattr(node.module, "accepts_aux_inputs", False)) else {}
            current[0] = 0
            with torch.enable_grad():
                out = node.module(*args, **kwargs)
            counts.append(current[0])
            outputs.append(out.detach() if isinstance(out, torch.Tensor) else out)
    with torch.no_grad():
        for b, v in saved:
            b.copy_(v)
    return counts


class MemorySplitter(ParamSplitter):
    """The fewest slices (at least ``n``) whose estimated training memory each fits
    ``slot_mb``, balanced by parameter count: for out-of-core training
    (``ts.OffloadExecutor``), where every slice must fit one device slot.

    A slice's memory is its parameters and buffers, the gradients and ``state_tensors``
    optimizer-state copies of its trainable parameters (AdamW: 2), and, with an
    ``example_input`` (one micro-batch), the activations its recompute keeps for
    backward. ``optimizer`` (``{"name": ..., "params": ...}``) measures ``state_tensors``.
    """

    name = "memory"

    def __init__(self, slot_mb: float | None = None, optimizer: dict | None = None,
                 state_tensors: int | None = None, example_input=None, aux: dict | None = None):
        super().__init__()
        self.slot_mb = slot_mb
        if state_tensors is None and optimizer is not None:
            from ..offload.executor import optimizer_state_tensors
            state_tensors = optimizer_state_tensors(optimizer)
        self.state_tensors = 2 if state_tensors is None else state_tensors
        self.example_input = example_input
        self.aux = aux

    @property
    def memory_factor(self) -> float:          # weights + gradients + optimizer state
        return 2.0 + self.state_tensors

    def split(self, model_graph, n_partitions: int, nodes: list | None = None) -> list:
        if not self.slot_mb:
            return super().split(model_graph, n_partitions, nodes)
        costs = [float(c) for c in self.node_costs(model_graph)]
        mem = self.node_memory(model_graph)
        self.last_memory = mem
        cap = self.slot_mb * 2**20
        valid = model_graph.valid_cuts()
        for n in range(max(1, n_partitions), len(valid) + 2):
            try:
                cuts = balanced_cuts(costs, valid, n, mem=mem, caps=[cap] * n)
            except ValueError:
                continue
            return self.partitions_from_cuts(model_graph, cuts)
        biggest = max(mem) / 2**20
        raise ValueError(f"no split fits slot_mb={self.slot_mb}: the largest node alone needs "
                         f"~{biggest:.0f} MB; raise slot_mb, or expand= / pack= for finer cuts")
