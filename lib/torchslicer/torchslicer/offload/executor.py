"""``OffloadExecutor``: train a model that does not fit in device memory on one node, by
paging its slices in one at a time (M5).

Schedule per batch (layer-major, as in L2L and Harmony):

1. **forward** slice by slice, each over all micro-batches, without an autograd graph;
   only every slice's input (the cut tensors) is kept, plus the RNG state before each
   slice/micro-batch;
2. the **last slice** runs forward with a graph, the loss and backward right away;
3. **backward** in reverse slice order: recompute the slice's forward from its stored
   input with the stored RNG state (dropout masks match) and its buffers restored
   afterwards (BatchNorm running stats update once), backward, then the optimizer step
   right away, and the slice is paged out while the previous one was already paged in.

While slice ``i`` computes, the pager prefetches the next slice into the other slot.
The cut hooks and the ``CommMeter`` stay in the loop (``_cut_forward``/``_cut_backward``),
so the SL metrics match the other executors. Compute time is measured with CUDA events,
not ``synchronize``, so measuring doesn't serialize the copies.
"""
import time

import torch

from ..executors.local import LocalExecutor, _unpack_inputs
from ..monitor import tracer
from ..monitor.process_logger import get_logger
from .pager import SlicePager

_LOG = get_logger("torchslicer.offload")


def _resolve_device(value: str) -> torch.device:
    from ..executors.worker import resolve_device
    return torch.device(resolve_device(value or None))


def measure_bandwidth(device, nbytes: int = 32 * 2**20, reps: int = 8, trials: int = 3) -> tuple:
    """Pinned host ↔ device copy bandwidth in bytes/s: ``(h2d, d2h, duplex)``, where
    ``duplex`` is the total of both directions copying at once. The best of ``trials``
    (the hardware can do at least that, so the bound it gives is conservative). Small
    buffers, so it fits a tight memory cap."""
    device = torch.device(device)
    if device.type != "cuda":
        return float("inf"), float("inf"), float("inf")
    host = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    dev = [torch.empty(nbytes, dtype=torch.uint8, device=device) for _ in range(2)]
    streams = [torch.cuda.Stream(device), torch.cuda.Stream(device)]

    def timed(pairs):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            for (src, dst), st in zip(pairs, streams):
                with torch.cuda.stream(st):
                    dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(device)
        return nbytes * reps * len(pairs) / (time.perf_counter() - t0)
    out = [0.0, 0.0, 0.0]
    for _ in range(trials):
        out[0] = max(out[0], timed([(host[0], dev[0])]))
        out[1] = max(out[1], timed([(dev[1], host[1])]))
        out[2] = max(out[2], timed([(host[0], dev[0]), (dev[1], host[1])]))
    del host, dev
    return tuple(out)


def optimizer_state_tensors(optimizer_cfg: dict) -> int:
    """How many parameter-sized state tensors the optimizer keeps (AdamW: 2, SGD with
    momentum: 1), measured on a one-element parameter."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = getattr(torch.optim, optimizer_cfg["name"])([p], **optimizer_cfg["params"])
    p.grad = torch.zeros(1)
    opt.step()
    return sum(1 for v in opt.state[p].values() if isinstance(v, torch.Tensor) and v.dim() > 0)


def host_memory_available() -> int:
    """Bytes this process can still allocate on the host: the cgroup limit when there is
    one (containers; ``free`` shows the host's RAM there), else ``MemAvailable``."""
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        with open("/sys/fs/cgroup/memory.current") as f:
            current = int(f.read())
        cache = 0                     # clean page cache is reclaimed before an OOM kill
        with open("/sys/fs/cgroup/memory.stat") as f:
            stat = dict(line.split() for line in f)
        cache = int(stat.get("file", 0)) - int(stat.get("file_dirty", 0)) - int(stat.get("file_writeback", 0))
        if limit != "max":
            cg = int(limit) - current + max(cache, 0)
            avail = cg if avail is None else min(avail, cg)
    except (OSError, ValueError):
        pass
    return avail if avail is not None else 1 << 62


def choose_optimizer_placement(optimizer_cfg: dict, n_params: int, h2d: float, d2h: float) -> tuple:
    """``"gpu"`` or ``"cpu"`` for the optimizer step, by measurement.

    The GPU step moves the optimizer state in and out plus the parameters out, per step;
    the CPU step moves the gradients out and runs the step on the host. Returns
    ``(choice, details)``: the CPU step time and the extra transfer time it saves.
    """
    n = max(1, min(n_params, 32 * 2**20))
    p = torch.nn.Parameter(torch.zeros(n))
    opt = getattr(torch.optim, optimizer_cfg["name"])([p], **optimizer_cfg["params"])
    p.grad = torch.randn(n) * 1e-3
    opt.step()                                         # creates the state; warm-up
    k = optimizer_state_tensors(optimizer_cfg)
    t0 = time.perf_counter()
    for _ in range(2):
        opt.step()
    cpu_s = (time.perf_counter() - t0) / 2 * n_params / n
    nbytes = 4 * n_params
    gpu_extra_s = k * nbytes / h2d + (k + 1) * nbytes / d2h   # state in, state + params out
    cpu_extra_s = cpu_s + nbytes / d2h                        # step + grads out
    choice = "cpu" if cpu_extra_s < gpu_extra_s else "gpu"
    return choice, {"cpu_step_s": round(cpu_s, 4), "gpu_transfer_s": round(gpu_extra_s, 4),
                    "state_tensors": k}


def _shared_trainable(split_layers) -> bool:
    """A trainable parameter used by several slices (tied weights)."""
    seen = {}
    for i, sl in enumerate(split_layers):
        for p in sl.parameters():
            if p.requires_grad and seen.setdefault(id(p), i) != i:
                return True
    return False


class OffloadExecutor(LocalExecutor):
    """Local executor whose slices live in host RAM (``mode="cpu"``) or files
    (``mode="disk"``) and are paged into ``device`` one at a time.

    Arguments left ``None`` come from ``run_config.offload`` (then the defaults: cpu,
    auto, 2 slots, ``network.device``).
    """

    def __init__(self, mode: str = None, optimizer: str = None, device: str = None,
                 slots: int = None, directory: str = None, direct_io: bool = None):
        super().__init__()
        self._args = {"mode": mode, "optimizer": optimizer, "device": device,
                      "slots": slots, "dir": directory, "direct_io": direct_io}
        self.pager: SlicePager = None
        self.device = None
        self.placement_info = {}
        self._bw = (float("inf"),) * 3
        self._timers = []
        self._epoch_bytes = (0, 0)

    def _opt(self, key, cfg, default):
        v = self._args[key]
        if v is None and cfg is not None:
            v = getattr(cfg.offload, key) or None
        return default if v is None else v

    def setup(self, model_graph, partitions, optimizer_cfg, criterion_cfg, mixed_precision=False,
              n_micro_batches=1, run_config=None, **kwargs):
        cfg = run_config
        if kwargs.get("eval_data") is not None:
            raise ValueError("evaluation (eval_data / data.eval_every) isn't supported with "
                             "offload yet: the weights live paged out of the device")
        mode = self._opt("mode", cfg, "cpu")
        mode = "cpu" if mode == "none" else mode
        dev_arg = self._opt("device", cfg, None) or (cfg.network.device if cfg else None)
        self.device = _resolve_device(dev_arg)
        super().setup(model_graph, partitions, optimizer_cfg, criterion_cfg,
                      mixed_precision=mixed_precision, n_micro_batches=n_micro_batches,
                      run_config=run_config, **kwargs)
        for a in self._attacks:
            if a.needs_query or a.needs_client:
                raise ValueError(f"{type(a).__name__} runs the client slices, which offload keeps "
                                 "paged out of the device; use the local executor")
        shared = _shared_trainable(self.split_layers)
        self._bw = measure_bandwidth(self.device)
        placement = self._opt("optimizer", cfg, "auto")
        if placement == "auto":
            if mode == "disk" or shared or self.device.type != "cuda":
                placement = "gpu"
            else:
                n_params = sum(p.numel() for sl in self.split_layers for p in sl.parameters()
                               if p.requires_grad)
                placement, self.placement_info = choose_optimizer_placement(
                    optimizer_cfg, n_params, *self._bw[:2])
        self._check_host_memory(mode, placement, optimizer_cfg)
        self.pager = SlicePager(self.split_layers, self.device, mode=mode, optimizer_on=placement,
                                n_slots=int(self._opt("slots", cfg, 2)),
                                directory=self._opt("dir", cfg, None),
                                direct_io=bool(self._opt("direct_io", cfg, False)))
        self.criterion = self.criterion.to(self.device)
        for prof in self._profilers:
            prof.device = self.device
        self._sync = None
        _LOG.info("offload mode=%s optimizer=%s device=%s slots=%d %s",
                  mode, placement, self.device, len(self.pager.slots), self.pager.footprint())

    def _check_host_memory(self, mode: str, placement: str, optimizer_cfg: dict) -> None:
        """The cpu tier must fit in host RAM: fail now, with the numbers, rather than be
        OOM-killed after the first step creates the optimizer state."""
        if mode != "cpu":
            return
        params = {id(p): p for sl in self.split_layers for p in sl.parameters()}.values()
        t_bytes = sum(p.numel() * p.element_size() for p in params if p.requires_grad)
        k = optimizer_state_tensors(optimizer_cfg)
        # The parameters are already in RAM (the model; each moves to its pinned home and
        # its old storage is freed), so what's new is the optimizer state, plus the
        # gradients when the step runs on the CPU.
        need = k * t_bytes + (t_bytes if placement == "cpu" else 0)
        avail = host_memory_available()
        if need > 0.9 * avail:
            raise MemoryError(
                f"offload mode 'cpu' needs ~{need / 2**30:.1f} GiB more host RAM (optimizer state "
                f"{k} x {t_bytes / 2**30:.1f} GiB{', gradients' if placement == 'cpu' else ''}) but ~{avail / 2**30:.1f} GiB is available (cgroup-aware); "
                f"use mode 'disk' (offload.dir) or a smaller model")

    # ── timing without synchronize ─────────────────────────────────────────────

    def _start(self, i: int):
        if self.device.type == "cuda":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return (i, ev)
        return (i, time.perf_counter())

    def _stop(self, t) -> None:
        i, start = t
        if self.device.type == "cuda":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._timers.append((i, start, ev))
        else:
            self._meter.add_compute(i, time.perf_counter() - start)

    def _flush_timers(self) -> None:
        for i, a, b in self._timers:
            b.synchronize()
            self._meter.add_compute(i, a.elapsed_time(b) / 1000.0)
        self._timers = []

    # ── one batch ──────────────────────────────────────────────────────────────

    def _standard_batch(self, inputs, labels, batch_id: int) -> float:
        return self._offload_batch(inputs, labels, batch_id, 1)

    def _gpipe_batch(self, inputs, labels, batch_id: int) -> float:
        return self._offload_batch(inputs, labels, batch_id, self._n_micro)

    def _rng_state(self):
        cuda = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        return torch.get_rng_state(), cuda

    def _replay(self, sl, x, aux, rng):
        """Recompute ``sl``'s forward with a graph, with the RNG state of the first time.
        Returns ``(out, restore)``: call ``restore()`` after the backward to put the
        buffers back as the first forward left them (BatchNorm running stats update once;
        not before the backward, which may have saved them, e.g. cuDNN batch norm)."""
        saved = [(b, b.detach().clone()) for b in sl.buffers()]
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(rng[0])
            if rng[1] is not None:
                torch.cuda.set_rng_state(rng[1], self.device)
            out = sl(x, **aux) if aux else sl(x)

        def restore():
            with torch.no_grad():
                for b, v in saved:
                    b.copy_(v)
        return out, restore

    def _offload_batch(self, inputs, labels, batch_id: int, M: int) -> float:
        dev, pager, layers = self.device, self.pager, self.split_layers
        n, last = len(layers), len(layers) - 1
        gpu_opt = pager.optimizer_on == "gpu"
        main, aux = _unpack_inputs(inputs)
        main, labels = main.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
        aux = {k: v.to(dev, non_blocking=True) for k, v in aux.items()}
        if M > 1:
            mains, labs = list(main.chunk(M)), list(labels.chunk(M))
            M = len(mains)
            auxs = [{k: v.chunk(M)[m] if v.dim() > 0 and v.shape[0] == main.shape[0] else v
                     for k, v in aux.items()} for m in range(M)]
        else:
            mains, labs, auxs = [main], [labels], [aux]
        self._count_io(main, aux, labels, msgs=M)
        bid = (lambda m: batch_id * M + m) if M > 1 else (lambda m: batch_id)
        self._set_truth(mains, labs, [bid(m) for m in range(M)], scale=1.0 / M)

        # 1. forward without a graph; keep each slice's inputs and RNG states
        stash, rng = [None] * n, [None] * n
        xs = mains
        pager.fetch(0, state=(last == 0))
        for i in range(last):
            sl = layers[i]
            pager.acquire(i)
            pager.fetch(i + 1, state=(i + 1 == last))
            stash[i], rng[i], outs = xs, [], []
            with tracer.span("torchslicer.partition.forward", partition=i, batch_id=batch_id):
                t = self._start(i)
                with self._profilers[i].phase("forward"), torch.no_grad():
                    for m in range(M):
                        rng[i].append(self._rng_state())
                        a = auxs[m] if i == 0 else None
                        outs.append(sl(xs[m], **a) if a else sl(xs[m]))
                        sl.pop_moe_aux_loss()            # recomputed with a graph in backward
                self._stop(t)
            pager.release(i, "buffers")
            xs = [self._cut_forward(i, y, bid(m)) for m, y in enumerate(outs)]

        # 2. last slice: forward with a graph, loss, backward, step
        sl = layers[last]
        pager.acquire(last, grads=True)
        if last > 0:
            pager.fetch(last - 1, state=True)
        total = torch.zeros((), device=dev)
        grads = []
        with tracer.span("torchslicer.partition.backward", partition=last, batch_id=batch_id):
            t = self._start(last)
            for m in range(M):
                a = auxs[m] if last == 0 else None
                with self._profilers[last].phase("forward"):
                    out = sl(xs[m], **a) if a else sl(xs[m])
                with self._profilers[last].phase("backward"):
                    loss = self.criterion(out, labs[m])
                    total += loss.detach()
                    moe = sl.pop_moe_aux_loss()
                    scaled = loss / M if M > 1 else loss
                    if moe is not None:
                        scaled = scaled + (moe / M if M > 1 else moe)
                    scaled.backward()
                grads.append(sl.x.grad if sl.x is not None else None)
            self._step(last)
            self._stop(t)

        # 3. backward in reverse order: recompute, backward, step
        for i in range(last - 1, -1, -1):
            sl = layers[i]
            pager.acquire(i, grads=True)
            if i > 0:
                pager.fetch(i - 1, state=True)
            new = []
            with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id):
                for m in range(M):
                    t = self._start(i)
                    with self._profilers[i].phase("backward"):
                        a = auxs[m] if i == 0 else None
                        out, restore = self._replay(sl, stash[i][m], a, rng[i][m])
                        g = self._cut_backward(i, grads[m], out, bid(m))
                        moe = sl.pop_moe_aux_loss()
                        if moe is not None:
                            out.backward(g, retain_graph=True)
                            (moe / M if M > 1 else moe).backward()
                        else:
                            out.backward(g)
                        restore()
                    new.append(sl.x.grad if sl.x is not None else None)
                    self._stop(t)
                t = self._start(i)
                self._step(i)
                self._stop(t)
            grads = new
        loss = total.item() / M
        self._flush_timers()
        pager.end_batch()
        return loss

    def _step(self, i: int) -> None:
        sl = self.split_layers[i]
        if self.pager.optimizer_on == "gpu":
            with self._profilers[i].phase("optimizer"):
                sl.optimize()
            self.pager.release(i, "all" if sl.optimizer is not None else "buffers")
        else:
            sl.x = None
            self.pager.step_on_cpu(i)

    # ── metrics, end of run ────────────────────────────────────────────────────

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False,
                    total_epochs: int = 0) -> dict:
        self._epoch_bytes = (self.pager.bytes_h2d, self.pager.bytes_d2h)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return super().train_epoch(data_loader, epoch=epoch, verbose=verbose,
                                   total_epochs=total_epochs)

    def _epoch_extras(self, duration_s: float) -> dict:
        h2d = self.pager.bytes_h2d - self._epoch_bytes[0]
        d2h = self.pager.bytes_d2h - self._epoch_bytes[1]
        out = {"offload_h2d_bytes": h2d, "offload_d2h_bytes": d2h, **self.pager.footprint(),
               "offload_optimizer": self.pager.optimizer_on}
        if self.device.type == "cuda":
            # Transfer bound: the bytes moved at the measured pinned-copy bandwidth. H2D and
            # D2H run on separate streams, so each direction bounds the epoch, and so does
            # their sum at the measured both-directions-at-once bandwidth.
            bound = max(h2d / self._bw[0], d2h / self._bw[1], (h2d + d2h) / self._bw[2])
            out.update({
                "offload_peak_mb": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1),
                "offload_h2d_gbps": round(self._bw[0] / 1e9, 2),
                "offload_d2h_gbps": round(self._bw[1] / 1e9, 2),
                "offload_duplex_gbps": round(self._bw[2] / 1e9, 2),
                "offload_transfer_bound_s": round(bound, 3),
                "offload_bound_fraction": round(bound / duration_s, 3) if duration_s else 0.0,
            })
        return out

    def teardown(self) -> None:
        if self.pager is not None:
            self.pager.close()
        super().teardown()
