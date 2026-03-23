"""
WorkerProfiler — lightweight per-phase timing and memory profiler for split-learning workers.

Design
------
- Zero overhead when verbosity=0: all methods return immediately.
- verbosity=1: epoch totals per phase (forward, backward, optimizer, send_fwd, send_bwd,
               idle_fwd, idle_bwd).  Aggregates: avg/min/max/p95/total.
- verbosity=2: same as 1 but phases are measured with higher precision and memory snapshots
               are optionally captured (memory=True).
- verbosity=3: raw per-batch records in addition to epoch aggregates.

Thread safety
-------------
WorkerProfiler is used from one compute thread (the ThreadPoolExecutor with max_workers=1)
plus one gRPC handler thread (get_stats RPC).  All mutations are protected by _lock.
Idle marks (mark_idle_start/end) are also serialised because they are called from the same
compute thread.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class _PhaseRecord:
    durations_ms: List[float] = field(default_factory=list)
    peak_mem_mb:  List[float] = field(default_factory=list)  # populated only when memory=True


@dataclass
class _BatchRecord:
    batch_id:     int   = 0
    forward_ms:   float = 0.0
    backward_ms:  float = 0.0
    optimizer_ms: float = 0.0
    send_fwd_ms:  float = 0.0
    send_bwd_ms:  float = 0.0
    idle_fwd_ms:  float = 0.0
    idle_bwd_ms:  float = 0.0
    peak_mem_mb:  float = 0.0


# Maps profiler phase name → _BatchRecord field name
_PHASE_TO_BATCH_FIELD = {
    "forward":   "forward_ms",
    "backward":  "backward_ms",
    "optimizer": "optimizer_ms",
    "send_fwd":  "send_fwd_ms",
    "send_bwd":  "send_bwd_ms",
    "idle_fwd":  "idle_fwd_ms",
    "idle_bwd":  "idle_bwd_ms",
}


class WorkerProfiler:
    """
    Accumulates per-phase timing (and optionally memory) for one worker across an epoch.

    Usage::

        profiler = WorkerProfiler(verbosity=2, memory=True, device=device)

        # In forward handler:
        profiler.mark_idle_end("fwd")
        with profiler.phase("forward"):
            out = layer(x)
        with profiler.phase("send_fwd"):
            next_stub.forward(...)
        profiler.mark_idle_start("bwd")

        # After each epoch the coordinator calls get_stats(); the worker calls:
        summary = profiler.epoch_summary(epoch)
        records = profiler.batch_records()   # only populated at verbosity=3
        profiler.reset_epoch()
    """

    def __init__(self, verbosity: int = 0, memory: bool = False, device=None):
        self.verbosity = verbosity
        self.memory    = memory
        self.device    = device
        self._lock     = threading.Lock()
        self._idle_start: Dict[str, float] = {}
        self._reset()

    # ── public API ──────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        return self.verbosity > 0

    @contextmanager
    def phase(self, name: str):
        """Time a named phase.  No-op when verbosity=0."""
        if not self.is_active():
            yield
            return

        if self.memory and self.device is not None:
            self._cuda_reset_peak()

        t0 = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        peak_mb = 0.0
        if self.memory and self.device is not None:
            peak_mb = self._cuda_peak_mb()

        with self._lock:
            rec = self._phases.setdefault(name, _PhaseRecord())
            rec.durations_ms.append(elapsed_ms)
            if self.memory:
                rec.peak_mem_mb.append(peak_mb)

            if self.verbosity >= 3 and self._current_batch is not None:
                field_name = _PHASE_TO_BATCH_FIELD.get(name)
                if field_name:
                    setattr(self._current_batch, field_name, elapsed_ms)

    def mark_idle_start(self, kind: str) -> None:
        """Record the moment a worker starts waiting.  kind = 'fwd' or 'bwd'."""
        if not self.is_active():
            return
        self._idle_start[kind] = time.perf_counter()

    def mark_idle_end(self, kind: str) -> None:
        """Record the end of an idle period and store the duration."""
        if not self.is_active():
            return
        t0 = self._idle_start.pop(kind, None)
        if t0 is None:
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        phase_name = f"idle_{kind}"
        with self._lock:
            rec = self._phases.setdefault(phase_name, _PhaseRecord())
            rec.durations_ms.append(elapsed_ms)
            if self.verbosity >= 3 and self._current_batch is not None:
                field_name = _PHASE_TO_BATCH_FIELD.get(phase_name)
                if field_name:
                    setattr(self._current_batch, field_name, elapsed_ms)

    def begin_batch(self, batch_id: int) -> None:
        """Start a per-batch record (only used at verbosity=3)."""
        if self.verbosity < 3:
            return
        with self._lock:
            self._current_batch = _BatchRecord(batch_id=batch_id)

    def end_batch(self) -> None:
        """Finalise the current per-batch record (only used at verbosity=3)."""
        if self.verbosity < 3:
            return
        with self._lock:
            if self._current_batch is not None:
                if self.memory and self.device is not None:
                    self._current_batch.peak_mem_mb = self._cuda_peak_mb()
                self._batch_records.append(self._current_batch)
                self._current_batch = None

    def epoch_summary(self, epoch: int) -> dict:
        """Return a flat dict of aggregated stats for this epoch."""
        with self._lock:
            summary: dict = {"epoch": epoch}
            for name, rec in self._phases.items():
                d = rec.durations_ms
                if not d:
                    continue
                n = len(d)
                sorted_d = sorted(d)
                summary[f"{name}_avg_ms"]   = round(sum(d) / n, 3)
                summary[f"{name}_min_ms"]   = round(sorted_d[0], 3)
                summary[f"{name}_max_ms"]   = round(sorted_d[-1], 3)
                summary[f"{name}_p95_ms"]   = round(sorted_d[max(0, int(0.95 * n) - 1)], 3)
                summary[f"{name}_total_ms"] = round(sum(d), 3)
                if self.memory and rec.peak_mem_mb:
                    summary[f"{name}_peak_mem_mb"] = round(max(rec.peak_mem_mb), 3)
            summary["n_batches"] = max(
                (len(r.durations_ms) for r in self._phases.values()), default=0
            )
            return summary

    def batch_records(self) -> List[dict]:
        """Return raw per-batch timing records (only populated at verbosity=3)."""
        with self._lock:
            return [
                {
                    "batch_id":    r.batch_id,
                    "forward_ms":  r.forward_ms,
                    "backward_ms": r.backward_ms,
                    "optimizer_ms":r.optimizer_ms,
                    "send_fwd_ms": r.send_fwd_ms,
                    "send_bwd_ms": r.send_bwd_ms,
                    "idle_fwd_ms": r.idle_fwd_ms,
                    "idle_bwd_ms": r.idle_bwd_ms,
                    "peak_mem_mb": r.peak_mem_mb,
                }
                for r in self._batch_records
            ]

    def reset_epoch(self) -> None:
        """Clear all accumulated data — call after get_stats() at epoch end."""
        with self._lock:
            self._reset()

    # ── internal ────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._phases:        Dict[str, _PhaseRecord] = {}
        self._batch_records: List[_BatchRecord]       = []
        self._current_batch: Optional[_BatchRecord]   = None

    def _cuda_reset_peak(self) -> None:
        try:
            import torch
            if self.device is not None and str(self.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass

    def _cuda_peak_mb(self) -> float:
        try:
            import torch
            if self.device is not None and str(self.device).startswith("cuda"):
                return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        except Exception:
            pass
        return 0.0
