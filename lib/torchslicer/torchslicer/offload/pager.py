"""Slice pager: every slice's tensors live in a slow tier and are paged into the device
one slice at a time ("virtual memory" for slices, M5).

Each slice has a **home** in the slow tier, made of contiguous byte regions:

- ``pb``: its parameters, then its buffers (BatchNorm running stats, ...);
- ``s``: the optimizer state of its parameters (only when the step runs on the device;
  laid out after the first step, when the optimizer has created it);
- ``g``: its gradients, only when the optimizer step runs on the CPU.

Homes are pinned host buffers (``mode="cpu"``) or files (``mode="disk"``). Files are read
and written with ``pread``/``pwrite`` in 4 KiB-aligned chunks spread over a few I/O
threads, through one pinned staging buffer per slot; one I/O "lane" thread per slot keeps
each slot's reads and writes in order. ``direct_io=True`` opens them with ``O_DIRECT``
(no page cache); it is off by default because on the machine M5 was measured on it was
3x slower under training's mixed reads and writes.

The device has a few **slots** (two by default: double buffering), each one byte buffer
allocated once and holding one slice as ``[pb | g | s]``. Paging a slice in is one
``non_blocking`` copy per region on a dedicated H2D stream; the slice's tensors are then
re-pointed at views of the slot (``tensor.data = view``), so the model's
``nn.Parameter`` objects, which the optimizer keys its state by, never change, and the
caching allocator never sees per-page allocations. Changes are written back on a D2H
stream as soon as the slice is released (eager write-back), so evicting a slice later is
free. Evicted tensors point at their home: a view of the pinned buffer (cpu), or a
memory-mapped view of the file (disk; it costs no RAM until read, and the kernel keeps
it coherent with the ``O_DIRECT`` writes). Either way, after training the model's
parameters hold the latest values.

Ordering uses CUDA events only: an H2D into a slot waits for the compute that last used
the slot (``free``) and for the D2H that drained it (``out``); compute waits for the H2D
(``ready``) and the last write-back. Nothing calls ``torch.cuda.synchronize`` on the hot
path. On a CPU device the same code runs with synchronous copies (tests, CPU matrix).
"""
import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch

_ALIGN = 256          # tensor offsets inside a region: a byte view can be re-viewed as any dtype
_PAGE = 4096          # O_DIRECT alignment of file offsets, sizes and buffers
_CHUNK = 16 * 2**20   # one I/O request


def _round(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) // a * a


def _floor(n: int, a: int = _PAGE) -> int:
    return n // a * a


class _Entry:
    """One tensor inside a region: how to read it, how to re-point it, where it sits."""
    __slots__ = ("get", "put", "offset", "nbytes", "dtype", "shape")

    def __init__(self, get, put, t: torch.Tensor, offset: int):
        self.get, self.put = get, put
        self.offset, self.nbytes = offset, t.numel() * t.element_size()
        self.dtype, self.shape = t.dtype, tuple(t.shape)

    def view(self, buf: torch.Tensor) -> torch.Tensor:
        return buf[self.offset:self.offset + self.nbytes].view(self.dtype).view(self.shape)


class _Layout:
    def __init__(self):
        self.entries: list = []
        self.nbytes = 0

    def add(self, get, put, t: torch.Tensor) -> _Entry:
        e = _Entry(get, put, t, self.nbytes)
        self.entries.append(e)
        self.nbytes = _round(self.nbytes + e.nbytes)
        return e


def _set_data(t):
    def put(v):
        t.data = v
    return put


def _set_state(state: dict, key):
    def put(v):
        state[key] = v
    return put


class _FileHome:
    """One region of one slice on disk: ``O_DIRECT`` when the filesystem and the buffers
    allow it (pinned, page-aligned staging), plain ``pread``/``pwrite`` otherwise."""

    def __init__(self, path: str, nbytes: int, direct: bool):
        self.path, self.size = path, _round(max(nbytes, 1), _PAGE)
        with open(path, "wb") as f:
            f.truncate(self.size)
            try:                      # allocate the blocks now: no allocation on the hot path,
                os.posix_fallocate(f.fileno(), 0, self.size)   # and a full disk fails here
            except (AttributeError, OSError) as e:
                if isinstance(e, OSError) and e.errno == 28:   # ENOSPC
                    raise
        flags = os.O_RDWR
        self.direct = False
        if direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(path, flags | os.O_DIRECT)
                self.direct = True
            except OSError:
                pass
        if not self.direct:
            self.fd = os.open(path, flags)

    def io(self, pool, write: bool, buf: torch.Tensor, offset: int) -> None:
        """Read (or write) ``buf`` at file ``offset``, split in chunks over ``pool``."""
        mv = memoryview(buf.numpy())
        n = len(mv)
        jobs = []
        for a in range(0, n, _CHUNK):
            part, at = mv[a:a + _CHUNK], offset + a
            fn = (lambda p=part, o=at: _full(os.pwritev, self.fd, p, o)) if write else \
                 (lambda p=part, o=at: _full(os.preadv, self.fd, p, o))
            jobs.append(pool.submit(fn))
        for j in jobs:
            j.result()

    def close(self) -> None:
        os.close(self.fd)

    def mmap(self) -> torch.Tensor:
        return torch.from_file(self.path, shared=True, size=self.size, dtype=torch.uint8)


def _full(call, fd, mv, offset) -> None:
    done = 0
    while done < len(mv):
        k = call(fd, [mv[done:]], offset + done)
        if k <= 0:
            raise OSError(f"offload: short {'write' if call is os.pwritev else 'read'} at {offset + done}")
        done += k


class _Page:
    """One slice's paging state."""

    def __init__(self, index: int, layer, optimizer, keep: set):
        self.index, self.layer, self.optimizer = index, layer, optimizer
        self.pb, self.g, self.s = _Layout(), _Layout(), None
        seen = set(keep)
        params = []
        for p in layer.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(self.pb.add(lambda p=p: p, _set_data(p), p.data))
        self.buf_start = self.pb.nbytes              # buffers come after the parameters
        for b in layer.buffers():
            if id(b) not in seen:
                seen.add(id(b))
                self.pb.add(lambda b=b: b, _set_data(b), b.data)
        for e in params:
            if e.get().requires_grad:
                self.g.add(e.get, None, e.get().data)
        self.params = params
        self.home = {}                # region → pinned tensor (cpu) or _FileHome (disk)
        self.home_g = None            # gradients' home (optimizer on the CPU)
        self.slot = None              # index of the slot holding it, or None
        self.state_in = False         # optimizer state is in the slot
        self.dirty = None             # None | "buffers" | "all"
        self.home_ready = None        # Future: home consistent after it (disk write, CPU step)
        self.home_ev = None           # event: the last D2H into its home (cpu mode)

    def layout(self, region: str) -> _Layout:
        return self.pb if region == "pb" else self.s

    def fresh_state(self) -> list:
        """Optimizer-state tensors of this slice's parameters (scalar ``step`` counters
        stay where the optimizer keeps them)."""
        if self.optimizer is None:
            return []
        out = []
        for e in self.params:
            st = self.optimizer.state.get(e.get())
            for k, v in (st or {}).items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    out.append((st, k, v))
        return out

    def slot_bytes(self, with_state: bool) -> int:
        s = self.s.nbytes if (with_state and self.s is not None) else 0
        return self.pb.nbytes + self.g.nbytes + s

    def base(self, region: str) -> int:
        """Where ``region`` starts in a slot."""
        return 0 if region == "pb" else self.pb.nbytes + self.g.nbytes

    @property
    def g_off(self) -> int:
        return self.pb.nbytes


class _Slot:
    def __init__(self, index: int):
        self.index = index
        self.buf = None
        self.cap = 0
        self.page: _Page | None = None
        self.busy = False
        self.ready = None             # event (or Future of one, disk): H2D into the slot done
        self.free = None              # event: compute stopped using it
        self.out = None               # event: D2H out of it done
        self.staging = None           # disk mode: pinned, page-aligned, the size of the slot
        self.lane = None              # disk mode: I/O thread, FIFO per slot
        self.last = 0                 # use counter, for LRU eviction
        self.fresh = False            # allocated, not yet copied into


def _resolve(ev):
    return ev.result() if isinstance(ev, Future) else ev


class SlicePager:
    """Pages ``split_layers`` between a home tier and ``n_slots`` device slots.

    ``optimizer_on``: ``"gpu"`` = the step runs on the device, so the optimizer state is
    paged with the slice; ``"cpu"`` = gradients go home and the step runs on the host in
    a background thread (as in ZeRO-Offload, but exact: no delayed update).
    Tensors used by several slices (tied weights, a rotary-embedding buffer shared by
    every block, ...) are not paged: they stay resident on the device (``resident``).
    ``io_threads``: parallel requests per disk transfer (queue depth).
    """

    def __init__(self, split_layers, device, mode: str = "cpu", optimizer_on: str = "gpu",
                 n_slots: int = 2, directory: str | None = None, io_threads: int = 4,
                 direct_io: bool = False):
        if mode not in ("cpu", "disk"):
            raise ValueError(f"offload mode must be 'cpu' or 'disk', not {mode!r}")
        if optimizer_on not in ("gpu", "cpu"):
            raise ValueError(f"offload optimizer must be 'gpu' or 'cpu', not {optimizer_on!r}")
        if mode == "disk" and optimizer_on == "cpu":
            raise ValueError("offload mode 'disk' runs the optimizer step on the device "
                             "(offload.optimizer: gpu)")
        if n_slots < 2:
            raise ValueError("offload needs at least 2 slots (one computes, one prefetches)")
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.mode, self.optimizer_on = mode, optimizer_on
        self.pin = self.cuda
        self.dir = directory
        self.disk = mode == "disk"
        self.direct_io = direct_io
        if self.disk:
            if not directory:
                raise ValueError("offload mode 'disk' needs offload.dir")
            os.makedirs(directory, exist_ok=True)
            self._io = ThreadPoolExecutor(io_threads, thread_name_prefix="ts-offload-io")
        if self.cuda:
            self.h2d = torch.cuda.Stream(self.device)
            self.d2h = torch.cuda.Stream(self.device)
        self.bytes_h2d = self.bytes_d2h = 0
        self._clock = 0
        users: dict = {}
        for i, sl in enumerate(split_layers):
            for t in [*sl.parameters(), *sl.buffers()]:
                users.setdefault(id(t), (t, set()))[1].add(i)
        self.resident = [t for t, used in users.values() if len(used) > 1]
        if optimizer_on == "cpu" and any(t.requires_grad for t in self.resident):
            raise ValueError("offload.optimizer 'cpu' doesn't support parameters shared by "
                             "several slices (e.g. tied embeddings); use 'gpu'")
        for t in self.resident:
            t.data = t.data.to(self.device)
        keep = {id(t) for t in self.resident}
        self.pages = [_Page(i, sl, sl.optimizer, keep) for i, sl in enumerate(split_layers)]
        for pg in self.pages:
            self._make_home(pg, "pb", [(e, e.get().detach()) for e in pg.pb.entries])
            if optimizer_on == "cpu" and pg.g.nbytes:
                pg.home_g = torch.empty(pg.g.nbytes, dtype=torch.uint8, pin_memory=self.pin)
        self.slots = [_Slot(i) for i in range(n_slots)]
        if self.disk:
            for s in self.slots:
                s.lane = ThreadPoolExecutor(1, thread_name_prefix=f"ts-offload-lane{s.index}")
        self._cpu_step = (ThreadPoolExecutor(1, thread_name_prefix="ts-offload-step")
                          if optimizer_on == "cpu" else None)
        self._ensure_cap(max((pg.slot_bytes(False) for pg in self.pages), default=0))

    # ── homes ─────────────────────────────────────────────────────────────────

    def _host(self, nbytes: int) -> torch.Tensor:
        return torch.empty(max(nbytes, 1), dtype=torch.uint8, pin_memory=self.pin)

    def _host_aligned(self, nbytes: int) -> torch.Tensor:
        """A host buffer starting on a page boundary (O_DIRECT; pinned buffers from the
        caching host allocator often aren't)."""
        raw = self._host(nbytes + _PAGE)
        skip = (-raw.data_ptr()) % _PAGE
        return raw[skip:skip + nbytes]

    def _make_home(self, pg: _Page, region: str, values) -> None:
        """Give ``region`` a home holding ``values`` (``(entry, tensor)`` pairs) and
        re-point the entries at it."""
        lay = pg.layout(region)
        if not self.disk:
            home = pg.home[region] = self._host(lay.nbytes)
            for e, v in values:
                e.view(home).copy_(v)
                e.put(e.view(home))
            return
        f = pg.home[region] = _FileHome(os.path.join(self.dir, f"slice{pg.index}.{region}"),
                                        lay.nbytes, direct=self.direct_io and self.pin)
        tmp = self._host_aligned(f.size)
        for e, v in values:
            e.view(tmp).copy_(v)
        f.io(self._io, True, tmp[:f.size], 0)
        f.view = f.mmap()
        for e, _ in values:
            e.put(e.view(f.view))

    def _home_view(self, pg: _Page, region: str, e: _Entry) -> torch.Tensor:
        home = pg.home[region]
        return e.view(home.view if self.disk else home)

    # ── events and streams ────────────────────────────────────────────────────

    def _event(self, stream=None):
        if not self.cuda:
            return None
        ev = torch.cuda.Event(blocking=True)     # threads waiting on it sleep, not spin
        ev.record(stream or torch.cuda.current_stream(self.device))
        return ev

    @staticmethod
    def _wait(stream, *events):
        for ev in events:
            ev = _resolve(ev)
            if ev is not None:
                stream.wait_event(ev)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _ensure_cap(self, nbytes: int) -> None:
        """Every slot holds ``nbytes``. Slots grow once, at the end of the first batch,
        when the optimizer state first exists; they are otherwise never reallocated."""
        nbytes = _round(max(nbytes, _ALIGN))
        if all(s.cap >= nbytes for s in self.slots):
            return
        if any(s.busy for s in self.slots):
            raise RuntimeError("offload: slots can only grow between batches")
        self.flush()
        if self.cuda:
            torch.cuda.current_stream(self.device).synchronize()
        for s in self.slots:
            s.buf = s.staging = None
        for s in self.slots:
            # + one page: aligned disk transfers may run past a region's end
            s.buf = torch.empty(nbytes + _PAGE, dtype=torch.uint8, device=self.device)
            s.cap = nbytes
            if self.disk:
                s.staging = self._host_aligned(nbytes + 2 * _PAGE)
            s.ready = s.free = s.out = None
            s.fresh = True

    def _victim(self) -> _Slot:
        idle = [s for s in self.slots if not s.busy]
        if not idle:
            raise RuntimeError("offload: every slot is in use; raise offload.slots")
        empty = [s for s in idle if s.page is None]
        return empty[0] if empty else min(idle, key=lambda s: s.last)

    # ── paging ────────────────────────────────────────────────────────────────

    def fetch(self, i: int, state: bool = False) -> None:
        """Start paging slice ``i`` in (non-blocking); ``state`` also brings its optimizer
        state. A slice already resident stays where it is."""
        pg = self.pages[i]
        want_state = state and self.optimizer_on == "gpu" and pg.s is not None and pg.s.nbytes > 0
        if pg.slot is not None:
            if want_state and not pg.state_in:
                if pg.slot_bytes(True) > self.slots[pg.slot].cap:
                    raise RuntimeError("offload: a slice outgrew the slots mid-batch")
                self._page_in(pg, self.slots[pg.slot], ("s",))
            return
        if pg.slot_bytes(want_state) > self.slots[0].cap:
            raise RuntimeError("offload: a slice outgrew the slots mid-batch")
        s = self._victim()
        self._evict(s)
        self._page_in(pg, s, ("pb", "s") if want_state else ("pb",))

    def acquire(self, i: int, grads: bool = False) -> None:
        """Slice ``i`` is about to compute: compute waits for its page-in (and for the last
        write-back out of the slot, before changing it); for a backward, its trainable
        parameters get zeroed gradients inside the slot."""
        pg = self.pages[i]
        if pg.slot is None:
            self.fetch(i, state=grads)
        s = self.slots[pg.slot]
        s.busy = True
        self._clock += 1
        s.last = self._clock
        if self.cuda:
            self._wait(torch.cuda.current_stream(self.device), s.ready, s.out)
        elif isinstance(s.ready, Future):
            s.ready.result()
        if grads:
            gbuf = s.buf[pg.g_off:]
            for e in pg.g.entries:
                g = e.view(gbuf)
                g.zero_()
                e.get().grad = g

    def release(self, i: int, dirty: str | None) -> None:
        """Slice ``i`` stopped computing; ``dirty`` = what changed in the slot
        (``"buffers"`` after a forward, ``"all"`` after an optimizer step)."""
        pg = self.pages[i]
        s = self.slots[pg.slot]
        s.busy = False
        s.free = self._event()
        if dirty == "all" or (dirty == "buffers" and pg.dirty is None):
            pg.dirty = dirty
        self._writeback(pg, s)       # eagerly: the D2H overlaps the next slice's compute

    def step_on_cpu(self, i: int) -> None:
        """Ship slice ``i``'s gradients (and buffers) home, page it out, and run its
        optimizer step on the host in the background. The next page-in of the slice
        waits for that step."""
        pg = self.pages[i]
        s = self.slots[pg.slot]
        s.busy = False
        s.free = self._event()
        home = pg.home["pb"]
        spans = []
        if pg.g.nbytes:
            spans.append((s.buf[pg.g_off:pg.g_off + pg.g.nbytes], pg.home_g[:pg.g.nbytes]))
        if pg.pb.nbytes > pg.buf_start:   # buffers may have changed; parameters didn't
            a, b = pg.buf_start, pg.pb.nbytes
            spans.append((s.buf[a:b], home[a:b]))
        ev = self._d2h(s, spans)
        pg.home_ev = ev
        for e in pg.pb.entries:
            e.put(e.view(home))
        for e in pg.g.entries:
            e.get().grad = e.view(pg.home_g)
        pg.slot, pg.dirty, s.page = None, None, None
        opt = pg.optimizer

        def step():
            if ev is not None:
                ev.synchronize()
            if opt is not None:
                opt.step()
                opt.zero_grad()
            else:
                for e in pg.g.entries:
                    e.get().grad = None
        pg.home_ready = self._cpu_step.submit(step)

    def _d2h(self, s: _Slot, spans):
        """Copy ``(src in slot, dst in host)`` spans after the slot's compute; returns the
        completion event (also stored as the slot's ``out``)."""
        self.bytes_d2h += sum(src.numel() for src, _ in spans)
        if not spans:
            return None
        if not self.cuda:
            for src, dst in spans:
                dst.copy_(src)
            return None
        self._wait(self.d2h, s.free)
        with torch.cuda.stream(self.d2h):
            for src, dst in spans:
                dst.copy_(src, non_blocking=True)
        s.out = self._event(self.d2h)
        return s.out

    def _regions_in(self, pg: _Page, regions) -> list:
        """``(region, nbytes)`` to page in."""
        out = []
        if "pb" in regions:
            out.append(("pb", pg.pb.nbytes))
        if "s" in regions and pg.s is not None and pg.s.nbytes:
            out.append(("s", pg.s.nbytes))
        return out

    def _page_in(self, pg: _Page, s: _Slot, regions) -> None:
        if isinstance(pg.home_ready, Future):
            pg.home_ready.result()             # its CPU step or disk write
            pg.home_ready = None
        todo = self._regions_in(pg, regions)
        # ``free``/``out`` order the copy after the slot's last users. A newly allocated
        # slot has none, and its first copy raced with compute already queued (NaN losses
        # under GPU load, every time), so that one copy also waits for the compute stream.
        waits = (s.free, s.out, pg.home_ev, self._event() if s.fresh else None)
        s.fresh = False
        if self.disk:
            s.ready = self._disk_read(pg, s, todo, waits)
        elif not self.cuda:
            for r, n in todo:
                b = pg.base(r)
                s.buf[b:b + n].copy_(pg.home[r][:n])
        else:
            self._wait(self.h2d, *waits)
            with torch.cuda.stream(self.h2d):
                for r, n in todo:
                    b = pg.base(r)
                    s.buf[b:b + n].copy_(pg.home[r][:n], non_blocking=True)
            s.ready = self._event(self.h2d)
        self.bytes_h2d += sum(n for _, n in todo)
        pg.slot, s.page = s.index, pg
        for e in pg.pb.entries:
            e.put(e.view(s.buf))
        if "s" in regions and pg.s is not None:
            sbuf = s.buf[pg.base("s"):]
            for e in pg.s.entries:
                e.put(e.view(sbuf))
            pg.state_in = True

    def _disk_read(self, pg: _Page, s: _Slot, todo, waits) -> Future:
        """File → pinned staging (I/O threads) → slot (H2D), on the slot's lane. Returns a
        Future of the H2D event."""
        prev_ready = s.ready

        def job():
            prev = _resolve(prev_ready)
            if prev is not None:
                prev.synchronize()              # the last H2D out of this staging buffer
            for r, n in todo:
                b, f = pg.base(r), pg.home[r]
                stb = _round(b, _PAGE)          # page-aligned place in the staging buffer
                f.io(self._io, False, s.staging[stb:stb + _round(n, _PAGE)], 0)
            if not self.cuda:
                for r, n in todo:
                    b = pg.base(r)
                    stb = _round(b, _PAGE)
                    s.buf[b:b + n].copy_(s.staging[stb:stb + n])
                return None
            self._wait(self.h2d, *waits)
            with torch.cuda.stream(self.h2d):
                for r, n in todo:
                    b = pg.base(r)
                    stb = _round(b, _PAGE)
                    s.buf[b:b + n].copy_(s.staging[stb:stb + n], non_blocking=True)
            return self._event(self.h2d)
        return s.lane.submit(job)

    def _writeback(self, pg: _Page, s: _Slot) -> None:
        """Copy what changed in the slot back home; the slice stays resident."""
        if pg.dirty is None:
            return
        if self.optimizer_on == "gpu" and pg.s is None and pg.dirty == "all":
            self._adopt_state(pg)
        spans = []                             # (region, start, end) inside the region
        if pg.dirty == "all":
            spans.append(("pb", 0, pg.pb.nbytes))
        elif pg.dirty == "buffers" and pg.pb.nbytes > pg.buf_start:
            spans.append(("pb", pg.buf_start, pg.pb.nbytes))
        if pg.state_in and pg.dirty == "all" and pg.s.nbytes:
            spans.append(("s", 0, pg.s.nbytes))
        if self.disk:
            self._disk_write(pg, s, spans)
        elif spans:
            pg.home_ev = self._d2h(s, [(s.buf[pg.base(r) + a:pg.base(r) + b], pg.home[r][a:b])
                                       for r, a, b in spans])
        pg.dirty = None

    def _disk_write(self, pg: _Page, s: _Slot, spans) -> None:
        """Slot → pinned staging (D2H), then the slot's lane writes staging → file, in
        page-aligned ranges (the slot holds the current value of every byte in them)."""
        if not spans:
            return
        staged = []
        if self.cuda:
            self._wait(self.d2h, s.free)
        for r, a, b in spans:
            a, b = _floor(a), min(_round(b, _PAGE), pg.home[r].size)
            base, stb = pg.base(r), _round(pg.base(r), _PAGE)
            src, dst = s.buf[base + a:base + b], s.staging[stb + a:stb + b]
            if self.cuda:
                with torch.cuda.stream(self.d2h):
                    dst.copy_(src, non_blocking=True)
            else:
                dst.copy_(src)
            staged.append((pg.home[r], dst, a))
            self.bytes_d2h += b - a
        ev = self._event(self.d2h) if self.cuda else None
        s.out = ev if ev is not None else s.out

        def job():
            if ev is not None:
                ev.synchronize()
            for f, buf, at in staged:
                f.io(self._io, True, buf, at)
        prev = pg.home_ready
        if isinstance(prev, Future):
            prev.result()
        pg.home_ready = s.lane.submit(job)

    def _adopt_state(self, pg: _Page) -> None:
        """After a slice's first optimizer step: lay out the state tensors the optimizer
        created, copy them home (once, synchronously) and re-point them there; later
        page-ins bring them into the slot with the parameters."""
        found = pg.fresh_state()
        lay = _Layout()
        for st, k, v in found:
            lay.add(lambda st=st, k=k: st[k], _set_state(st, k), v)
        pg.s = lay
        if not found:
            return
        if self.cuda:
            torch.cuda.current_stream(self.device).synchronize()
        self._make_home(pg, "s", [(e, v) for e, (_, _, v) in zip(lay.entries, found)])
        self.bytes_d2h += lay.nbytes

    def _evict(self, s: _Slot) -> None:
        """Page out whatever ``s`` holds: write back what changed, re-point the tensors at
        their home."""
        pg = s.page
        if pg is None:
            return
        self._writeback(pg, s)
        for e in pg.pb.entries:
            e.put(self._home_view(pg, "pb", e))
        if pg.s is not None and "s" in pg.home:
            for e in pg.s.entries:
                e.put(self._home_view(pg, "s", e))
        pg.slot, pg.state_in, s.page = None, False, None

    # ── batch boundaries, end of the run ──────────────────────────────────────

    def end_batch(self) -> None:
        """Between batches. After the first one, the optimizer state exists: lay it out and
        grow the slots to hold it (once)."""
        if self.optimizer_on != "gpu":
            return
        if any(pg.s is None and pg.fresh_state() for pg in self.pages):
            self.flush()
        need = max((pg.slot_bytes(True) for pg in self.pages), default=0)
        self._ensure_cap(need)

    def drain(self) -> None:
        """Wait for every pending transfer, disk write and CPU step."""
        for pg in self.pages:
            if isinstance(pg.home_ready, Future):
                pg.home_ready.result()
                pg.home_ready = None
        for s in self.slots:
            if isinstance(s.ready, Future):
                s.ready.result()
        if self.cuda:
            self.h2d.synchronize()
            self.d2h.synchronize()

    def flush(self) -> None:
        """Page everything out and wait: afterwards every parameter, buffer and
        optimizer-state tensor is at home, holding the latest value."""
        for s in self.slots:
            if s.busy:
                raise RuntimeError("offload: flush while a slice is computing")
            self._evict(s)
        self.drain()

    def close(self) -> None:
        self.flush()
        for s in self.slots:
            if s.lane is not None:
                s.lane.shutdown()
            s.buf = s.staging = None
        if self._cpu_step is not None:
            self._cpu_step.shutdown()
        if self.disk:
            self._io.shutdown()
            for pg in self.pages:
                for f in pg.home.values():
                    f.close()

    def footprint(self) -> dict:
        """Bytes in the home tier and in the device slots."""
        home = sum(pg.pb.nbytes + (pg.s.nbytes if pg.s else 0)
                   + (pg.g.nbytes if pg.home_g is not None else 0) for pg in self.pages)
        return {"offload_home_mb": round(home / 2**20, 1),
                "offload_slots_mb": round(sum(s.cap for s in self.slots) / 2**20, 1)}
