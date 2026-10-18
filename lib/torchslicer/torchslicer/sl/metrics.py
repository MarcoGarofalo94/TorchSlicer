"""Split-learning metrics: bytes per cut, client-side compute, simulated link time.

Bytes are tensor payload bytes as they go on the wire (after cut hooks; a hook's
payload is counted packed, header included), not gRPC/TCP framing. ``raw`` bytes are
the same tensors before encoding, so ``raw / wire`` is the compression a hook gives.
Every executor fills one ``CommMeter`` per epoch; ``summary`` turns it into the
fields added to the epoch's history entry and ``metrics.jsonl`` row.
"""

from .roles import Placement


class CommMeter:
    def __init__(self, n_slices: int, per_client: bool = True):
        self.n_slices = n_slices
        self._per_client = per_client
        self.reset()

    def reset(self) -> None:
        n_cuts = max(0, self.n_slices - 1)
        self.fwd = [0] * n_cuts
        self.bwd = [0] * n_cuts
        self.raw_fwd = [0] * n_cuts
        self.raw_bwd = [0] * n_cuts
        self.msgs_fwd = [0] * n_cuts
        self.msgs_bwd = [0] * n_cuts
        self.input = self.label = 0
        self.msgs_input = self.msgs_label = 0
        self.compute = [0.0] * self.n_slices
        self.batches = 0
        self.client = None          # the session's active client: its counts also go to
        self.clients: dict = {}     # clients[k], a CommMeter of its own (heterogeneity)

    def _mirror(self):
        if self.client is None or not self._per_client:
            return None
        sub = self.clients.get(self.client)
        if sub is None:
            sub = self.clients[self.client] = CommMeter(self.n_slices, per_client=False)
        return sub

    def add_fwd(self, cut: int, wire: int, raw: int, msgs: int = 1) -> None:
        if (sub := self._mirror()) is not None:
            sub.add_fwd(cut, wire, raw, msgs)
        self.fwd[cut] += wire
        self.raw_fwd[cut] += raw
        self.msgs_fwd[cut] += msgs

    def add_bwd(self, cut: int, wire: int, raw: int, msgs: int = 1) -> None:
        if (sub := self._mirror()) is not None:
            sub.add_bwd(cut, wire, raw, msgs)
        self.bwd[cut] += wire
        self.raw_bwd[cut] += raw
        self.msgs_bwd[cut] += msgs

    def add_input(self, nbytes: int, msgs: int = 1) -> None:
        if (sub := self._mirror()) is not None:
            sub.add_input(nbytes, msgs)
        self.input += nbytes
        self.msgs_input += msgs

    def add_label(self, nbytes: int, msgs: int = 1) -> None:
        if (sub := self._mirror()) is not None:
            sub.add_label(nbytes, msgs)
        self.label += nbytes
        self.msgs_label += msgs

    def add_compute(self, i: int, seconds: float) -> None:
        if (sub := self._mirror()) is not None:
            sub.add_compute(i, seconds)
        self.compute[i] += seconds

    def summary(self, placement: Placement, bandwidth_mbps: float = 0.0, latency_ms: float = 0.0) -> dict:
        p = placement
        cuts = range(len(self.fwd))
        cut_total = sum(self.fwd) + sum(self.bwd)
        total = cut_total + self.input + self.label
        client = sum(self.fwd[c] + self.bwd[c] for c in cuts if p.cut_crosses_owner(c))
        client += (self.input if p.inputs_leave else 0) + (self.label if p.labels_leave else 0)
        raw_total = sum(self.raw_fwd) + sum(self.raw_bwd) + self.input + self.label
        out = {
            "bytes_fwd":       list(self.fwd),
            "bytes_bwd":       list(self.bwd),
            "bytes_input":     self.input,
            "bytes_label":     self.label,
            "bytes_total":     total,
            "bytes_client":    client,
            "bytes_per_batch": round(total / self.batches, 1) if self.batches else 0.0,
            "raw_bytes_total": raw_total,
            "compute_s":       [round(s, 6) for s in self.compute],
            "client_compute_s": round(sum(self.compute[i] for i in p.client_slices), 6),
            "server_compute_s": round(sum(s for i, s in enumerate(self.compute)
                                          if not p.is_client(i)), 6),
            "sl_setting":      p.setting,
        }
        if bandwidth_mbps > 0 or latency_ms > 0:
            def _t(nbytes, msgs):
                bw = nbytes * 8 / (bandwidth_mbps * 1e6) if bandwidth_mbps > 0 else 0.0
                return bw + msgs * latency_ms / 1e3
            per_cut = [_t(self.fwd[c] + self.bwd[c], self.msgs_fwd[c] + self.msgs_bwd[c]) for c in cuts]
            extra = _t(self.input, self.msgs_input) + _t(self.label, self.msgs_label)
            out["sim_comm_s"] = round(sum(per_cut) + extra, 6)
            out["sim_client_comm_s"] = round(
                sum(t for c, t in zip(cuts, per_cut) if p.cut_crosses_owner(c))
                + (_t(self.input, self.msgs_input) if p.inputs_leave else 0.0)
                + (_t(self.label, self.msgs_label) if p.labels_leave else 0.0), 6)
        return out


# ── heterogeneous clients (simulated) ─────────────────────────────────────────

def assign_profiles(profiles: list, n_clients: int, seed: int = 0) -> list:
    """One profile index per client: each ``profiles[i]`` is drawn with probability
    ``share`` (default equal), seeded. ``profiles``: dicts with ``compute_scale``
    (client compute = measured x this), ``bandwidth_mbps``, ``latency_ms``, ``share``."""
    import numpy as np
    if not profiles:
        return []
    shares = np.array([float(p.get("share", 1.0)) for p in profiles])
    if (shares < 0).any() or shares.sum() <= 0:
        raise ValueError("sl.client_profiles: shares must be >= 0 and not all 0")
    for p in profiles:
        unknown = set(p) - {"name", "share", "compute_scale", "bandwidth_mbps", "latency_ms"}
        if unknown:
            raise ValueError(f"sl.client_profiles: unknown keys {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(len(profiles), size=n_clients, p=shares / shares.sum())]


def client_times(meter: CommMeter, placement: Placement, profiles: list, tiers: list,
                 parallel: bool) -> dict:
    """Simulated time of each client this epoch on its own profile, and of the round.
    A client's time = its client-slice compute x ``compute_scale`` + its bytes and
    messages across the data owner's boundary over its link. The round: the slowest
    client plus the server's compute when clients run in parallel, else the sum."""
    times = {}
    for k, sub in sorted(meter.clients.items()):
        prof = profiles[tiers[k]]
        s = sub.summary(placement, float(prof.get("bandwidth_mbps", 0.0)), float(prof.get("latency_ms", 0.0)))
        times[k] = s["client_compute_s"] * float(prof.get("compute_scale", 1.0)) + s.get("sim_client_comm_s", 0.0)
    server = sum(t for i, t in enumerate(meter.compute) if not placement.is_client(i))
    clients = list(times.values())
    rnd = (max(clients) if parallel else sum(clients)) + server if clients else server
    return {"sim_client_time_s": {k: round(t, 6) for k, t in times.items()},
            "sim_round_s": round(rnd, 6)}

