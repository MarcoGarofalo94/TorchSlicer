"""Split-learning metrics: bytes per cut, client-side compute, simulated link time.

Bytes are tensor payload bytes as they go on the wire (after cut hooks; a hook's
payload is counted packed, header included), not gRPC/TCP framing. ``raw`` bytes are
the same tensors before encoding, so ``raw / wire`` is the compression a hook gives.
Every executor fills one ``CommMeter`` per epoch; ``summary`` turns it into the
fields added to the epoch's history entry and ``metrics.jsonl`` row.
"""

from .roles import Placement


class CommMeter:
    def __init__(self, n_slices: int):
        self.n_slices = n_slices
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

    def add_fwd(self, cut: int, wire: int, raw: int, msgs: int = 1) -> None:
        self.fwd[cut] += wire
        self.raw_fwd[cut] += raw
        self.msgs_fwd[cut] += msgs

    def add_bwd(self, cut: int, wire: int, raw: int, msgs: int = 1) -> None:
        self.bwd[cut] += wire
        self.raw_bwd[cut] += raw
        self.msgs_bwd[cut] += msgs

    def add_input(self, nbytes: int, msgs: int = 1) -> None:
        self.input += nbytes
        self.msgs_input += msgs

    def add_label(self, nbytes: int, msgs: int = 1) -> None:
        self.label += nbytes
        self.msgs_label += msgs

    def add_compute(self, i: int, seconds: float) -> None:
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
