"""M9 tensor transports (ipc, nccl) in real OS processes, and device-side cut payloads.

On CPU, ``transport.tensor: nccl`` runs the same code path (rendezvous, metadata
sentinel, rank mapping, hook decode) on the gloo backend, so it is checked here
without GPUs. Real NCCL needs one GPU per slice: on a single GPU it must stop with
a clear reason instead of hanging.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch
import yaml

from torchslicer.sl.hooks import pack_payload, unpack_payload

ROOT = pathlib.Path(__file__).resolve().parent.parent
P2P_CFG = ROOT / "experiments/run_anywhere_p2p.yaml"
CUDA = torch.cuda.is_available()
QUANTIZE = [{"cut": "all", "name": "quantize", "params": {"bits": 8}}]


def _launch(tmp_path, env, hooks=None, check=True):
    cfg = yaml.safe_load(P2P_CFG.read_text())
    if hooks:
        cfg["sl"] = {"cut_hooks": hooks}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, "-m", "torchslicer", "launch",
                        "examples/run_anywhere/train.py", "-c", str(path),
                        "--target", "processes", "-o", str(tmp_path / "out")],
                       cwd=ROOT, capture_output=True, text=True, timeout=240,
                       env={**os.environ, **env})
    if not check:
        return r
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("HISTORY "))
    return json.loads(line[len("HISTORY "):])   # the losses per epoch


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not CUDA, reason="needs CUDA"))])
def test_pack_payload_stays_on_the_payload_device(device):
    payload = [torch.randn(3, 5, device=device), torch.arange(7, device=device, dtype=torch.int8)]
    packed = pack_payload(payload)
    assert packed.device.type == device and packed.dtype == torch.uint8
    back = unpack_payload(packed)
    assert all(b.device.type == device for b in back)
    assert all(torch.equal(a, b) for a, b in zip(payload, back))


def test_unpack_payload_header_longer_than_the_peek():
    payload = [torch.full([1] * 8, float(i)) for i in range(255)]   # ~17 KB of header
    back = unpack_payload(pack_payload(payload))
    assert all(torch.equal(a, b) for a, b in zip(payload, back))


def test_nccl_path_on_gloo_matches_tcp(tmp_path):
    tcp = _launch(tmp_path, {"TENSOR_TRANSPORT": "tcp"})
    nccl = _launch(tmp_path, {"TENSOR_TRANSPORT": "nccl"})
    assert nccl == pytest.approx(tcp, abs=1e-6)


def test_nccl_path_decodes_cut_hooks(tmp_path):
    # before the fix the receiver fed the packed uint8 payload to the next layer
    tcp = _launch(tmp_path, {"TENSOR_TRANSPORT": "tcp"}, hooks=QUANTIZE)
    nccl = _launch(tmp_path, {"TENSOR_TRANSPORT": "nccl"}, hooks=QUANTIZE)
    assert nccl == pytest.approx(tcp, abs=1e-6)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("hooks", [None, QUANTIZE])
def test_ipc_matches_tcp_on_gpu(tmp_path, hooks):
    env = {"DEVICE": "cuda", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    tcp = _launch(tmp_path, {**env, "TENSOR_TRANSPORT": "tcp"}, hooks=hooks)
    ipc = _launch(tmp_path, {**env, "TENSOR_TRANSPORT": "ipc"}, hooks=hooks)
    assert ipc == pytest.approx(tcp, abs=1e-5)


@pytest.mark.skipif(not CUDA or torch.cuda.device_count() != 1, reason="needs exactly one GPU")
def test_nccl_on_one_gpu_fails_fast_with_the_reason(tmp_path):
    r = _launch(tmp_path, {"DEVICE": "cuda", "TENSOR_TRANSPORT": "nccl"}, check=False)
    assert r.returncode != 0
    assert "needs one GPU per slice" in r.stdout + r.stderr


def test_nccl_send_starts_the_metadata_call_before_isend(monkeypatch):
    # Real NCCL connects two ranks at their first transfer: isend blocks until the peer
    # joins, which it does when the metadata call (the shape) reaches it. gloo's isend
    # doesn't block, so the tests above can't see the order. Here isend blocks the same way.
    import threading
    import torch.distributed as dist
    from torchslicer.executors.worker import WorkerServicer

    arrived = threading.Event()

    class _Work:
        def wait(self):
            pass

    def isend(tensor, dst, group=None):
        assert arrived.wait(5), "isend ran before the metadata call: deadlock with real NCCL"
        return _Work()

    monkeypatch.setattr(dist, "isend", isend)
    calls = []
    WorkerServicer()._nccl_send(torch.ones(2), 1, lambda: (arrived.set(), calls.append("meta")))
    assert calls == ["meta"]
