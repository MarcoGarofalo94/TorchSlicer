"""
TinyGPT + LoRA P2P worker — parameter-efficient fine-tuning via split learning.

Identical to examples/train/lm/worker/main.py except:
  1. peft.get_peft_model() wraps TinyGPT with LoRA adapters on the QKV
     attention projections (frozen base weights, trainable A/B matrices).
  2. ts.peft_unwrap() extracts the inner model so ts.slice() sees the
     original children (embed, block_0..3, head) as usual.
  3. The optimizer only receives trainable (requires_grad=True) parameters
     — LoRA A/B matrices only; frozen base weights are skipped automatically
     via the trainable-params filter in LocalExecutor / WorkerServicer.

LoRA config (defaults):
  r=8, lora_alpha=16, target_modules=["qkv"]
  trainable params: ~12K  (vs ~830K total for TinyGPT)

Environment variables:
  IS_DRIVER          true/false (default false)
  WORKER_INDEX       0-based index
  WORKER_PEERS       comma-separated "host:port" list in slice-assignment order
  WORKER_ADDRESS     address advertised to peers
  EXPERIMENT_CONFIG  path to YAML experiment config
  LORA_R             LoRA rank (default 8)
  LORA_ALPHA         LoRA alpha (default 16)
"""

import io
import os
import sys
import socket
import threading
import time
from concurrent import futures

import grpc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.executors.worker import (
    WorkerServicer,
    serialize_tensor,
    get_available_memory_mb,
    _channel,
    _GRPC_OPTS,
)
from torchslicer.transport.grpc.worker import worker_service_pb2, worker_service_pb2_grpc
from torchslicer.transport.grpc.coordinator import (
    coordinator_service_pb2,
    coordinator_service_pb2_grpc,
)
from torchslicer.core.split_layer import SplitLayer
from torchslicer.monitor import tracer, WorkerProfiler
from torchslicer.monitor.run_logger import RunLogger
from torchslicer.monitor.callback import TrainingCallback
from torchslicer.discovery.base import NodeInfo
from torchslicer.config import RunConfig


# ── model ─────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 256    # byte-level
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 4
MAX_SEQ    = 64


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.d_head  = D_MODEL // N_HEADS
        self.qkv     = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(MAX_SEQ, MAX_SEQ)).view(1, 1, MAX_SEQ, MAX_SEQ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL, bias=False),
            nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(D_MODEL)
        self.ff   = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TokenEmbedding(nn.Module):
    """Token + positional embedding. Accepts int64 byte IDs."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return self.tok(x) + self.pos(torch.arange(T, device=x.device))


class LMHead(nn.Module):
    """LayerNorm + linear projection. Returns logits for the last position only."""

    def __init__(self):
        super().__init__()
        self.ln   = nn.LayerNorm(D_MODEL)
        self.proj = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x[:, -1, :]))   # [B, VOCAB_SIZE]


class TinyGPT(nn.Module):
    """
    Flat sequential model: each direct child is an opaque leaf for TorchSlicer's
    shallow tracer.  Input: int64 byte IDs [B, MAX_SEQ].  Output: [B, VOCAB_SIZE].
    """

    def __init__(self):
        super().__init__()
        self.embed   = TokenEmbedding()
        self.block_0 = TransformerBlock()
        self.block_1 = TransformerBlock()
        self.block_2 = TransformerBlock()
        self.block_3 = TransformerBlock()
        self.head    = LMHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return self.head(x)


def _pretrain_local(model: nn.Module, data_loader, epochs: int,
                    device: torch.device, lr: float = 3e-4) -> nn.Module:
    """
    Full-parameter local pre-training on the driver before applying LoRA.
    Runs a standard training loop (no split) so the base weights are meaningful
    before they get frozen by get_peft_model().
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    model     = model.to(device)
    model.train()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total += loss.item(); n += 1
        print(f"[pretrain] epoch {epoch}  loss={total/n:.4f}")
    model.cpu()
    return model


def build_model(data_loader=None, cfg=None) -> nn.Module:
    """
    Phase 1 (optional): local full-parameter pre-training on the driver.
    Phase 2: apply LoRA adapters (freezes base weights, adds trainable A/B).
    Phase 3: ts.peft_unwrap() so ts.slice() sees the original children.

    PRETRAIN_EPOCHS env var controls phase 1 (default 5; set 0 to skip).
    """
    from peft import LoraConfig, get_peft_model

    lora_r          = int(os.environ.get("LORA_R",          "8"))
    lora_alpha      = int(os.environ.get("LORA_ALPHA",      "16"))
    pretrain_epochs = int(os.environ.get("PRETRAIN_EPOCHS", "5"))

    model  = TinyGPT()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrain_epochs > 0 and data_loader is not None:
        lr = float((cfg.training.optimizer.get("params", {}) if cfg else {}).get("lr", 3e-4))
        print(f"[lora] phase 1: pre-training {pretrain_epochs} epoch(s) "
              f"(all params, lr={lr}, device={device}) ...")
        model = _pretrain_local(model, data_loader, pretrain_epochs, device, lr)
        print("[lora] phase 1 complete — applying LoRA adapters ...")

    lora_cfg   = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                            target_modules=["qkv"], bias="none")
    peft_model = get_peft_model(model, lora_cfg)

    n_total     = sum(p.numel() for p in peft_model.parameters())
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[lora] r={lora_r}  alpha={lora_alpha}  "
          f"trainable={n_trainable:,}/{n_total:,} ({100*n_trainable/n_total:.2f}%)")

    return ts.peft_unwrap(peft_model)


# ── dataset ───────────────────────────────────────────────────────────────────

def get_dataset(data_dir: str = "/workspace", batch_size: int = 64,
                n_train: int = 50_000):
    """Byte-level LM on .md files under data_dir. Samples up to n_train examples."""
    corpus = bytearray()
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".md"):
            path = os.path.join(data_dir, fname)
            try:
                corpus += open(path, "rb").read()
            except OSError:
                pass

    if len(corpus) < MAX_SEQ + 2:
        raise RuntimeError(
            f"corpus too short ({len(corpus)} bytes); need at least {MAX_SEQ + 2}")

    ids = torch.frombuffer(bytes(corpus), dtype=torch.uint8).long()
    n   = min(len(ids) - MAX_SEQ, n_train)

    torch.manual_seed(42)
    indices = torch.randperm(len(ids) - MAX_SEQ)[:n]
    inputs  = torch.stack([ids[i : i + MAX_SEQ] for i in indices])
    targets = ids[indices + MAX_SEQ]

    print(f"[dataset] corpus={len(corpus):,} bytes  samples={n:,}  "
          f"vocab={VOCAB_SIZE}  seq_len={MAX_SEQ}")
    return DataLoader(TensorDataset(inputs, targets),
                      batch_size=batch_size, shuffle=True, drop_last=True)


# ── P2P infrastructure (identical to lm/worker/main.py) ──────────────────────

class _P2PCoordinatorServicer(
    coordinator_service_pb2_grpc.CoordinatorServiceServicer
):
    def __init__(self):
        self._batch_done   = threading.Event()
        self._batch_losses: list[float] = []
        self._lock = threading.Lock()

    def batch_done(self, request, context):
        self._batch_done.set()
        return coordinator_service_pb2.Empty()

    def report_metrics(self, request, context):
        with self._lock:
            self._batch_losses.append(request.loss)
        return coordinator_service_pb2.Empty()

    def register(self, request, context):
        return coordinator_service_pb2.RegisterResponse(
            ok=False, run_id="", worker_index=0,
            message="P2P topology does not use coordinator registration",
        )

    def signal_batch_done(self):
        self._batch_done.set()

    def wait_batch(self) -> float:
        self._batch_done.wait()
        self._batch_done.clear()
        with self._lock:
            loss = (sum(self._batch_losses) / len(self._batch_losses)
                    if self._batch_losses else 0.0)
            self._batch_losses.clear()
        return loss


class P2PDriverServicer(WorkerServicer):
    def __init__(self, coordinator_svc: _P2PCoordinatorServicer):
        super().__init__()
        self._coordinator_svc = coordinator_svc
        self._last_stub = None

    def set_last_stub(self, stub):
        self._last_stub = stub

    def run_own_forward(self, batch_id: int, inputs: torch.Tensor):
        try:
            self._profiler.begin_batch(batch_id)
            self._profiler.mark_idle_end("fwd")
            tensor = inputs.to(self.device)
            with self._profiler.phase("forward"):
                out   = self.layer(tensor)
                x_ref = self.layer.x
            with self._lock:
                self._outputs[batch_id] = (out, x_ref)
            with self._profiler.phase("send_fwd"):
                self._next_stub.forward(worker_service_pb2.ForwardRequest(
                    batch_id=batch_id,
                    input=serialize_tensor(out),
                ))
            self._profiler.mark_idle_start("bwd")
        except Exception as e:
            print(f"[p2p-driver forward] ERROR batch_id={batch_id}: {e}")
            import traceback; traceback.print_exc()

    def _send_backward(self, batch_id: int, grad: torch.Tensor, is_last_micro: bool = True):
        if self._prev_stub:
            super()._send_backward(batch_id, grad, is_last_micro)
        else:
            if is_last_micro:
                self._coordinator_svc.signal_batch_done()
            self._profiler.mark_idle_start("fwd")
            self._profiler.end_batch()


def _build_layer_configs(layers: list) -> list:
    configs = []
    for layer in layers:
        buf = io.BytesIO()
        torch.save(layer, buf)
        configs.append(worker_service_pb2.LayerConfig(
            layer_type=layer.__class__.__name__,
            serialized=buf.getvalue(),
        ))
    return configs


def _build_optimizer_config(cfg: dict):
    extra = {k: v for k, v in cfg.get("params", {}).items() if k != "lr"}
    buf   = io.BytesIO()
    torch.save(extra, buf)
    return worker_service_pb2.OptimizerConfig(
        name        = cfg["name"],
        lr          = float(cfg["params"].get("lr", 0.001)),
        extra_params= buf.getvalue(),
    )


def _build_criterion_config(cfg: dict):
    buf = io.BytesIO()
    torch.save(cfg.get("params", {}), buf)
    return worker_service_pb2.CriterionConfig(
        name        = cfg["name"],
        extra_params= buf.getvalue(),
    )


def _follower_stats_to_dict(resp, epoch: int) -> dict:
    def _ps(ps, name: str) -> dict:
        if ps.count == 0:
            return {}
        return {
            f"{name}_avg_ms":   round(ps.avg_ms,   3),
            f"{name}_min_ms":   round(ps.min_ms,   3),
            f"{name}_max_ms":   round(ps.max_ms,   3),
            f"{name}_p95_ms":   round(ps.p95_ms,   3),
            f"{name}_total_ms": round(ps.total_ms, 3),
        }
    entry = {
        "step": epoch, "epoch": epoch,
        "worker": resp.worker_index, "phase": "worker_epoch",
        "n_batches": resp.n_batches,
    }
    for name, proto_field in [
        ("forward",   resp.forward),   ("backward",  resp.backward),
        ("optimizer", resp.optimizer), ("send_fwd",  resp.send_fwd),
        ("send_bwd",  resp.send_bwd),  ("idle_fwd",  resp.idle_fwd),
        ("idle_bwd",  resp.idle_bwd),
    ]:
        entry.update(_ps(proto_field, name))
    if resp.peak_mem_mb > 0:
        entry["peak_mem_mb"] = round(resp.peak_mem_mb, 2)
    if resp.end_mem_mb > 0:
        entry["end_mem_mb"] = round(resp.end_mem_mb, 2)
    return entry


def _configure_driver_slice(driver, partition, all_layers, peers, opt_cfg, run_id, cfg):
    driver._reset_state()
    own_layers   = [all_layers[j] for j in partition.layer_indices]
    predecessors = (
        [list(p) for p in partition.predecessors]
        if partition.predecessors else None
    )
    driver.layer         = SplitLayer(own_layers, is_last=False, predecessors=predecessors)
    driver.is_last       = False
    driver._run_id       = run_id
    driver._worker_index = 0
    driver.prev_worker   = None
    driver.next_worker   = peers[1] if len(peers) > 1 else None
    driver._n_micro      = cfg.pipeline.n_micro if cfg.pipeline.use_gpipe else 1

    if driver.next_worker:
        driver._next_stub = worker_service_pb2_grpc.WorkerServiceStub(
            _channel(driver.next_worker))

    extra     = {k: v for k, v in opt_cfg.get("params", {}).items() if k != "lr"}
    trainable = [p for p in driver.layer.parameters() if p.requires_grad]
    params    = trainable if trainable else list(driver.layer.parameters())
    opt       = getattr(optim, opt_cfg["name"])(
        params,
        lr=float(opt_cfg.get("params", {}).get("lr", 0.001)),
        **extra,
    )
    driver.layer.set_optimizer(opt)
    driver.layer = driver.layer.to(driver.device)
    driver._profiler = WorkerProfiler(
        verbosity=cfg.profile.verbosity,
        memory=cfg.profile.memory,
        device=driver.device,
    )
    layer_names = [type(l).__name__ for l in driver.layer.layers]
    n_trainable = sum(p.numel() for p in params)
    print(f"[p2p-driver] own slice configured: layers={layer_names}  "
          f"next={driver.next_worker}  device={driver.device}  "
          f"trainable_params={n_trainable:,}")


def run_training(driver, coordinator_svc, last_stub, data_loader, cfg,
                 follower_stubs=None, run_logger=None, callbacks=None, verbose=True):
    n_micro        = cfg.pipeline.n_micro if cfg.pipeline.use_gpipe else 1
    n_total        = len(data_loader)
    callbacks      = callbacks or []
    follower_stubs = follower_stubs or []

    for cb in callbacks:
        try: cb.on_train_begin(run_id=cfg.run_id, config={})
        except Exception as e: print(f"[callback] on_train_begin error: {e}")

    for epoch in range(cfg.training.epochs):
        total_loss = data_load_ms = send_ms = wait_ms = 0.0
        n_batches  = 0
        epoch_t0   = time.perf_counter()

        for cb in callbacks:
            try: cb.on_epoch_begin(epoch)
            except Exception as e: print(f"[callback] on_epoch_begin error: {e}")

        iter_t0 = time.perf_counter()
        for inputs, labels in data_loader:
            data_load_ms += (time.perf_counter() - iter_t0) * 1000.0
            batch_id = epoch * n_total + n_batches

            send_t0 = time.perf_counter()
            if n_micro > 1:
                mi, ml = inputs.chunk(n_micro), labels.chunk(n_micro)
                for m in range(n_micro):
                    mbid = batch_id * n_micro + m
                    last_stub.forward(worker_service_pb2.ForwardRequest(
                        batch_id=mbid, label=serialize_tensor(ml[m])))
                    driver._pool.submit(driver.run_own_forward, mbid, mi[m])
            else:
                last_stub.forward(worker_service_pb2.ForwardRequest(
                    batch_id=batch_id, label=serialize_tensor(labels)))
                driver._pool.submit(driver.run_own_forward, batch_id, inputs)
            send_ms += (time.perf_counter() - send_t0) * 1000.0

            wait_t0 = time.perf_counter()
            loss = coordinator_svc.wait_batch()
            wait_ms += (time.perf_counter() - wait_t0) * 1000.0

            total_loss += loss
            n_batches  += 1

            if run_logger:
                run_logger.log(step=batch_id, epoch=epoch, batch=n_batches,
                               loss=round(loss, 6), phase="batch")
            if verbose:
                print(f"  [epoch {epoch} | batch {n_batches}/{n_total}] loss={loss:.4f}")

            iter_t0 = time.perf_counter()

        avg      = total_loss / n_batches if n_batches else 0.0
        duration = time.perf_counter() - epoch_t0
        print(f"[epoch {epoch}] avg_loss={avg:.4f}  duration={duration:.1f}s")

        epoch_metrics = {"step": epoch, "epoch": epoch, "loss": round(avg, 6),
                         "duration_s": round(duration, 3), "phase": "epoch"}
        for cb in callbacks:
            try:
                result = cb.on_epoch_end(epoch, epoch_metrics)
                if isinstance(result, dict): epoch_metrics = result
            except Exception as e: print(f"[callback] on_epoch_end error: {e}")

        if run_logger:
            run_logger.log(**epoch_metrics)
            if cfg.profile.verbosity >= 1:
                run_logger.log(step=epoch, epoch=epoch,
                               data_load_total_ms=round(data_load_ms, 3),
                               send_total_ms=round(send_ms, 3),
                               wait_total_ms=round(wait_ms, 3),
                               n_batches=n_batches, phase="coordinator_epoch")
                driver_summary = driver._profiler.epoch_summary(epoch)
                driver._profiler.reset_epoch()
                driver_entry = {"step": epoch, "epoch": epoch, "worker": 0,
                                "phase": "worker_epoch",
                                "n_batches": driver_summary.get("n_batches", 0)}
                for phase_name in ("forward", "backward", "optimizer",
                                   "send_fwd", "send_bwd", "idle_fwd", "idle_bwd"):
                    for stat in ("avg_ms", "min_ms", "max_ms", "p95_ms", "total_ms"):
                        k = f"{phase_name}_{stat}"
                        if k in driver_summary:
                            driver_entry[k] = driver_summary[k]
                run_logger.log(**driver_entry)
                for fi, (peer_addr, stub) in enumerate(follower_stubs):
                    try:
                        resp = stub.get_stats(worker_service_pb2.GetStatsRequest(
                            run_id=cfg.run_id, epoch=epoch,
                            verbosity=cfg.profile.verbosity))
                        run_logger.log(**_follower_stats_to_dict(resp, epoch))
                    except Exception as e:
                        print(f"[get_stats] follower {fi+1} ({peer_addr}): {e}")

    log_history = run_logger.log_history if run_logger else []
    for cb in callbacks:
        try: cb.on_train_end(log_history)
        except Exception as e: print(f"[callback] on_train_end error: {e}")


def _init_with_retry(stub, slice_cfg, peer_addr: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    attempt  = 0
    while time.monotonic() < deadline:
        try:
            return stub.init(slice_cfg)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                if attempt == 0:
                    print(f"[p2p-driver] waiting for follower {peer_addr} ...")
                time.sleep(1.0); attempt += 1
            else:
                print(f"[p2p-driver] init → {peer_addr}: FAILED — {e}"); return None
    print(f"[p2p-driver] init → {peer_addr}: timeout after {timeout:.0f}s"); return None


# ── entry point ───────────────────────────────────────────────────────────────

def serve():
    tracer.auto_configure_if_env()

    port         = sys.argv[1] if len(sys.argv) > 1 else "50051"
    is_driver    = os.environ.get("IS_DRIVER", "false").lower() in ("true", "1")
    worker_index = int(os.environ.get("WORKER_INDEX", "0"))
    peers_env    = os.environ.get("WORKER_PEERS", "")
    hostname     = socket.gethostname()
    node_address = os.environ.get("WORKER_ADDRESS", f"{hostname}:{port}")

    cfg   = RunConfig.load(os.environ.get("EXPERIMENT_CONFIG"))
    peers = (
        [p.strip() for p in peers_env.split(",") if p.strip()]
        if peers_env else cfg.discovery.peers
    )

    # ── Follower ───────────────────────────────────────────────────────────────
    if not is_driver:
        servicer = WorkerServicer()
        server   = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                               options=_GRPC_OPTS)
        worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        servicer.set_server(server)
        print(f"[p2p-follower] worker_{worker_index} started on {node_address}")
        server.wait_for_termination()
        print(f"[p2p-follower] {hostname} terminated cleanly")
        return

    # ── Driver ─────────────────────────────────────────────────────────────────
    if not peers:
        print("[p2p-driver] ERROR: WORKER_PEERS env var or discovery.peers must be set")
        sys.exit(1)
    n = len(peers)
    if n < 2:
        print("[p2p-driver] ERROR: P2P topology requires at least 2 workers")
        sys.exit(1)

    coordinator_svc = _P2PCoordinatorServicer()
    driver_svc      = P2PDriverServicer(coordinator_svc)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(driver_svc, server)
    coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(
        coordinator_svc, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    driver_svc.set_server(server)
    print(f"[p2p-driver] started on {node_address}  (n_workers={n})")

    opt_cfg  = cfg.training.optimizer
    crit_cfg = cfg.training.criterion
    run_id   = cfg.run_id

    # Build dataset first — needed for local pre-training phase
    batch_size  = opt_cfg.get("batch_size", 64) if isinstance(opt_cfg, dict) else 64
    data_loader = get_dataset(batch_size=batch_size)

    # Phase 1 (local pre-train) + Phase 2 (apply LoRA) + peft_unwrap
    model      = build_model(data_loader=data_loader, cfg=cfg)
    n_params   = sum(p.numel() for p in model.parameters())
    n_train    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sliced     = ts.slice(model, strategy="uniform", n=n)
    all_layers = sliced.graph.get_layers()
    partitions = sliced.partitions
    print(f"[p2p-driver] TinyGPT+LoRA  total={n_params:,}  trainable={n_train:,}  "
          f"vocab={VOCAB_SIZE}  d_model={D_MODEL}  blocks={N_LAYERS}  "
          f"layers={len(all_layers)}")

    # Connect to follower peers and send SliceConfig
    follower_stubs: list[tuple[str, object]] = []
    for peer_addr in peers[1:]:
        stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(peer_addr))
        follower_stubs.append((peer_addr, stub))

    print(f"[p2p-driver] distributing slices to {len(follower_stubs)} follower(s) ...")
    for fi, (peer_addr, stub) in enumerate(follower_stubs):
        wi      = fi + 1
        is_last = (wi == n - 1)
        partition_layers = [all_layers[j] for j in partitions[wi].layer_indices]
        pred_proto = [
            worker_service_pb2.PredecessorList(indices=list(p))
            for p in (partitions[wi].predecessors or [[] for _ in partition_layers])
        ]
        slice_cfg = worker_service_pb2.SliceConfig(
            layers            = _build_layer_configs(partition_layers),
            optimizer         = _build_optimizer_config(opt_cfg),
            criterion         = _build_criterion_config(crit_cfg) if is_last else None,
            is_last           = is_last,
            prev_worker       = peers[wi - 1],
            next_worker       = peers[wi + 1] if wi < n - 1 else "",
            coordinator       = node_address,
            n_micro           = cfg.pipeline.n_micro if cfg.pipeline.use_gpipe else 1,
            run_id            = run_id,
            checkpoint_path   = "",
            worker_index      = wi,
            profile_verbosity = cfg.profile.verbosity,
            profile_memory    = cfg.profile.memory,
            predecessors      = pred_proto,
        )
        res = _init_with_retry(stub, slice_cfg, peer_addr)
        if res is None:
            sys.exit(1)
        print(f"[p2p-driver] init → {peer_addr}  ok={res.ok}  {res.message}")

    _configure_driver_slice(driver_svc, partitions[0], all_layers,
                            peers, opt_cfg, run_id, cfg)

    last_stub = follower_stubs[-1][1]
    driver_svc.set_last_stub(last_stub)

    run_logger = None
    if cfg.logging.enabled:
        run_dir    = os.path.join(cfg.logging.dir, run_id)
        run_logger = RunLogger(run_id=run_id, run_dir=run_dir)
        layer_names = [type(l).__name__ for l in all_layers]
        run_logger.record_executor("p2p")
        run_logger.record_config(cfg)
        run_logger.record_model("TinyGPT+LoRA", layer_names)
        run_logger.record_split(partitions, layer_names, "uniform")
        run_logger.record_workers([
            NodeInfo(node_id=f"worker{i}", address=peers[i],
                     device="cuda" if torch.cuda.is_available() else "cpu",
                     memory_mb=get_available_memory_mb(
                         "cuda" if torch.cuda.is_available() else "cpu"))
            for i in range(n)
        ])

    print(f"[p2p-driver] training start  run_id={run_id}  epochs={cfg.training.epochs}  "
          f"gpipe={cfg.pipeline.use_gpipe}  n_micro={cfg.pipeline.n_micro}")

    run_training(driver_svc, coordinator_svc, last_stub, data_loader, cfg,
                 follower_stubs=follower_stubs, run_logger=run_logger,
                 callbacks=[], verbose=True)

    run_dir = os.path.join(cfg.logging.dir, run_id) if cfg.logging.enabled else ""
    for fi, (peer_addr, stub) in enumerate(follower_stubs):
        try:
            stub.shutdown(worker_service_pb2.ShutdownRequest(
                save_checkpoint=cfg.checkpoint.enabled,
                checkpoint_dir=run_dir,
                run_id=run_id,
                epoch=cfg.training.epochs - 1,
                worker_index=fi + 1,
            ))
            print(f"[p2p-driver] shutdown → {peer_addr}")
        except Exception as e:
            print(f"[p2p-driver] shutdown failed → {peer_addr}: {e}")

    server.stop(grace=2)

    if cfg.checkpoint.enabled and run_logger:
        for i in range(n):
            run_logger.record_artifact(
                "checkpoint", f"worker_{i}_epoch_{cfg.training.epochs - 1}.pt")
    if run_logger:
        run_logger.flush()

    print(f"[p2p-driver] done  run_id={run_id}")


if __name__ == "__main__":
    serve()
