"""A server slice that any number of data owners reach over the network, one batch at a time.

``SplitServer`` hosts the server side of a split model (slices ``1 .. n-1``, one shared copy,
trained by every client in turn: vanilla SL with many clients, or SplitFed V2 when something
outside aggregates the client slices). ``SplitClient`` holds the client slice (slice 0) and
the data: for each batch it sends the cut activation and the labels, gets back the gradient
of the activation, and steps its own optimizer. A client is not tied to a session or an
executor, so another framework can run it, e.g. as a Flower ``ClientApp``
(``torchslicer.interop.flower.split_client_app``).

    server = ts.SplitServer(model, cuts=["2"], optimizer={"name": "SGD", "params": {"lr": 0.1}},
                            criterion={"name": "CrossEntropyLoss", "params": {}}, port=50070)
    server.start()
    client = ts.SplitClient(model, cuts=["2"], address="server:50070",
                            optimizer={"name": "SGD", "params": {"lr": 0.1}})
    client.fit(loader, epochs=1)

The server applies its optimizer step per batch, under a lock: concurrent clients are
served one batch at a time. A cut hook (``cut_hook=``) runs on the cut as in the other
executors: ``encode``/``decode_grad`` on the client, ``decode``/``encode_grad`` on the server,
each side with its own copy. Tensors cross as packed payloads (``pack_payload``), never
pickled. Labels go to the server (vanilla SL). Give each side its own model instance, built
by the same code (the slices wrap the model's layers); the server's client slice is unused.
``multi_tensor=True`` (on both sides) cuts where several tensors cross, as in ``ts.slice``.
"""
from __future__ import annotations

import threading
from concurrent import futures

import torch
import torch.nn as nn
import torch.optim as optim

from ..core.split_layer import SplitLayer
from ..core.tied import find_tied_parameters, optimizer_parameters
from .hooks import CutContext, pack_payload, unpack_payload

_SERVICE = "torchslicer.SplitServer"
_MAX_MSG = 256 * 1024 * 1024
_GRPC_OPTS = [("grpc.max_send_message_length", _MAX_MSG), ("grpc.max_receive_message_length", _MAX_MSG)]


def _slices(model: nn.Module, cuts: list, cut_hook=None, **slice_kwargs) -> tuple[list, object]:
    """The model's slices as ``SplitLayer``s, in order (no optimizers yet), and the hook to
    run on cut 0 (wrapped when several tensors cross it)."""
    from .. import slice as ts_slice
    from ..core.multicut import wrap_packed_hooks
    sliced = ts_slice(model, cuts=cuts, **slice_kwargs)
    hook = wrap_packed_hooks({0: cut_hook}, sliced.graph, sliced.partitions)[0] if cut_hook else None
    layers = sliced.graph.get_layers()
    parts = [[layers[j] for j in p.layer_indices] for p in sliced.partitions]
    for g in find_tied_parameters(parts):
        if g.requires_grad and 0 in g.members and len(g.members) > 1:
            raise ValueError("a trainable parameter is tied across the client and the server slices; "
                             "SplitServer/SplitClient cannot keep the copies equal")
    n = len(parts)
    return [SplitLayer(parts[i], is_last=(i == n - 1), predecessors=p.predecessors or None)
            for i, p in enumerate(sliced.partitions)], hook


def _optimizer(module: nn.Module, cfg: dict):
    params = optimizer_parameters(module)
    return getattr(optim, cfg["name"])(params, **(cfg.get("params") or {})) if params else None


def _to_bytes(tensors: list) -> bytes:
    return pack_payload([t.detach().cpu() for t in tensors]).numpy().tobytes()


def _from_bytes(raw: bytes) -> list:
    return unpack_payload(torch.frombuffer(bytearray(raw), dtype=torch.uint8))


def _flat(payload) -> list:
    """A hook payload (tensor or list) as ``[is_list flag, *tensors]``."""
    if isinstance(payload, torch.Tensor):
        return [torch.tensor([0]), payload]
    return [torch.tensor([1]), *payload]


def _unflat(tensors: list):
    return list(tensors[1:]) if int(tensors[0]) else tensors[1]


class SplitServer:
    """The server slices of ``model`` (all but slice 0), served on ``port``."""

    def __init__(self, model: nn.Module, cuts: list, optimizer: dict, criterion: dict,
                 port: int = 50070, device: str = "cpu", cut_hook=None, max_workers: int = 8,
                 multi_tensor: bool = False):
        slices, cut_hook = _slices(model, cuts, cut_hook, multi_tensor=multi_tensor)
        self.slices = [s.to(device) for s in slices[1:]]
        self.optimizers = [_optimizer(s, optimizer) for s in self.slices]
        self.criterion = getattr(nn, criterion["name"])(**(criterion.get("params") or {}))
        self.device, self.port, self.hook = device, port, cut_hook
        self.steps = 0
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._server = None

    # ── one batch ─────────────────────────────────────────────────────────────

    def _activation(self, tensors: list, batch_id: int) -> torch.Tensor:
        payload = _unflat([t.to(self.device) for t in tensors])
        if self.hook is not None:
            payload = self.hook.decode(payload, CutContext(0, "forward", batch_id, len(self.slices) + 1))
        return payload

    def train_step(self, act_payload: list, labels: torch.Tensor) -> tuple[list, float]:
        """Forward + backward + step on one batch; returns (gradient payload, loss)."""
        with self._lock:
            self.steps += 1
            z = self._activation(act_payload, self.steps)
            labels = labels.to(self.device)
            for s in self.slices:
                s.train()
            outs, x = [], z
            for s in self.slices:
                x = s(x)
                outs.append(x)
            loss = self.criterion(outs[-1], labels)
            g = None
            for i in range(len(self.slices) - 1, -1, -1):
                g = (self.slices[i].backward(loss=loss) if i == len(self.slices) - 1
                     else self.slices[i].backward(prev_g=g, out=outs[i]))
            for s, opt in zip(self.slices, self.optimizers):
                if opt is not None:
                    opt.step()
                    opt.zero_grad()
            if self.hook is not None:
                g = self.hook.encode_grad(g, CutContext(0, "backward", self.steps, len(self.slices) + 1))
            return _flat(g), float(loss.detach())

    @torch.no_grad()
    def eval_step(self, act_payload: list, labels: torch.Tensor) -> list:
        """``[loss * n, correct, n]`` on one batch (correct counts argmax hits for ``(N, C)``
        outputs, else 0)."""
        with self._lock:
            x = self._activation(act_payload, 0)
            labels = labels.to(self.device)
            for s in self.slices:
                s.eval()
                x = s.run(x)
            n = labels.shape[0]
            correct = (x.argmax(1) == labels).sum().item() if x.dim() == 2 and labels.dim() == 1 else 0
            return [float(self.criterion(x, labels)) * n, float(correct), float(n)]

    # ── gRPC ──────────────────────────────────────────────────────────────────

    def _train_rpc(self, raw, context):
        t = _from_bytes(raw)
        g, loss = self.train_step(t[1:], t[0])
        return _to_bytes([torch.tensor([loss], dtype=torch.float64), *g])

    def _eval_rpc(self, raw, context):
        t = _from_bytes(raw)
        return _to_bytes([torch.tensor(self.eval_step(t[1:], t[0]), dtype=torch.float64)])

    def start(self) -> "SplitServer":
        import grpc
        handler = grpc.method_handlers_generic_handler(_SERVICE, {
            "Train": grpc.unary_unary_rpc_method_handler(self._train_rpc),
            "Eval": grpc.unary_unary_rpc_method_handler(self._eval_rpc),
        })
        self._server = grpc.server(futures.ThreadPoolExecutor(self._max_workers), options=_GRPC_OPTS)
        self._server.add_generic_rpc_handlers((handler,))
        bound = self._server.add_insecure_port(f"[::]:{self.port}")
        if not bound:
            raise OSError(f"SplitServer: cannot bind port {self.port}")
        self.port = bound
        self._server.start()
        return self

    def stop(self, grace: float = 1.0) -> None:
        if self._server is not None:
            self._server.stop(grace).wait()
            self._server = None

    def wait(self) -> None:
        self._server.wait_for_termination()


class SplitClient:
    """Slice 0 of ``model``, trained against a ``SplitServer`` at ``address``."""

    def __init__(self, model: nn.Module, cuts: list, address: str, optimizer: dict,
                 device: str = "cpu", cut_hook=None, timeout: float = 600.0, multi_tensor: bool = False):
        slices, cut_hook = _slices(model, cuts, cut_hook, multi_tensor=multi_tensor)
        self.slice = slices[0].to(device)
        self.optimizer_cfg = optimizer
        self.optimizer = _optimizer(self.slice, optimizer)
        self.device, self.hook, self.timeout = device, cut_hook, timeout
        self.address = address
        self._train = self._eval = None
        self.steps = 0
        self.bytes_sent = self.bytes_received = 0

    def _connect(self):
        if self._train is None:
            import grpc
            ch = grpc.insecure_channel(self.address, options=_GRPC_OPTS)
            self._train = ch.unary_unary(f"/{_SERVICE}/Train")
            self._eval = ch.unary_unary(f"/{_SERVICE}/Eval")

    # weights as a state dict, e.g. for a federated aggregation of the client slices
    def state_dict(self) -> dict:
        return self.slice.state_dict()

    def load_state_dict(self, state: dict, reset_optimizer: bool = True) -> None:
        self.slice.load_state_dict(state)
        if reset_optimizer:
            self.optimizer = _optimizer(self.slice, self.optimizer_cfg)

    def _send(self, z, ctx):
        payload = self.hook.encode(z, ctx) if self.hook is not None else z
        return _flat(payload)

    def train_batch(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self._connect()
        self.steps += 1
        self.slice.train()
        z = self.slice(x.to(self.device))
        raw = _to_bytes([y, *self._send(z.detach(), CutContext(0, "forward", self.steps, 2))])
        reply = self._train(raw, timeout=self.timeout)
        self.bytes_sent += len(raw)
        self.bytes_received += len(reply)
        t = _from_bytes(reply)
        g = _unflat([u.to(self.device) for u in t[1:]])
        if self.hook is not None:
            g = self.hook.decode_grad(g, CutContext(0, "backward", self.steps, 2))
        self.slice.backward(prev_g=g, out=z)
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return float(t[0])

    def fit(self, loader, epochs: int = 1) -> dict:
        """Train on ``loader``; ``{"loss": mean batch loss, "num_examples": samples seen}``."""
        losses, n = [], 0
        for _ in range(epochs):
            for x, y in loader:
                losses.append(self.train_batch(x, y))
                n += len(y)
        return {"loss": sum(losses) / max(1, len(losses)), "num_examples": n}

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """``{"loss", "accuracy", "num_examples"}`` over ``loader`` (the server's criterion)."""
        self._connect()
        self.slice.eval()
        tot = [0.0, 0.0, 0.0]
        for x, y in loader:
            z = self.slice.run(x.to(self.device))
            raw = _to_bytes([y, *self._send(z, CutContext(0, "forward", 0, 2))])
            r = _from_bytes(self._eval(raw, timeout=self.timeout))[0].tolist()
            tot = [a + b for a, b in zip(tot, r)]
        n = max(1.0, tot[2])
        return {"loss": tot[0] / n, "accuracy": tot[1] / n, "num_examples": int(tot[2])}
