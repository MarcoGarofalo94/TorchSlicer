import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseExecutor
from ..core.split_layer import SplitLayer
from ..core.tied import find_tied_parameters, non_owned_names, optimizer_parameters
from ..monitor import tracer
from ..monitor.profiler import WorkerProfiler
from ..monitor.process_logger import get_logger, configure as configure_process_logging
from ..monitor.run_logger import RunLogger
from ..monitor.callback import TrainingCallback
from ..sl.hooks import CutContext, resolve_cut_hooks, payload_nbytes, tensor_nbytes
from ..sl.roles import Placement, resolve_setting
from ..sl.metrics import CommMeter, assign_profiles, client_times
from ..sl.clients import (ClientSession, SplitLayerClientState, algorithm_round_end,
                          resolve_client_store, select_clients)
from ..algorithm.base import Algorithm, AlgorithmContext, BackwardPolicy
from ..algorithm.builtin import DefaultAlgorithm
from ..privacy.base import Observation, resolve_privacy

_LOG = get_logger("torchslicer.local")


class _ClientCopy(nn.Module):
    """A frozen copy of client slices for white-box attacks, chained with
    :meth:`SplitLayer.run` so gradients reach its input. ``fresh``: re-initialized
    parameters (the architecture only)."""

    def __init__(self, slices, fresh: bool = False):
        super().__init__()
        self.slices = nn.ModuleList(
            SplitLayer(copy.deepcopy(list(sl.layers)), predecessors=sl._predecessors,
                       mixed_precision=sl._mixed_precision) for sl in slices)
        if fresh:
            for m in self.modules():
                if m is not self and hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        for sl in self.slices:
            x = sl.run(x)
        return x

def _detach_payload(payload):
    if isinstance(payload, torch.Tensor):
        return payload.detach()
    return [t.detach() for t in payload]


def check_session_ties(tied_groups, placement) -> None:
    """Client states are swapped per client; a parameter tied across a client and a
    server slice would then be shared by every client, so it is refused."""
    for g in tied_groups:
        sides = {placement.is_client(p) for p in g.members}
        if len(sides) > 1:
            raise ValueError("multi-client sessions don't support parameters tied across a "
                             "client slice and a server slice (e.g. tied embeddings with "
                             "u_shaped); untie them or change the cuts")


def _unpack_inputs(inputs):
    """Unpack batch inputs into ``(main_tensor, aux_dict)``.

    Supports three DataLoader conventions:

    * ``torch.Tensor``           → ``(inputs, {})``
    * ``dict``                   → ``(inputs["input_ids"], rest)`` — key
      ``"input_ids"`` is the main tensor; everything else is aux.  If
      ``"input_ids"`` is absent, the first key is treated as main.
    * ``(Tensor, dict)``         → ``(inputs[0], inputs[1])``
    """
    if isinstance(inputs, torch.Tensor):
        return inputs, {}
    if isinstance(inputs, dict):
        main_key = "input_ids" if "input_ids" in inputs else next(iter(inputs))
        return inputs[main_key], {k: v for k, v in inputs.items() if k != main_key}
    if (isinstance(inputs, (list, tuple))
            and len(inputs) == 2
            and isinstance(inputs[1], dict)):
        return inputs[0], inputs[1]
    if isinstance(inputs, (list, tuple)):
        return inputs[0], {}
    return inputs, {}


class LocalExecutor(BaseExecutor):
    def __init__(self):
        self.split_layers: list[SplitLayer] = []
        self.criterion = None
        self._n_micro  = 1

        self._run_logger: RunLogger              = None
        self._callbacks:  list[TrainingCallback] = []
        self._profilers:  list[WorkerProfiler]   = []  # one per partition
        self._hooks: dict = {}
        self._placement: Placement = None
        self._meter: CommMeter = None
        self._link = (0.0, 0.0)
        self._sync = None
        self._algorithm: Algorithm = DefaultAlgorithm()
        self._round: int = 0
        self._eval_loader = None
        self._eval_every = 0
        self._client_states = "cpu"
        self._client_store = None
        self._server_store = None
        self._attacks: list = []
        self._defences: list = []
        self._truth: dict = {}
        self._privacy_scale = 1.0
        self._rng_devices: list = []

    def setup(
        self,
        model_graph,
        partitions,
        optimizer_cfg: dict,
        criterion_cfg: dict,
        mixed_precision: bool = False,
        n_micro_batches: int = 1,
        run_config=None,
        callbacks: list = None,
        model_name: str = "unknown",
        strategy_name: str = "unknown",
        cut_hooks=None,
        algorithm: "Algorithm | None" = None,
        eval_data=None,
        eval_every: int = 1,
        privacy=None,
    ) -> None:
        """``cut_hooks``: ``ts.slice(cut_hooks=)`` (see ``torchslicer.sl.hooks``); the
        config's ``sl`` section adds hooks, the setting and the link model.
        ``eval_data``: a test loader, evaluated every ``eval_every`` epochs (``evaluate``).
        ``privacy``: ``ts.privacy`` attacks and defences, added to the config's ``privacy``."""
        self._eval_loader = eval_data
        self._eval_every = int(eval_every or 0) if eval_data is not None else 0
        layers = model_graph.get_layers()
        n = len(partitions)
        self.split_layers = []
        self._callbacks   = list(callbacks or [])

        partitions_layers = [[layers[j] for j in p.layer_indices] for p in partitions]
        # A tied parameter is one shared tensor here, so its .grad sums every use;
        # only its owner (lowest partition, which steps last) may update it.
        self.tied_groups = find_tied_parameters(partitions_layers)
        for i, partition in enumerate(partitions):
            is_last = (i == n - 1)
            partition_layers = partitions_layers[i]
            preds = partition.predecessors if partition.predecessors else None
            sl = SplitLayer(partition_layers, is_last=is_last, predecessors=preds,
                            mixed_precision=mixed_precision)
            sl.train()
            params = optimizer_parameters(sl, exclude=non_owned_names(i, self.tied_groups))
            opt = getattr(optim, optimizer_cfg["name"])(params, **optimizer_cfg["params"]) if params else None
            sl.set_optimizer(opt)
            self.split_layers.append(sl)

        self.criterion = getattr(nn, criterion_cfg["name"])(**criterion_cfg["params"])
        self._n_micro  = max(1, n_micro_batches)

        # Split-learning setting: where each slice would run, hooks on the cuts, metrics
        cfg = run_config
        sl_cfg = cfg.sl if cfg else None
        self._placement = Placement(resolve_setting(sl_cfg.setting if sl_cfg else "", "local"), n)
        self._hooks = resolve_cut_hooks(cut_hooks, n, sl_cfg.cut_hooks if sl_cfg else None)
        self._meter = CommMeter(n)
        self._link = ((sl_cfg.link_bandwidth_mbps, sl_cfg.link_latency_ms) if sl_cfg else (0.0, 0.0))
        self._client_states = (sl_cfg.client_states if sl_cfg else "") or "cpu"
        self._client_store = self._server_store = None
        self._profiles = list(sl_cfg.client_profiles or []) if sl_cfg else []
        self._profile_seed = int(sl_cfg.profile_seed) if sl_cfg else 0
        self._tiers = None

        # Initialise profilers (one per partition) and RunLogger
        if cfg:
            configure_process_logging(cfg.logging.level)
        verbosity = cfg.profile.verbosity if cfg else 0
        memory    = cfg.profile.memory    if cfg else False

        # For local execution the "device" is whatever the split_layers are on
        device = next(iter(self.split_layers[0].parameters()), torch.tensor(0)).device \
            if self.split_layers else None

        self._profilers = [
            WorkerProfiler(verbosity=verbosity, memory=memory, device=device)
            for _ in range(n)
        ]
        self._sync = ((lambda: torch.cuda.synchronize(device))
                      if device is not None and device.type == "cuda" else None)

        if cfg and cfg.logging.enabled:
            run_dir = os.path.join(cfg.logging.dir, cfg.run_id)
            self._run_logger = RunLogger(run_id=cfg.run_id, run_dir=run_dir)
            layer_names = [type(l).__name__ for l in layers]
            self._run_logger.record_executor("local")
            self._run_logger.record_config(cfg)
            self._run_logger.record_model(model_name, layer_names,
                                          node_names=model_graph.names(), expand=model_graph.expand)
            self._run_logger.record_split(partitions, layer_names, strategy_name)

        # Algorithm — resolve from registry if a name was given via config
        if algorithm is None and cfg and getattr(cfg.sl, "algorithm", ""):
            from ..algorithm.registry import get as _get_algo
            algorithm = _get_algo(cfg.sl.algorithm)
        self._algorithm = algorithm if algorithm is not None else DefaultAlgorithm()
        self._algorithm.configure(len(partitions), self.criterion, self._placement)

        self._attacks, self._defences = resolve_privacy(privacy, cfg.privacy if cfg else None, n)
        self._rng_devices = [device] if device is not None and device.type == "cuda" else []
        for a in self._attacks:
            if a.needs_query or a.needs_client:
                self._check_query(a)
            if a.needs_query:
                a.bind(lambda x, cut=a.cut: self._query(cut, x))
            if a.needs_client:
                a.bind_client(lambda fresh=False, cut=a.cut: _ClientCopy(self.split_layers[:cut + 1], fresh))
            if hasattr(a, "set_aux"):
                a.set_aux(eval_data)
            if hasattr(a, "bind_hook"):
                a.bind_hook(self._hooks.get(a.cut))

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb.on_train_begin(
                    run_id=cfg.run_id if cfg else "local",
                    config={},
                )
            except Exception as e:
                _LOG.warning("callback on_train_begin failed: %s", e)

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False,
                    total_epochs: int = 0) -> dict:
        self._round = epoch
        total_loss  = 0.0
        n_batches   = 0
        n_total     = len(data_loader)
        session     = data_loader if isinstance(data_loader, ClientSession) else None

        # Per-client server state tracking (SplitFed V1)
        needs_server_states = session is not None and self._algorithm.needs_per_client_server_state()
        server_adapter = None
        server_states: dict = {}
        prev_client_for_server: "int | None" = None
        initial_server_state = None
        if needs_server_states:
            client_set = set(self._placement.client_slices)
            server_idxs = [i for i in range(len(self.split_layers)) if i not in client_set]
            if server_idxs:
                layers = {i: self.split_layers[i] for i in server_idxs}
                if self._server_store is None:
                    self._server_store = resolve_client_store(
                        self._client_states, layers, session.n_clients, what="per-client server")
                server_adapter = SplitLayerClientState(layers, store=self._server_store)
                initial_server_state = server_adapter.get_state()

        if session is not None:
            clients = self._placement.client_slices
            if not clients:
                raise ValueError("a multi-client session needs client slices: set sl.setting "
                                 "to vanilla or u_shaped (pipeline hosts none on the data owner)")
            check_session_ties(self.tied_groups, self._placement)
            layers = {i: self.split_layers[i] for i in clients}
            if self._client_store is None:      # once: later rounds see the states' own memory
                self._client_store = resolve_client_store(self._client_states, layers, session.n_clients)
            adapter = SplitLayerClientState(layers, store=self._client_store)
            selected = select_clients(self._algorithm, session.n_clients, epoch)
            batches = session.batches(adapter, epoch, clients=selected)
        else:
            batches = ((None, b) for b in data_loader)
        self._meter.reset()
        for a in self._attacks:
            a.reset()

        for cb in self._callbacks:
            try:
                cb.on_epoch_begin(epoch)
            except Exception as e:
                _LOG.warning("callback on_epoch_begin failed: %s", e)

        epoch_t0 = time.perf_counter()

        with tracer.span("torchslicer.epoch", epoch=epoch) as epoch_span:
            for client, (inputs, labels) in batches:
                # V1: swap per-client server state on client switch
                if needs_server_states and server_adapter is not None and client != prev_client_for_server:
                    if prev_client_for_server is not None:
                        server_states[prev_client_for_server] = server_adapter.get_state()
                    if client in server_states:
                        server_adapter.set_state(server_states[client])
                    elif initial_server_state is not None:
                        server_adapter.set_state(copy.deepcopy(initial_server_state))
                    prev_client_for_server = client

                batch_id = epoch * n_total + n_batches
                if self._profiles:
                    self._meter.client = client if client is not None else 0
                _shape = (str(tuple(inputs.shape)) if isinstance(inputs, torch.Tensor)
                          else "multimodal")

                with tracer.span(
                    "torchslicer.batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    batch_index=n_batches,
                    input_shape=_shape,
                ) as batch_span:

                    for p in self._profilers:
                        p.begin_batch(batch_id)

                    if self._n_micro > 1:
                        loss_val = self._gpipe_batch(inputs, labels, batch_id)
                    else:
                        loss_val = self._standard_batch(inputs, labels, batch_id)

                    for p in self._profilers:
                        p.end_batch()

                    total_loss += loss_val
                    n_batches  += 1
                    if batch_span:
                        batch_span.set_attribute("loss", loss_val)

                if self._run_logger:
                    extra = {"client": client} if client is not None else {}
                    self._run_logger.log(
                        step=batch_id, epoch=epoch, batch=n_batches,
                        loss=round(loss_val, 6), phase="batch", **extra,
                    )

                if verbose:
                    ep_suffix = f"/{total_epochs}" if total_epochs else ""
                    _LOG.info(
                        "epoch progress epoch=%s%s batch=%s/%s loss=%.4f",
                        epoch,
                        ep_suffix,
                        n_batches,
                        n_total,
                        loss_val,
                    )

        epoch_duration_s = time.perf_counter() - epoch_t0
        avg = total_loss / n_batches if n_batches > 0 else 0.0
        if session is not None:
            session.finish(adapter)
            # Save last client's server state for V1
            if needs_server_states and server_adapter is not None and prev_client_for_server is not None:
                server_states[prev_client_for_server] = server_adapter.get_state()
        if verbose:
            suffix = f"/{total_epochs}" if total_epochs else ""
            _LOG.info("epoch complete epoch=%s%s avg_loss=%.4f", epoch, suffix, avg)

        # Build epoch metrics, let callbacks augment
        epoch_metrics = {
            "step":       epoch,
            "epoch":      epoch,
            "loss":       round(avg, 6),
            "duration_s": round(epoch_duration_s, 3),
            "phase":      "epoch",
            **self._meter.summary(self._placement, *self._link),
            **self._epoch_extras(epoch_duration_s),
        }
        for a in self._attacks:
            epoch_metrics.update({f"leak_{a.key}_{k}": v for k, v in a.metrics().items()})
        if self._profiles:
            n_clients = session.n_clients if session is not None else 1
            if self._tiers is None:
                self._tiers = assign_profiles(self._profiles, n_clients, self._profile_seed)
                epoch_metrics["client_tiers"] = list(self._tiers)
            parallel = session is not None and session.mode == "parallel"
            epoch_metrics.update(client_times(self._meter, self._placement, self._profiles,
                                              self._tiers, parallel))

        # Algorithm round-end: aggregation and per-client state updates
        if session is not None:
            if len(selected) < session.n_clients:
                epoch_metrics["clients"] = selected
            agg_server = algorithm_round_end(session, self._algorithm, adapter, selected,
                                             epoch_metrics, epoch, server_states)
            if agg_server is not None and server_adapter is not None:
                server_adapter.set_state(agg_server)
                for k in list(server_states):
                    server_states[k] = copy.deepcopy(agg_server)

        if self._eval_every and epoch % self._eval_every == 0:
            epoch_metrics.update(self._evaluate_round(session, adapter if session is not None else None))

        for cb in self._callbacks:
            try:
                result = cb.on_epoch_end(epoch, epoch_metrics)
                if isinstance(result, dict):
                    epoch_metrics = result
            except Exception as e:
                _LOG.warning("callback on_epoch_end failed: %s", e)

        if self._run_logger:
            self._run_logger.log(**epoch_metrics)

            # Per-partition profiling entries
            for i, profiler in enumerate(self._profilers):
                if not profiler.is_active():
                    break
                summary = profiler.epoch_summary(epoch)
                if summary.get("n_batches", 0) > 0:
                    self._run_logger.log(
                        step=epoch, epoch=epoch, partition=i,
                        phase="partition_epoch",
                        **{k: v for k, v in summary.items() if k != "epoch"},
                    )
                    # Verbosity=3: raw per-batch records
                    for rec in profiler.batch_records():
                        self._run_logger.log(
                            step=epoch, epoch=epoch, partition=i,
                            phase="partition_batch",
                            **rec,
                        )
                profiler.reset_epoch()

        return epoch_metrics

    def evaluate(self, loader) -> dict:
        """Test loss (the training criterion) and, for classification (``(N, C)`` outputs and
        ``(N,)`` labels, or per token: ``(N, C, T)`` and ``(N, T)``), accuracy of the current weights: forward only, no gradients,
        every slice in eval mode, the RNG state restored after. Cut hooks transform the activations as in training
        (``CutContext.direction == "eval"``; ``Recorder`` skips it); no bytes are counted."""
        device = next(self.split_layers[0].parameters()).device
        was_training = [sl.training for sl in self.split_layers]
        for sl in self.split_layers:
            sl.eval()
        total, correct, n, n_items, classify = 0.0, 0, 0, 0, True
        # iterating a DataLoader draws from the global RNG: keep training's stream intact
        rng_devices = [device] if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
                for inputs, labels in loader:
                    main, aux = _unpack_inputs(inputs)
                    x = main.to(device)
                    labels = labels.to(device)
                    aux = {k: v.to(device) for k, v in aux.items()}
                    for i, sl in enumerate(self.split_layers):
                        if i > 0:
                            x = self._eval_cut(i - 1, x)
                        x = sl(x, **aux) if (i == 0 and aux) else sl(x)
                        sl.pop_moe_aux_loss()
                    b = labels.shape[0]
                    total += self.criterion(x, labels).item() * b
                    n += b
                    # classes on dim 1: (N, C) vs (N,), or (N, C, T...) vs (N, T...) for tokens
                    if (x.dim() == labels.dim() + 1 and not labels.is_floating_point()
                            and x.shape[0] == b and x.shape[2:] == labels.shape[1:]):
                        correct += (x.argmax(1) == labels).sum().item()
                        n_items += labels.numel()
                    else:
                        classify = False
        finally:
            for sl, mode in zip(self.split_layers, was_training):
                sl.train(mode)
        if n == 0:
            return {}
        out = {"test_loss": round(total / n, 6), "test_samples": n}
        if classify:
            out["test_accuracy"] = round(correct / n_items, 6)
        return out

    def _eval_cut(self, cut: int, x):
        hook = self._hooks.get(cut)
        if hook is None:
            return x
        ctx = CutContext(cut, "eval", -1, len(self.split_layers))
        return hook.decode(_detach_payload(hook.encode(x, ctx)), ctx)

    def _evaluate_round(self, session, adapter) -> dict:
        """``evaluate`` for the epoch's history entry. In a session, every distinct client
        state is evaluated with the shared server slices (clients that share one state
        object, e.g. after ``fedavg``, once): the mean, plus ``*_clients`` lists."""
        if session is None:
            return self.evaluate(self._eval_loader)
        results, by_state = [], {}
        for state in session.states:
            state = state if state is not None else session._initial   # not trained yet
            if id(state) not in by_state:
                adapter.set_state(state)
                by_state[id(state)] = self.evaluate(self._eval_loader)
            results.append(by_state[id(state)])
        if session.current is not None and session.states[session.current] is not None:
            adapter.set_state(session.states[session.current])
        out = {}
        for key in ("test_loss", "test_accuracy"):
            vals = [r[key] for r in results if key in r]
            if vals:
                out[key] = round(sum(vals) / len(vals), 6)
                if len(by_state) > 1:
                    out[f"{key}_clients"] = vals
        out["test_samples"] = results[0].get("test_samples", 0)
        return out

    def _epoch_extras(self, duration_s: float) -> dict:
        """Executor-specific fields for the epoch's history entry."""
        return {}

    # ── cuts: hooks and byte counting ──────────────────────────────────────────

    def _clock(self) -> float:
        if self._sync is not None:
            self._sync()
        return time.perf_counter()

    def _cut_forward(self, cut: int, x, batch_id: int):
        """What slice ``cut + 1`` receives from slice ``cut`` (hooks applied, bytes counted)."""
        raw = tensor_nbytes(x)
        hook = self._hooks.get(cut)
        if hook is None:
            self._meter.add_fwd(cut, raw, raw)
        else:
            ctx = CutContext(cut, "forward", batch_id, len(self.split_layers))
            payload = _detach_payload(hook.encode(x, ctx))
            self._meter.add_fwd(cut, payload_nbytes(payload), raw)
            x = hook.decode(payload, ctx)
        if self._attacks:
            self._observe(cut, "forward", x, batch_id)
        return x

    def _cut_backward(self, cut: int, g, out, batch_id: int):
        """The gradient slice ``cut`` backpropagates through ``out``, from ``g`` sent by
        slice ``cut + 1``."""
        if g is None:
            return None
        if self._attacks:
            g = self._forge(cut, g, batch_id)
        raw = tensor_nbytes(g)
        hook = self._hooks.get(cut)
        if hook is None:
            self._meter.add_bwd(cut, raw, raw)
        else:
            ctx = CutContext(cut, "backward", batch_id, len(self.split_layers))
            payload = _detach_payload(hook.encode_grad(g, ctx))
            self._meter.add_bwd(cut, payload_nbytes(payload), raw)
            g = hook.decode_grad(payload, ctx)
        if self._attacks:
            self._observe(cut, "backward", g, batch_id)
        if self._defences:
            g = self._defend(cut, g, out, batch_id)
        return g.to(out.dtype) if g.dtype != out.dtype else g

    # ── privacy: attacks observe the receiving side, defences add a client term ──

    def _set_truth(self, inputs, labels, ids, scale: float = 1.0) -> None:
        """The (micro-)batches' raw inputs and labels by batch id, for attack scoring and
        the defences."""
        if self._attacks or self._defences:
            self._truth = dict(zip(ids, zip(inputs, labels)))
            self._privacy_scale = scale

    def _obs(self, cut, direction, t, batch_id) -> Observation:
        inputs, labels = self._truth.get(batch_id, (None, None))
        return Observation(t.detach(), CutContext(cut, direction, batch_id, len(self.split_layers)),
                           inputs, labels)

    def _observe(self, cut: int, direction: str, t, batch_id: int) -> None:
        attacks = [a for a in self._attacks if a.cut == cut and a.direction == direction]
        if not attacks:
            return
        obs = self._obs(cut, direction, t, batch_id)
        # attacks draw their own random numbers (decoder init, sampling): training's stay intact
        with torch.random.fork_rng(devices=self._rng_devices):
            for a in attacks:
                a.observe(obs)

    def _forge(self, cut: int, g, batch_id: int):
        """A malicious server's gradient replaces the honest one before it is sent."""
        for a in self._attacks:
            if a.forges and a.cut == cut:
                with torch.random.fork_rng(devices=self._rng_devices):
                    forged = a.forge(self._obs(cut, "backward", g, batch_id))
                if forged is not None:
                    if forged.shape != g.shape:
                        raise ValueError(f"{type(a).__name__}.forge returned shape {tuple(forged.shape)} "
                                         f"for a cut gradient of shape {tuple(g.shape)}")
                    g = (self._privacy_scale * forged).to(device=g.device, dtype=g.dtype)
        return g

    def _defend(self, cut: int, g, out, batch_id: int):
        obs = None
        for d in self._defences:
            if d.cut != cut:
                continue
            obs = obs or self._obs(cut, "backward", g, batch_id)
            extra = d.grad_term(out, obs)
            if extra is not None:
                g = g + (self._privacy_scale * extra).to(g.dtype)
        return g

    def _check_query(self, attack) -> None:
        for sl in self.split_layers[:attack.cut + 1]:
            if any(hasattr(layer, "pop_aux_loss") for layer in sl.layers):
                raise ValueError(f"{type(attack).__name__} runs the client slices, which hold MoE "
                                 "blocks (their auxiliary losses would mix with training's)")

    def _query(self, cut: int, x):
        """Black-box query access for an attack: what slice ``cut + 1`` would receive for
        inputs ``x`` (eval mode, no gradients, hooks applied as in ``evaluate``). Leaves the
        slices' training state (mode, saved cut inputs) as it was."""
        layers = self.split_layers[:cut + 1]
        saved = [(sl.x, sl.training) for sl in layers]
        try:
            for sl in layers:
                sl.eval()
            with torch.no_grad():
                for i, sl in enumerate(layers):
                    if i > 0:
                        x = self._eval_cut(i - 1, x)
                    x = sl(x)
                return self._eval_cut(cut, x)
        finally:
            for sl, (x_saved, mode) in zip(layers, saved):
                sl.x = x_saved
                sl.train(mode)

    def _count_io(self, main, aux: dict, labels, msgs: int = 1) -> None:
        """Raw inputs / labels that would leave the data owner (``sl.setting``)."""
        if self._placement.inputs_leave:
            self._meter.add_input(tensor_nbytes(main) + sum(tensor_nbytes(v) for v in aux.values()), msgs)
        if self._placement.labels_leave:
            self._meter.add_label(tensor_nbytes(labels), msgs)
        self._meter.batches += 1

    # ── standard (non-GPipe) batch ─────────────────────────────────────────────

    def _standard_batch(self, inputs, labels, batch_id: int) -> float:
        device = next(self.split_layers[0].parameters()).device
        main, aux = _unpack_inputs(inputs)
        main   = main.to(device)
        labels = labels.to(device)
        aux    = {k: v.to(device) for k, v in aux.items()}
        self._count_io(main, aux, labels)
        self._set_truth([main], [labels], [batch_id])

        ctx    = AlgorithmContext(round=self._round, batch_id=batch_id,
                                  n_slices=len(self.split_layers))
        policy = self._algorithm.backward_policy(self._round)

        if policy == BackwardPolicy.LOCAL_LOSS:
            return self._local_loss_batch(main, aux, labels, batch_id, ctx)

        # ── FULL: standard forward → loss → backward ──────────────────────────
        outputs = []
        x = main
        for i, sl in enumerate(self.split_layers):
            if i > 0:
                x = self._cut_forward(i - 1, x, batch_id)
            with tracer.span(
                "torchslicer.partition.forward",
                partition=i,
                batch_id=batch_id,
                input_shape=str(tuple(x.shape)),
            ) as fwd_span:
                t0 = self._clock()
                with self._profilers[i].phase("forward"):
                    x = sl(x, **aux) if (i == 0 and aux) else sl(x)
                self._meter.add_compute(i, self._clock() - t0)
                if fwd_span:
                    fwd_span.set_attribute("output_shape", str(tuple(x.shape)))
            outputs.append(x)

        t0 = self._clock()
        custom = self._algorithm.loss(outputs[-1], labels, ctx)
        loss   = custom if custom is not None else self.criterion(outputs[-1], labels)

        # backward — pop MoE aux loss BEFORE backward (same graph, would fail after)
        with tracer.span("torchslicer.partition.backward",
                         partition=len(self.split_layers) - 1, batch_id=batch_id):
            with self._profilers[-1].phase("backward"):
                moe = self.split_layers[-1].pop_moe_aux_loss()
                total_loss = loss + moe if moe is not None else loss
                grad = self.split_layers[-1].backward(loss=total_loss)
            with self._profilers[-1].phase("optimizer"):
                self.split_layers[-1].optimize()
        self._meter.add_compute(len(self.split_layers) - 1, self._clock() - t0)

        for i in range(len(self.split_layers) - 2, -1, -1):
            grad = self._cut_backward(i, grad, outputs[i], batch_id)
            t0 = self._clock()
            with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id):
                with self._profilers[i].phase("backward"):
                    moe = self.split_layers[i].pop_moe_aux_loss()
                    if moe is not None:
                        # retain graph so moe.backward() can reuse the same nodes
                        outputs[i].backward(gradient=grad, retain_graph=True)
                        moe.backward()
                        sl_x = self.split_layers[i].x
                        grad = sl_x.grad if sl_x is not None else None
                    else:
                        grad = self.split_layers[i].backward(prev_g=grad, out=outputs[i])
                with self._profilers[i].phase("optimizer"):
                    self.split_layers[i].optimize()
            self._meter.add_compute(i, self._clock() - t0)

        return loss.item()

    def _local_loss_batch(self, main, aux, labels, batch_id: int,
                          ctx: AlgorithmContext) -> float:
        """LOCAL_LOSS: forward + backward on client slices only; no server forward."""
        cut_idx = self._algorithm.client_cut()
        if cut_idx is None:
            client_slices = sorted(self._placement.client_slices)
            cut_idx = client_slices[-1] if client_slices else 0

        outputs = []
        x = main
        for i in range(cut_idx + 1):
            if i > 0:
                x = self._cut_forward(i - 1, x, batch_id)
            with tracer.span("torchslicer.partition.forward", partition=i,
                             batch_id=batch_id, input_shape=str(tuple(x.shape))):
                t0 = self._clock()
                with self._profilers[i].phase("forward"):
                    x = self.split_layers[i](x, **aux) if (i == 0 and aux) else self.split_layers[i](x)
                self._meter.add_compute(i, self._clock() - t0)
            outputs.append(x)

        loss = self._algorithm.client_loss(outputs[-1], labels, ctx)

        # Backward through client slices: the cut slice is NOT is_last, so we
        # call loss.backward() directly and read x.grad from the cut slice.
        t0 = self._clock()
        with tracer.span("torchslicer.partition.backward", partition=cut_idx, batch_id=batch_id):
            with self._profilers[cut_idx].phase("backward"):
                moe = self.split_layers[cut_idx].pop_moe_aux_loss()
                total = loss + moe if moe is not None else loss
                total.backward()
                grad = self.split_layers[cut_idx].x.grad if self.split_layers[cut_idx].x is not None else None
            with self._profilers[cut_idx].phase("optimizer"):
                self.split_layers[cut_idx].optimize()
        self._algorithm.step_aux(ctx)
        self._meter.add_compute(cut_idx, self._clock() - t0)

        for i in range(cut_idx - 1, -1, -1):
            grad = self._cut_backward(i, grad, outputs[i], batch_id)
            t0 = self._clock()
            with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id):
                with self._profilers[i].phase("backward"):
                    moe = self.split_layers[i].pop_moe_aux_loss()
                    if moe is not None:
                        outputs[i].backward(gradient=grad, retain_graph=True)
                        moe.backward()
                        sl_x = self.split_layers[i].x
                        grad = sl_x.grad if sl_x is not None else None
                    else:
                        grad = self.split_layers[i].backward(prev_g=grad, out=outputs[i])
                with self._profilers[i].phase("optimizer"):
                    self.split_layers[i].optimize()
            self._meter.add_compute(i, self._clock() - t0)

        return loss.item()

    # ── GPipe micro-batch batch ────────────────────────────────────────────────

    def _gpipe_batch(self, inputs, labels, batch_id: int) -> float:
        """
        GPipe-style micro-batch pipelining:
          1. Forward all M micro-batches, saving cut-point tensors (x_refs).
          2. Backward all M micro-batches, accumulating gradients (no zero_grad between).
          3. Single optimizer step across all split layers.

        Each micro-batch loss is scaled by 1/M so accumulated gradients match
        the magnitude of a full-batch gradient.
        """
        device = next(self.split_layers[0].parameters()).device
        main, aux = _unpack_inputs(inputs)
        main   = main.to(device)
        labels = labels.to(device)
        aux    = {k: v.to(device) for k, v in aux.items()}

        M    = self._n_micro
        n_sl = len(self.split_layers)
        micro_mains  = main.chunk(M)
        micro_labels = labels.chunk(M)
        self._count_io(main, aux, labels, msgs=len(micro_mains))
        self._set_truth(micro_mains, micro_labels, [batch_id * M + m for m in range(len(micro_mains))],
                        scale=1.0 / M)
        # Chunk aux tensors that share the batch dimension; broadcast scalars/constants
        micro_aux = [
            {k: v.chunk(M)[m] if v.dim() > 0 and v.shape[0] == main.shape[0] else v
             for k, v in aux.items()}
            for m in range(M)
        ]

        # Step 1: forward all micro-batches; collect per-micro MoE aux losses immediately
        # (must pop before backward — they share the same computation graph nodes)
        micro_outs: list[list[torch.Tensor]] = []
        x_refs:     list[list[torch.Tensor]] = []
        micro_moe:  list[list] = []  # micro_moe[m][i] = Optional[Tensor]

        for m in range(M):
            outs: list[torch.Tensor] = []
            refs: list[torch.Tensor] = []
            x = micro_mains[m]
            for i, sl in enumerate(self.split_layers):
                if i > 0:
                    x = self._cut_forward(i - 1, x, batch_id * M + m)
                with tracer.span(
                    "torchslicer.partition.forward",
                    partition=i,
                    batch_id=batch_id * M + m,
                    input_shape=str(tuple(x.shape)),
                ):
                    t0 = self._clock()
                    with self._profilers[i].phase("forward"):
                        x = sl(x, **micro_aux[m]) if (i == 0 and micro_aux[m]) else sl(x)
                    self._meter.add_compute(i, self._clock() - t0)
                refs.append(sl.x)
                outs.append(x)
            micro_outs.append(outs)
            x_refs.append(refs)
            # Pop MoE aux per micro-batch right after forward so graph is still alive
            micro_moe.append([sl.pop_moe_aux_loss() for sl in self.split_layers])

        # Step 2: backward all micro-batches, accumulate gradients
        total_loss = 0.0
        for m in range(M):
            t0 = self._clock()
            loss_m = self.criterion(micro_outs[m][-1], micro_labels[m])
            total_loss += loss_m.item()
            scaled = loss_m / M

            # Last partition: add MoE aux (scaled by 1/M) to activation loss
            moe_last = micro_moe[m][-1]
            if moe_last is not None:
                scaled = scaled + moe_last / M

            with tracer.span("torchslicer.partition.backward",
                             partition=n_sl - 1, batch_id=batch_id * M + m):
                with self._profilers[-1].phase("backward"):
                    scaled.backward()
            self._meter.add_compute(n_sl - 1, self._clock() - t0)

            for i in range(n_sl - 2, -1, -1):
                g = self._cut_backward(i, x_refs[m][i + 1].grad, micro_outs[m][i], batch_id * M + m)
                t0 = self._clock()
                with tracer.span("torchslicer.partition.backward",
                                 partition=i, batch_id=batch_id * M + m):
                    with self._profilers[i].phase("backward"):
                        moe_i = micro_moe[m][i]
                        if moe_i is not None:
                            micro_outs[m][i].backward(g, retain_graph=True)
                            (moe_i / M).backward()
                        else:
                            micro_outs[m][i].backward(g)
                self._meter.add_compute(i, self._clock() - t0)

        # Step 3: optimizer
        for i, sl in enumerate(self.split_layers):
            t0 = self._clock()
            with self._profilers[i].phase("optimizer"):
                sl.optimize()
            self._meter.add_compute(i, self._clock() - t0)

        return total_loss / M

    def teardown(self) -> None:
        log_history = self._run_logger.log_history if self._run_logger else []
        for cb in self._callbacks:
            try:
                cb.on_train_end(log_history)
            except Exception as e:
                _LOG.warning("callback on_train_end failed: %s", e)

        if self._run_logger:
            self._run_logger.flush()

        self.split_layers = []
        self.criterion    = None
        self._profilers   = []
