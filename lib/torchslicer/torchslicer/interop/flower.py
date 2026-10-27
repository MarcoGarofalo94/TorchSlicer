"""Flower (https://flower.ai) strategies as SplitFed aggregation.

``FlowerAggregation(strategy)`` is a ``ts.Algorithm`` that, at each round's end, hands the
participating clients' client-slice weights to a Flower ``Strategy.aggregate_fit``, as Flower
results with each client's sample count, and gives every client the aggregate. So any Flower
strategy (``FedAvg``, ``FedMedian``, ``FedTrimmedAvg``, ``Krum``, ``FedAvgM``, ``FedAdam``, ...)
aggregates the client side of a split model. ``server_strategy`` does the same for per-client
server copies (SplitFed V1). Requires ``pip install flwr`` (tested with 1.40).

Weights cross as one NumPy array per state-dict entry, ordered by slice and then by key;
integer buffers (e.g. BatchNorm's ``num_batches_tracked``) are cast back after aggregation.
Optimizer states start fresh after a round, as with ``ts.SplitFedAlgorithm``. Strategies
built on ``FedOpt`` (FedAdam, FedYogi, FedAdagrad) need ``initial_parameters``: use
:func:`state_to_parameters` on the model's initial client slices.
"""

import torch

from ..algorithm.base import Algorithm


def _flwr():
    try:
        from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    except ImportError as e:
        raise ImportError("torchslicer.interop.flower needs Flower: pip install flwr") from e
    return Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays


def _keys(state: dict) -> list:
    return [(i, k) for i in sorted(state) for k in state[i]["model"]]


def state_to_parameters(state: dict):
    """A client (or server) state ``{slice: {"model": state_dict, ...}}`` as Flower
    ``Parameters``; or pass ``{slice: module}`` for modules."""
    _, _, _, to_params, _ = _flwr()
    state = {i: (s if isinstance(s, dict) else {"model": s.state_dict()}) for i, s in state.items()}
    return to_params([state[i]["model"][k].detach().cpu().numpy() for i, k in _keys(state)])


class _Proxy:
    """Stands in for Flower's ``ClientProxy`` in the results list (strategies only read
    its ``cid``)."""

    def __init__(self, cid: str):
        self.cid = cid


class FlowerAggregation(Algorithm):
    """Aggregate the clients' client slices (and, with ``server_strategy``, per-client
    server copies) with Flower strategies at the end of every round."""

    def __init__(self, strategy, server_strategy=None):
        _flwr()
        self.strategy, self.server_strategy = strategy, server_strategy
        self._samples = None
        self.last_metrics = {}

    def needs_per_client_server_state(self) -> bool:
        return self.server_strategy is not None

    def on_round_end(self, client_states, metrics, round):
        self._samples = metrics.get("client_samples") or [1] * len(client_states)
        return None

    def aggregate(self, client_states, server_states, round):
        client = self._aggregate(self.strategy, client_states, round)
        server = self._aggregate(self.server_strategy, server_states, round) if self.server_strategy else None
        return client, server

    def _aggregate(self, strategy, states, round):
        states = [s for s in states if s is not None]
        if not states:
            return None
        Code, FitRes, Status, to_params, from_params = _flwr()
        keys = _keys(states[0])
        samples = self._samples if self._samples and len(self._samples) == len(states) else [1] * len(states)
        results = [(_Proxy(str(c)), FitRes(Status(Code.OK, ""), state_to_parameters(s), int(n), {}))
                   for c, (s, n) in enumerate(zip(states, samples))]
        params, self.last_metrics = strategy.aggregate_fit(round, results, [])
        if params is None:
            return None
        arrays = from_params(params)
        out = {}
        for (i, k), a in zip(keys, arrays):
            ref = states[0][i]["model"][k]
            out.setdefault(i, {"model": {}, "optimizer": None})["model"][k] = (
                torch.from_numpy(a.copy()).to(device=ref.device, dtype=ref.dtype))
        return out


# ── TorchSlicer clients run by Flower's engine ──────────────────────────────────

def initial_client_arrays(model_fn, cuts: list, multi_tensor: bool = False):
    """The client slice of ``model_fn()`` as a Flower ``ArrayRecord``: the
    ``initial_arrays`` of a strategy that aggregates :func:`split_client_app` clients."""
    from flwr.app import ArrayRecord
    from ..sl.remote import _slices
    return ArrayRecord(_slices(model_fn(), cuts, multi_tensor=multi_tensor)[0].state_dict())


def split_client_app(model_fn, data_fn, server_address: str, cuts: list, optimizer: dict,
                     local_epochs: int = 1, cut_hook_fn=None, device: str = "cpu",
                     multi_tensor: bool = False):
    """A Flower ``ClientApp`` whose node trains the client slice of ``model_fn()`` against a
    :class:`torchslicer.SplitServer` at ``server_address`` (which holds the other slices).

    ``data_fn(partition_id, num_partitions) -> (train_loader, eval_loader)`` gives the node
    its data (the ids come from the node config, as in Flower's simulation engine).
    ``train`` loads the strategy's arrays into the client slice, trains ``local-epochs``
    (train config, default ``local_epochs``) with a fresh optimizer and returns the slice with
    ``num-examples``; ``evaluate`` returns ``eval_loss``/``eval_acc``. With ``FedAvg`` on the
    server side this is SplitFed V2: one server copy, the client slices averaged each round.
    ``cut_hook_fn()`` builds this node's cut hook; ``multi_tensor`` as in ``ts.SplitClient``.
    """
    from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from ..sl.remote import SplitClient

    app, cache = ClientApp(), {}

    def _client(context):
        if "client" not in cache:
            cache["client"] = SplitClient(model_fn(), cuts, server_address, optimizer, device=device,
                                          cut_hook=cut_hook_fn() if cut_hook_fn else None,
                                          multi_tensor=multi_tensor)
        pid = int(context.node_config.get("partition-id", 0))
        n = int(context.node_config.get("num-partitions", 1))
        if cache.get("pid") != pid:
            cache["pid"], cache["data"] = pid, data_fn(pid, n)
        return cache["client"], cache["data"]

    @app.train()
    def train(msg, context):
        client, (train_loader, _) = _client(context)
        client.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        cfg = msg.content["config"] if "config" in msg.content else {}
        out = client.fit(train_loader, epochs=int(cfg.get("local-epochs", local_epochs)))
        content = RecordDict({"arrays": ArrayRecord(client.state_dict()),
                              "metrics": MetricRecord({"train_loss": out["loss"],
                                                       "num-examples": out["num_examples"]})})
        return Message(content=content, reply_to=msg)

    @app.evaluate()
    def evaluate(msg, context):
        client, (_, eval_loader) = _client(context)
        client.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        out = client.evaluate(eval_loader)
        content = RecordDict({"metrics": MetricRecord({"eval_loss": out["loss"], "eval_acc": out["accuracy"],
                                                       "num-examples": out["num_examples"]})})
        return Message(content=content, reply_to=msg)

    return app
