# Flower Strategies as SplitFed Aggregation

SplitFed averages the clients' client-side models after every round. That step is
federated learning's aggregation, and Flower (https://flower.ai) has many strategies for
it: FedAvg, FedMedian, FedTrimmedAvg, Krum, FedAvgM, FedAdam, FedYogi, and more.
`torchslicer.interop.flower.FlowerAggregation` lets any of them aggregate the client side
of a split model.

```bash
pip install "torchslicer[flower]"      # flwr >= 1.40, Python 3.11+
```

```python
import torchslicer as ts
from flwr.server.strategy import FedAvg, FedMedian
from torchslicer.interop.flower import FlowerAggregation

session = ts.ClientSession(client_loaders, mode="parallel")
sliced = ts.slice(model, cuts=["2"])
sliced.train(session, run_config=cfg, algorithm=FlowerAggregation(FedMedian()))

# SplitFed V1: per-client server copies, aggregated by a second strategy
sliced.train(session, run_config=cfg, algorithm=FlowerAggregation(FedAvg(), server_strategy=FedAvg()))
```

- At each round's end the participating clients' client-slice weights go to
  `strategy.aggregate_fit` as Flower results, with each client's sample count as
  `num_examples`. Every client then gets the aggregate, and its optimizer state starts fresh.
- `FlowerAggregation(FedAvg())` gives exactly `ts.SplitFedAlgorithm()`, and with a server
  strategy exactly `ts.SplitFedV1Algorithm()`: both weight the clients by n_k / n, as in
  SplitFed's Algorithm 1. `tests/test_flower.py` checks this on clients of unequal size.
- Strategies built on `FedOpt` (FedAdam, FedYogi, FedAdagrad) need the starting point:
  `FedAdam(initial_parameters=state_to_parameters({0: client_slice}))`.
- Weights cross as one NumPy array per state-dict entry, ordered by slice and then key.
  Integer buffers are cast back after aggregation.

## TorchSlicer clients in Flower's engine

Flower can also run the clients itself: `flwr run`, the simulation engine, or a
SuperLink deployment. Each Flower node holds the client slice and its own data. The
server slices live in one `ts.SplitServer` that every node reaches over gRPC. A Flower
strategy aggregates the client slices, so with `FedAvg` this is SplitFed V2: one server
copy, the client slices averaged every round.

```python
import torchslicer as ts
from flwr.app import ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torchslicer.interop.flower import initial_client_arrays, split_client_app

OPT = {"name": "SGD", "params": {"lr": 0.1, "momentum": 0.9}}
CUTS = ["2"]

# the server slices, one process (here, or on another machine)
server = ts.SplitServer(build_model(), CUTS, OPT, {"name": "CrossEntropyLoss", "params": {}},
                        port=50070).start()

# Flower nodes: data_fn(partition_id, num_partitions) -> (train_loader, eval_loader)
client_app = split_client_app(build_model, data_fn, "server-host:50070", CUTS, OPT)

server_app = ServerApp()

@server_app.main()
def main(grid: Grid, context: Context):
    FedAvg().start(grid=grid, initial_arrays=initial_client_arrays(build_model, CUTS),
                   num_rounds=10, train_config=ConfigRecord({"local-epochs": 1}))
```

Run the two apps with `flwr run` or `flwr.simulation.run_simulation(server_app=...,
client_app=..., num_supernodes=N)`, which needs `pip install "flwr[simulation]"`.

- **Per batch:** each node sends the cut activation and the labels to the server, and
  gets back the gradient of the activation. The server steps its own optimizer under a
  lock, so concurrent nodes are served one batch at a time.
- **Per round:** `train` loads the strategy's arrays into the client slice and trains
  `local-epochs` with a fresh optimizer. It returns the slice and `num-examples`.
  `evaluate` returns `eval_loss` and `eval_acc`.
- **Cut hooks:** `split_client_app(cut_hook_fn=...)` gives each node its own hook, and
  `ts.SplitServer(cut_hook=...)` gives the server its own. Compression hooks such as
  `ts.CircularConv` or `ts.Quantize` then work over the network.
- **Wire format:** tensors cross as packed buffers, never pickled. Build both sides'
  model with the same code.
- **Equivalence:** `tests/test_flower.py` runs Flower's simulation engine. With one node,
  the weights equal `ts.SplitFedAlgorithm` on one client. `tests/test_split_server.py`
  checks one client through `SplitServer` against local vanilla SL.

`ts.SplitServer` and `ts.SplitClient` don't depend on Flower. Any other framework, or a
plain script, can drive a client: `SplitClient(...).fit(loader)`.
