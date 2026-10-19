# Flower Strategies as SplitFed Aggregation

SplitFed averages the clients' client-side models after every round. That step is
federated learning's aggregation, and Flower (https://flower.ai) has many strategies for
it: FedAvg, FedMedian, FedTrimmedAvg, Krum, FedAvgM, FedAdam, FedYogi, and more.
`torchslicer.interop.flower.FlowerAggregation` lets any of them aggregate the client side
of a split model.

```bash
pip install "torchslicer[flower]"      # flwr >= 1.40
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

What this is not (yet): TorchSlicer clients run by Flower's own engine (`flwr run`, the
simulation engine, or a SuperLink deployment). There, each Flower client is its own
process, so the server slices would have to be reachable over the network, as
TorchSlicer's distributed workers are. That integration is planned.
