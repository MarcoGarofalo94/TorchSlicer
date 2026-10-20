# Split-learning settings: roles, cut hooks, metrics, multiple clients

TorchSlicer slices a model into a chain of slices. Split learning (SL) adds three
questions on top of the chain:

1. **Who runs which slice?** The *data owner* (the client) must keep its raw data, and
   in some settings its labels too.
2. **What crosses each cut?** Activations go forward and gradients come back. You may
   want to compress them, add noise, or watch them as an attacker would.
3. **What does it cost?** That means bytes per cut, compute on the client, and time on
   a given link.

A fourth question follows: **what if there are many clients?** This page covers all
four, with the clients simulated in one process (section 4) or as their own nodes
(section 5). Every code block below runs as written (`tests/test_docs.py` executes them).

## 1. Settings: what the data owner hosts

The data owner is the process that holds the data. That is the process that calls
`data_fn` in `ts.run`, or the one that iterates the loader. Which process it is
depends on the topology:

- the P2P driver;
- the centralized coordinator;
- the single process of a local run.

The setting (`sl.setting`, env `SL_SETTING`) says which slices this process hosts
*in-process*. What it hosts never crosses the network:

| setting    | data owner hosts       | raw inputs | labels | usual name |
|------------|------------------------|------------|--------|------------|
| `pipeline` | nothing                | leave      | leave  | pipeline parallelism |
| `vanilla`  | slice 0                | stay       | leave  | split learning with label sharing |
| `u_shaped` | slice 0 and the last   | stay       | stay   | U-shaped split learning (no label sharing) |

The default is `pipeline` for the centralized topology, because its coordinator holds
no layers (this is the pre-M4 behaviour). It is `vanilla` everywhere else, since a P2P
driver always holds slice 0. `u_shaped` needs at least 3 slices: the client front, the
server, and the client tail (the head and the loss).

A setting never changes the math. Training is the same as unsplit PyTorch in every
setting and topology. What changes is where each slice runs, and so which tensors
cross the network. The metrics show it:

```python
import torch
import torch.nn as nn
import torchslicer as ts


def build_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
                         nn.Linear(32, 4))


def get_data():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, 8, generator=g), torch.randint(0, 4, (64,), generator=g)
    return [(x[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16)]


def config(setting="", topology="local", **sl):
    cfg = ts.RunConfig()
    cfg.training.epochs = 1
    cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.05}}
    cfg.discovery.n_workers = 3                  # slices
    cfg.deploy.topology = topology
    cfg.network.device = "cpu"
    cfg.logging.enabled = False
    cfg.sl.setting = setting
    for k, v in sl.items():
        setattr(cfg.sl, k, v)
    return cfg


for setting in ["pipeline", "vanilla", "u_shaped"]:
    h = ts.run(build_model, get_data, run_config=config(setting), cuts=["1", "3"], verbose=False)
    e = h[0]
    print(f"{setting:9} loss={e['loss']:.4f} client bytes={e['bytes_client']:6} "
          f"(inputs {e['bytes_input']}, labels {e['bytes_label']}, per cut {e['bytes_fwd']})")
```

`bytes_client` counts what crosses the data owner's boundary: the cuts between one of
its slices and a server slice, plus any raw inputs and labels that leave it.

In a distributed run the data owner hosts those slices itself:

- **P2P:** the driver hosts slice 0 on its own gRPC server, as before. With
  `u_shaped` it also hosts the last slice on `sl.tail_port`. The default tail port is
  the driver's port + 2, and the TCP listener is that port + `transport.tensor_port_offset`.
- **Centralized:** with `vanilla` or `u_shaped`, the coordinator hosts the same
  slices, so raw inputs no longer reach worker 0.

Either way, `discovery.n_workers` stays the number of slices, and there is one remote
worker per slice the data owner doesn't host. The launcher counts nodes the same way:
`u_shaped` on 4 slices is the driver plus 2 followers. On k8s, the driving node's
Service also exposes the tail port.

```python
h = ts.run(build_model, get_data, run_config=config("u_shaped", topology="p2p"),
           cuts=["1", "3"], verbose=False)      # driver + 1 follower, on loopback
assert h[0]["bytes_label"] == 0 and h[0]["bytes_input"] == 0
```

## 2. Cut hooks: change or watch what crosses a cut

Cut `k` is the boundary between slice `k` and slice `k + 1`. A `ts.CutHook` has four
methods, one for each side of the cut and each direction. Each method defaults to the
identity:

| method        | runs on          | does |
|---------------|------------------|------|
| `encode`      | slice `k`        | activation -> payload sent on the wire |
| `decode`      | slice `k + 1`    | payload -> activation it computes on |
| `encode_grad` | slice `k + 1`    | gradient -> payload |
| `decode_grad` | slice `k`        | payload -> gradient it backpropagates |

A payload is a tensor or a list of tensors. Slice `k + 1` takes its gradient with
respect to what it received, and slice `k` backpropagates the decoded gradient
straight through. This is what a lossy cut means in split learning.

These hooks are built in:

| hook | what it does |
|------|--------------|
| `Float16`, `BFloat16` | send the tensor at half precision |
| `Quantize(bits=1/2/4/8)` | uniform per-tensor quantization, bit-packed |
| `TopK(ratio)` | per-sample top-k sparsification |
| `CircularConv(ratio)` | C3-SL batch-wise compression: each group of `ratio` samples goes as one vector (circular convolution with random keys), both ways; the gradient is exact for the decoded activation |
| `GaussianNoise(std)` | add noise, from a generator seeded per `(seed, cut, batch)`, so the global RNG is untouched and runs reproduce in every topology |
| `Recorder` | an observer; see below |
| `Chain([...])` | several hooks on one cut |

```python
plain = ts.run(build_model, get_data, run_config=config(), cuts=["1", "3"], verbose=False)
q4 = ts.run(build_model, get_data, run_config=config(), cuts=["1", "3"], verbose=False,
            cut_hooks={0: ts.Quantize(bits=4)})          # only the client -> server cut
print("bytes on cut 0:", plain[0]["bytes_fwd"][0], "->", q4[0]["bytes_fwd"][0])
print("loss:", plain[0]["loss"], "->", q4[0]["loss"])
```

The bytes are what actually goes on the wire. A hook's payload is packed into one
uint8 tensor, and its small header is counted too. `raw_bytes_total` is the same
traffic before encoding.

To write your own hook, override what you need. This one sends the activation's sign
and its mean magnitude, which is 1-bit compression of the forward pass only:

```python
class SignHook(ts.CutHook):
    def encode(self, x, ctx):
        return [x.sign().to(torch.int8), x.abs().mean().reshape(1)]

    def decode(self, payload, ctx):
        sign, scale = payload
        return sign.float() * scale


h = ts.run(build_model, get_data, run_config=config(), cuts=["1", "3"], verbose=False,
           cut_hooks={0: SignHook()})
print("sign hook: cut 0 sends", h[0]["bytes_fwd"][0], "bytes")
```

In a distributed run, each side gets its own copy of the hook (it is shipped with the
slice). A custom hook class must therefore be importable on the workers, like a custom
layer, and hook state is per side. The hook above is defined in a script, so here it
only runs locally.

Config files name the built-in hooks, and `ts.register_hook(name, cls)` adds yours:

```yaml
sl:
  cut_hooks:
    - {cut: 0, name: quantize, params: {bits: 8}}
    - {cut: all, name: gaussian_noise, params: {std: 0.01}}   # several on one cut: chained
```

### Observing a cut (attack metrics)

`ts.Recorder` keeps CPU copies of what the receiving side gets:
activations at slice `k + 1`, gradients at slice `k`. Those are the inputs to a
reconstruction or label-inference attack. In a distributed run the recorder runs on
the worker; pass `path=` to get its records back as files. Here it watches the
gradients the server sends back to the client in vanilla SL:

```python
rec = ts.Recorder(what="backward", max_batches=4)
ts.run(build_model, get_data, run_config=config("vanilla"), cuts=["1", "3"], verbose=False,
       cut_hooks={0: rec})
grads = [t for _, _, t in rec.records]            # [16, 32] gradients at cut 0, one per batch
labels = torch.cat([y for _, y in get_data()])
norms = torch.cat([g.norm(dim=1) for g in grads])
print("per-sample gradient norm, by label:",
      [round(norms[labels == c].mean().item(), 4) for c in range(4)])
```

## 3. Metrics

Every epoch's history entry (and its `metrics.jsonl` row) carries these keys, in every
executor:

| key | meaning |
|-----|---------|
| `bytes_fwd`, `bytes_bwd` | per cut, payload bytes sent forward / backward |
| `bytes_input`, `bytes_label` | raw inputs / labels sent over the network (0 when they stay on the data owner) |
| `bytes_total`, `bytes_per_batch`, `raw_bytes_total` | all of the above; before hooks for `raw_` |
| `bytes_client` | what crosses the data owner's boundary |
| `compute_s`, `client_compute_s`, `server_compute_s` | forward + backward + optimizer wall time per slice, and summed by owner |
| `sim_comm_s`, `sim_client_comm_s` | with `sl.link_bandwidth_mbps` / `sl.link_latency_ms`: the time those bytes would take on that link (per message: latency + bytes / bandwidth, no overlap), for every message and for the client's only |
| `sl_setting` | the setting in effect |

The bytes are tensor payload bytes, without gRPC/TCP framing. They are identical
whether `LocalExecutor` counts them or the workers do (the tests check this). So a
local run predicts a distributed run's traffic exactly. Compute time is wall time; on
CUDA it is measured with a synchronize at the end of each phase.

```python
h = ts.run(build_model, get_data, run_config=config("vanilla", link_bandwidth_mbps=10,
           link_latency_ms=20), cuts=["1", "3"], verbose=False)
e = h[0]
print(f"client compute {e['client_compute_s'] * 1e3:.2f} ms, server {e['server_compute_s'] * 1e3:.2f} ms, "
      f"client link time on 10 Mbit/s + 20 ms: {e['sim_client_comm_s']:.3f} s")
```

## 4. Several clients: `ts.ClientSession`

Each client has its own data and its own copy of the client slices (the slices the
data owner hosts). The server slices are shared. The executor holds one copy of the
client slices and swaps each client's state in and out between batches. So sessions
work wherever the client slices are in-process: local runs, P2P, and centralized runs
with `vanilla` or `u_shaped`.

- `mode="sequential"`: client 0 trains on all its batches, then client 1, and so on.
  One round is one epoch. Moving from one client to the next calls `handoff`:
  - `"copy"` (the default) passes the client model on, with its weights and optimizer
    state, as in multi-client SplitNN. With "copy", a sequential session trains
    exactly like one client on the concatenated data.
  - `"keep"` gives each client back its own state, so only the server is shared.
  - A callable `handoff(prev_state, next_state, info)` does anything else.
- `mode="parallel"`: the clients take turns one batch at a time, each with its own
  client state. At the end of each round, `on_round_end(states, info)` may combine
  them. TorchSlicer itself never averages: averaging is a hook. FedAvg over the client
  slices, as in SplitFed, is a few lines.

```python
data = get_data()
clients = [data[:2], data[2:]]                   # two data owners


def fedavg(states, info):
    """Average the clients' client-side weights (SplitFed-style); keep optimizer states."""
    avg = {i: {"model": {k: sum(s[i]["model"][k] for s in states) / len(states)
                         for k in states[0][i]["model"]}} for i in states[0]}
    return [{i: {**s[i], "model": avg[i]["model"]} for i in s} for s in states]


seq = ts.ClientSession(clients, mode="sequential")                  # handoff="copy"
par = ts.ClientSession(clients, mode="parallel", on_round_end=fedavg)
for session in (seq, par):
    sliced = ts.slice(build_model(), cuts=["1", "3"])
    h = sliced.train(session, run_config=config("u_shaped"))
    print(session.mode, h[0]["loss"], "client slices per state:", sorted(session.states[0]))
```

A state is `{slice_index: {"model": state_dict, "optimizer": state_dict}}`. After
training, `session.states[k]` is client `k`'s final client-side state. With
`ts.run`, set `sl.clients: K` (and `sl.client_mode`, `sl.handoff`), and `data_fn()`
returns one loader per client.

## 5. Clients as nodes: `sl.client_nodes`

A session simulates its clients in one process: right for accuracy studies, wrong for
measuring what a deployment costs, since every client then runs on one machine and the
data owner holds every client's data. With `sl.client_nodes: true` (centralized
topology), each client is its own node:

- a **client node** holds slice 0 and only its own loader (`ROLE=client`,
  `CLIENT_INDEX=k`; or `ts.start_worker(data=loader, client_index=k)`);
- **server workers** hold slices 1.., shared by every client;
- the **coordinator** holds no data and no slice: it runs the schedule, the client
  selection and the round-end hooks.

For each batch, the coordinator tells a client to run its next batch. The client sends
the labels to the last slice and the activation to slice 1, and the gradient comes back
to it: slice 1 knows every client's address and returns each gradient to the client of
its batch. Clients only open connections outward, so the NAT tunnels of
[nat.md](nat.md) work unchanged, as do gRPC, `tcp` and `ipc` (not `nccl`, whose process
group can't change after init). Client states stay on their nodes and cross the network
only when needed: a sequential `copy` handoff (`keep` moves nothing) and the round-end
hooks of an `Algorithm` (e.g. `ts.SplitFedAlgorithm`). The history counts those bytes
in `bytes_state`. An algorithm with per-client server copies (`ts.SplitFedV1Algorithm`,
`needs_per_client_server_state`) makes every server worker keep one copy of its slice
per client, swapped in when that client's batch arrives; at the end of the round the
coordinator gathers the participants' copies for `aggregate` and every copy restarts
from the result.

One batch is in flight at a time, so a run on client nodes trains exactly like the
`ClientSession` of section 4 on the same clients. On the coordinator,
`ts.RemoteSession(n_clients, mode, handoff)` takes the place of the loaders:

```python
cfg = config("vanilla", topology="centralized", clients=2)
clients_data = [get_data()[:2], get_data()[2:]]
servers = [ts.start_worker(ts.deploy.free_port(), host="127.0.0.1", run_config=cfg)
           for _ in range(2)]                    # slices 1 and 2
clients = [ts.start_worker(ts.deploy.free_port(), host="127.0.0.1", run_config=cfg,
                           data=clients_data[k], client_index=k) for k in range(2)]
executor = ts.ClientNodesExecutor(
    discovery=ts.StaticDiscovery(peers=[w.address for w in servers + clients]),
    coordinator_addr=f"127.0.0.1:{ts.deploy.free_port()}", run_config=cfg)
remote = ts.RemoteSession(2, mode="parallel")
h = ts.slice(build_model(), cuts=["1", "3"], executor=executor).train(
    remote, run_config=cfg, algorithm=ts.SplitFedAlgorithm())

local = ts.ClientSession(clients_data, mode="parallel")
ref = ts.slice(build_model(), cuts=["1", "3"]).train(
    local, run_config=config("vanilla"), algorithm=ts.SplitFedAlgorithm())
assert round(h[0]["loss"], 6) == ref[0]["loss"]
print("client nodes:", h[0]["loss"], "state bytes moved:", h[0]["bytes_state"])
for w in servers + clients:
    w.stop()
```

After training, `remote.states[k]` is client `k`'s final state, read from its node.

A client node that dies leaves the run (`sl.drop_failed_clients`, default true): its batch
in flight is dropped, the other clients go on, and it takes no part in later rounds or
aggregations (a dead server worker still stops the run). A node counts as dead only if it
also fails a liveness probe, so a live client that raises an error stops the run. The
history lists `clients_lost` and `batches_skipped`. `sl.round_deadline_s` stops scheduling
a round's batches after that many seconds.
With `ts.run`, set `sl.client_nodes: true` with `sl.clients` (and `sl.client_mode`,
`sl.handoff`): every node runs the same script, `data_fn()` returns one loader per
client, and a client node keeps only its own. `python -m torchslicer launch --target
processes|compose|baremetal` starts or renders one node per client
(`experiments/sl_client_nodes.yaml`; k8s is not supported yet).

## Limits

- **Where clients run.** With a `ClientSession`, all clients run in the data owner's
  process. With `sl.client_nodes` each client is its own node, but the schedule keeps
  one batch in flight: clients don't train concurrently yet.
- **Server batches.** In `parallel` mode the server takes the clients' batches one
  at a time and updates after each one. There is no batching across clients.
- **Labels on another party.** Every setting assumes that the data owner holds both
  the inputs and the labels. Vertical SL, where a separate party holds the labels, is
  not supported.
- **Lost slices.** If one of the data owner's own slices fails, the run stops: those
  slices hold the data (and with `u_shaped`, the labels), so they can't move. Sessions
  can't be combined with `fault_tolerance.enabled`.
