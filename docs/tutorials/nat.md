# NAT-friendly transport: clients that only dial out

In split learning the data owner (the client) often sits behind NAT or a firewall: a
laptop, a phone, a hospital network. It can open connections to the server, but nothing
can connect to it. By default, though, every TorchSlicer node *receives* connections:

- the gradient for slice 0 comes back by a call **to** the data owner;
- in U-shaped SL the server sends the activation **to** the client's tail slice;
- the last slice reports each loss to the coordinator, any worker reports errors to it,
  and tied weights send gradients and updates both ways.

M6 removes that requirement. Every code block below runs as written (`tests/test_docs.py`
executes them).

## 1. How it works

A node behind NAT (`transport.nat: true`, env `TRANSPORT_NAT`) listens on nothing. It
dials every node that has to call it, opens one gRPC stream there (`TunnelService.open`)
and announces the addresses it serves. From then on, that node sends its calls to the
NAT'd node down the stream instead of dialing it. The NAT'd node's own calls go out
directly, as before. So the gradient, the U-shaped activation, the metrics, heartbeats,
errors and tied-weight messages all come back on connections the client opened.

```
  client (behind NAT)                          server (reachable)
  slice 0, tail ──── activations ──────────▶  slice 1
                ──── tunnel (opened by the client) ─────
                ◀─── gradients, tail activations, metrics, heartbeats
```

Nothing else changes. A tunnel carries the same serialized messages gRPC would send, so
what crosses each cut (bytes, hooks, metrics) is unchanged and training is identical.
Every TorchSlicer gRPC server (worker, coordinator, tail slice, relay) accepts tunnels.
The code is in `torchslicer/transport/tunnel.py`: a per-node `Router` decides, call by
call, whether an address is served in-process, behind an inbound tunnel, through the
relay, or directly.

Raw TCP (`transport.tensor: tcp`) still works. It is used for every link that can be
dialed, and the tunnel carries the rest.

## 2. The data owner behind NAT

This is the common case: the P2P driver, or a centralized coordinator hosting the client
slices (`sl.setting: vanilla` or `u_shaped`), is the client. Set `transport.nat` on it.
Here the followers are started in this process on loopback, so the run is a faithful
local simulation:

```python
import torch
import torch.nn as nn
import torchslicer as ts


def build_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
                         nn.Linear(16, 4))


def data():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, 8, generator=g), torch.randint(0, 4, (64,), generator=g)
    return [(x[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16)]


def config(**transport):
    cfg = ts.RunConfig()
    cfg.training.epochs = 2
    cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
    cfg.logging.enabled = False
    cfg.network.device = "cpu"
    cfg.sl.setting = "u_shaped"          # the driver hosts slice 0 and the last slice
    for k, v in transport.items():
        setattr(cfg.transport, k, v)
    return cfg


follower = ts.start_worker(host="127.0.0.1", run_config=config())
cfg = config(nat=True)
driver = ts.P2PExecutor(peers=[follower.address], address="client:1", run_config=cfg)
history = ts.slice(build_model(), cuts=["1", "3"], executor=driver).train(data(), run_config=cfg)
follower.stop()

assert driver._grpc_server is None                  # the client never listened
print("calls that reached the client by tunnel:", driver._router.stats["calls_in"])
print([round(h["loss"], 4) for h in history])
```

`address="client:1"` is only a name here: nobody dials it. The followers learn it from
the slice config and reach it through the tunnel the driver opened to each of them before
sending their slices.

With `ts.run` and the launcher, `transport.nat` in the shared config means the driving
node (coordinator or driver). `plan_nodes` gives it `TRANSPORT_NAT=true` and the others
`false`, and `ROLE=all` simulates the same thing. A ready-made config:
`experiments/nat_u_shaped_p2p.yaml`.

```python
cfg = config(nat=True)
cfg.deploy.topology = "p2p"
cfg.discovery.n_workers = 3                         # slices: 2 on the client, 1 follower
same = ts.run(build_model, data, run_config=cfg, verbose=False, cuts=["1", "3"])
assert [h["loss"] for h in same] == [h["loss"] for h in history]
```

A NAT'd centralized coordinator can't take registrations, because workers would have to
dial it. List the workers (`discovery.backend: static`) or use a relay. The executor says
so if you don't.

## 3. Workers behind NAT

A worker can be behind NAT too (env `TRANSPORT_NAT=true` on that node). It opens a
tunnel to whoever drives it: the coordinator, or the P2P driver. That address comes
from `network.coordinator_address`, else the first of `discovery.peers`, and the
launcher sets it on every worker. When its slice arrives (`init`), it also opens tunnels
to its neighbours and tied-weight peers. A NAT'd worker has a name (`WORKER_ADDRESS`),
not a port. It registers as usual, so `CoordinatorDiscovery` works.

```python
coordinator_port = ts.deploy.runner.free_port()
coordinator = f"127.0.0.1:{coordinator_port}"
cfg = config()
cfg.sl.setting = "pipeline"
nat_worker = ts.start_worker(address="home-gpu:1", run_config=cfg, nat=True,
                             tunnel_to=[coordinator])
assert nat_worker.servicer._tensor_server is None and nat_worker.servicer.nat
nat_worker.stop()
```

Two NAT'd nodes can't talk to each other this way, because neither can dial the other.
If two neighbours are both behind NAT, the second one's `init` fails and says to use a
relay.

## 4. Everyone behind NAT: the relay

A relay is a small reachable process that holds no slice:

    python -m torchslicer relay --port 50060

Every node with `transport.relay: host:50060` (env `TRANSPORT_RELAY`) dials only the
relay, announces its addresses there, and sends every call through it. The relay forwards
each call down the tunnel that announced the target address, or dials the target directly
if no tunnel announced it. With a relay, any mix of NAT'd and reachable nodes works,
including CoordinatorDiscovery with a NAT'd coordinator. Each call takes two hops, and raw
TCP is not used.

```python
from torchslicer.transport.tunnel import start_relay

relay = start_relay(host="127.0.0.1")
cfg = config(nat=True, relay=relay.address)
followers = [ts.start_worker(address=f"phone-{i}:1", run_config=cfg) for i in range(2)]
driver = ts.P2PExecutor(peers=[f.address for f in followers], address="client:1", run_config=cfg)
h = ts.slice(build_model(), cuts=["1", "2", "3"], executor=driver).train(data(), run_config=cfg)
for f in followers:
    f.stop()
relay.stop()
print("calls forwarded by the relay:", relay.router.stats["forwarded"])
```

Every node above (driver and both followers) is behind NAT, and the two followers are
adjacent. Only the relay accepts connections. In a config, `transport.relay: auto` asks
the launcher for a relay node (`python -m torchslicer relay`, a Deployment + Service on
k8s), and `ROLE=all` starts one on loopback.

## 5. Failures and keepalive

- **A NAT'd node dies.** Its tunnels close, and calls to it fail at once with
  `UNAVAILABLE`. A NAT'd address is never dialed directly. Heartbeats catch it like any
  dead worker: `WorkerFailureError`, or recovery with `fault_tolerance.enabled`.
- **The tunnel breaks.** The dialing side dials again with backoff and announces itself
  again, so a short outage is not a failure.
- **Idle NAT mappings.** A dialing node pings its idle tunnels every
  `transport.tunnel_keepalive_s` (20 s), so the mapping in the NAT stays open.
- **Errors keep their meaning.** A failed tunneled call raises an error with the same
  `code()` as gRPC: `UNAVAILABLE`, `DEADLINE_EXCEEDED`, or `UNKNOWN` with the remote
  exception text. Retry and failure detection don't change.

## 6. Cost

`benchmarks/nat/bench_nat.py` runs one U-shaped P2P run as OS processes on loopback,
with 1 MiB per cut and direction per batch. It runs the same thing with every node
reachable, with the client behind NAT, and with everything through a relay. The numbers
are in `notes/validation.md` (round 7). The tunnel costs about as much as the direct
gRPC path it replaces. The relay adds a hop to every message. Over a real WAN the extra
hop matters more than the extra copy.

## Not covered

- **NAT hole punching (STUN/ICE).** Two NAT'd peers always go through the relay. Direct
  NAT'd-to-NAT'd connections are not attempted.
- **Encryption and authentication.** Tunnels and the relay are plain gRPC, like every
  TorchSlicer channel. Put them behind TLS or a VPN on untrusted networks; TLS support is
  future work.
