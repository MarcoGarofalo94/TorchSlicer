# Run anywhere: one script, one config, every environment

A TorchSlicer experiment is **one training script** and **one YAML config**. The same pair runs
in one process on a laptop, as one process per node on a single machine, in Docker Compose,
on Kubernetes, or on several bare-metal hosts. Only the launch changes. The training does not:
on CPU, every target gives the losses of unsplit PyTorch
(`tests/test_deploy.py`, `examples/run_anywhere/`).

## The script

```text
import torchslicer as ts

def build_model(): ...          # called only on the node that drives training
def get_loader(): ...           # likewise: workers never see the data

if __name__ == "__main__":
    ts.run(build_model, get_loader, strategy="params")   # any ts.slice option
```

`ts.run` reads the config (`EXPERIMENT_CONFIG`, plus env overrides) and does what this node's
role says:

| `deploy.topology` | roles (`deploy.role`, env `ROLE`) | |
|---|---|---|
| `local` | `all` | `LocalExecutor`, one process, no network |
| `centralized` | `coordinator`, `worker`, `all` | a coordinator ships slices and drives; workers chain activations peer to peer |
| `p2p` | `driver`, `follower`, `all` | the driver holds the data and slice 0 and drives; no coordinator process |

`all` (the default when `ROLE` is unset) runs **every node in this process**. Workers or followers
start on loopback with `ts.start_worker` and the run goes through the real gRPC/TCP stack. This
makes it a faithful local simulation of the distributed run, and it is what `python train.py`
does with no environment set. The number of slices is `discovery.n_workers`; in P2P the driver
holds one of them.

## The launcher

```bash
python -m torchslicer launch SCRIPT -c CONFIG --target TARGET [options]
```

| target | what it does | guide |
|---|---|---|
| `local` | runs the script here with `ROLE=all` | [local.md](local.md) |
| `processes` | starts one OS process per node on this machine and waits for the run | [local.md](local.md) |
| `compose` | writes a `docker-compose.yml`, one service per node | [compose.md](compose.md) |
| `k8s` | writes `k8s.yaml`: ConfigMap, Indexed Job of workers + headless Service, coordinator/driver Job + Service | [k8s.md](k8s.md) |
| `baremetal` | writes one shell script per node for `--hosts` | [baremetal.md](baremetal.md) |

Every target is rendered from the same node list (`torchslicer.deploy.plan_nodes`). Each node
gets its name, role, address and a few env vars (`TOPOLOGY`, `ROLE`, `RUN_ID`, `N_WORKERS`,
`PORT`, `WORKER_ADDRESS`, `WORKER_PEERS`, `COORDINATOR_ADDRESS`, `DISCOVERY_BACKEND`,
`EXPERIMENT_CONFIG`). The targets differ only in addresses (loopback ports, service names,
pod DNS names, your hosts) and in packaging. Everything else (model, strategy, optimizer,
epochs, transport, fault tolerance, planning) comes from the YAML, which every node reads.

## Config

```yaml
discovery:
  n_workers: 3           # slices
  backend: coordinator   # centralized: workers register (any order) | static: fixed list
network:
  device: auto           # every node, local included ("auto" = cuda if present)
  worker_port: 50051
  coordinator_bind_address: 0.0.0.0:50054
transport:
  tensor: grpc           # or tcp (listener at worker port + tensor_port_offset)
deploy:
  topology: centralized  # local | centralized | p2p
  name: torchslicer      # compose project / k8s object prefix
  image: torchslicer:cpu # compose and k8s
  namespace: default     # k8s
  gpus: 0                # per node: compose device reservation / k8s nvidia.com/gpu
```

**Priority:** Python kwargs > env vars > YAML > defaults. `RunConfig.load()` applies env vars
*on top of* the YAML, so a per-node env var (for example `ROLE`, `PORT` or `DEVICE`) overrides
the shared file. The full env var → field table is `_ENV_OVERRIDES` in
`torchslicer/config.py`.

## Start-up order

Nodes may start in any order. With `discovery.backend: coordinator`, the coordinator waits up
to `discovery.timeout` for registrations. With a static list (P2P, k8s), the coordinator or
driver waits up to `discovery.timeout` for every listed worker to accept connections before
it probes their capabilities and plans. A worker that never comes up fails the run with
`worker <addr> not reachable within discovery.timeout=...`.
