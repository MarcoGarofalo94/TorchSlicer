# Bare metal: several hosts

Every host needs the project (script + config) at the same path, a Python environment with
`torchslicer` and the model's dependencies, and open ports between the hosts (the worker
port, plus port + `transport.tensor_port_offset` for TCP tensors, and the coordinator port).

```bash
python -m torchslicer launch examples/run_anywhere/train.py \
    -c experiments/run_anywhere_centralized.yaml --target baremetal \
    --hosts 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 --workdir /srv/TorchSlicer --python python3
```

`--hosts` has one entry per node, with the driving node (coordinator or driver) first. The
launcher writes `runs/deploy/<name>-baremetal/<node>.sh`. Each script `cd`s to `--workdir`, exports
the node's env (role, addresses, peers, `EXPERIMENT_CONFIG`, `RUN_ID`) and `exec`s the script.
Copy each script to its host and run it, workers first:

```bash
ssh 10.0.0.2 sh -s < runs/deploy/torchslicer-baremetal/worker-0.sh &
ssh 10.0.0.3 sh -s < runs/deploy/torchslicer-baremetal/worker-1.sh &
ssh 10.0.0.4 sh -s < runs/deploy/torchslicer-baremetal/worker-2.sh &
ssh 10.0.0.1 sh -s < runs/deploy/torchslicer-baremetal/coordinator.sh
```

Nodes listed on the same host get consecutive ports, two apart with TCP tensors. Per-host
settings that are not in the shared YAML can be added to a script or exported before it
runs: `DEVICE`, `WORKER_COMPUTE_GFLOPS`, `WORKER_MAX_MEMORY_MB`, `WORKER_FIRST_SLICE`.

With `discovery.backend: coordinator`, workers register with the coordinator, so their
order (and slice) is the registration order unless one declares `first_slice`. With
`static`, the slice order is the `--hosts` order.

Tested here with all nodes on `127.0.0.1` (the scripts, run as written, match unsplit PyTorch).
Multi-host runs over a real network have not been tested yet.
