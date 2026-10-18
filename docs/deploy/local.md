# Local: one process, or one process per node

## One process

```bash
python examples/run_anywhere/train.py                       # topology local (LocalExecutor)
EXPERIMENT_CONFIG=experiments/run_anywhere_centralized.yaml \
    python examples/run_anywhere/train.py                   # ROLE=all: coordinator + 3 loopback workers
python -m torchslicer launch examples/run_anywhere/train.py \
    -c experiments/run_anywhere_p2p.yaml --target local     # the same, through the launcher
```

With a distributed topology and no `ROLE`, `ts.run` starts the workers (or followers) in this
process on free loopback ports and runs the coordinator (or driver) against them over real
gRPC, or TCP with `transport.tensor: tcp`. This is the quickest way to test a distributed
config, and it is how the tests check that every mode matches unsplit PyTorch.

From Python, the same thing works without a YAML:

```python
cfg = ts.RunConfig()
cfg.deploy.topology = "p2p"
cfg.discovery.n_workers = 3
history = ts.run(build_model, get_loader, run_config=cfg)
```

## One OS process per node

```bash
python -m torchslicer launch examples/run_anywhere/train.py \
    -c experiments/run_anywhere_centralized.yaml --target processes [-o logs/] [--timeout 600]
```

The launcher starts the workers first, then the coordinator or driver, each as its own
process on `127.0.0.1` with free ports. The driving node's output goes to the terminal and
every other node's output goes to `<out>/<node>.log` (default `runs/deploy/<name>-processes/`).
The launcher returns the driving node's exit code. The driving node shuts the workers down
at the end of the run; any worker still alive 30 s later is terminated.

This is the bare-metal layout on one machine: the env each process gets is exactly what
`--target baremetal` writes into the per-host scripts.
