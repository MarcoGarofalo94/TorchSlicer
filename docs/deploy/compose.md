# Docker Compose

```bash
make build-cpu                                    # once: the torchslicer:cpu image
python -m torchslicer launch examples/run_anywhere/train.py \
    -c experiments/run_anywhere_centralized.yaml --target compose
docker compose -f runs/deploy/torchslicer-compose/docker-compose.yml up --exit-code-from coordinator
docker compose -f runs/deploy/torchslicer-compose/docker-compose.yml down
```

The generated file has one service per node (`coordinator` + `worker-0..n-1`, or
`driver` + `follower-0..n-2`), on one bridge network, so a node's address is its service
name. The launcher's working directory (the project) is mounted at `/workspace`, so the
script and config paths are the same as on the host. In this repository,
`lib/torchslicer` is also mounted over the installed library, as in the dev compose files,
so code changes need no rebuild.

- `deploy.image` picks the image (`torchslicer:gpu` for CUDA). `deploy.gpus: 1` adds an NVIDIA
  device reservation to every service.
- `--exit-code-from <coordinator|driver>` makes `up` return when the run ends, with the
  driving node's exit code. It stops the other services too.
- The hand-written `docker-compose*.yml` files at the repo root (fixed 4 workers, the M0–M2
  examples, `make run-centralized` / `make run-p2p`) still work. The generated file is the
  same layout, with one node per slice the coordinator/driver doesn't host (`sl.setting`).

Not tested in this repository's CI: this container has no Docker. The services' env and
commands are the ones `--target processes` runs and checks.
