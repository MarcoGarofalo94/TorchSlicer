# Kubernetes

```bash
python -m torchslicer launch examples/run_anywhere/train.py \
    -c experiments/run_anywhere_p2p.yaml --target k8s
kubectl apply -f runs/deploy/torchslicer-k8s/k8s.yaml
kubectl wait --for=condition=complete job/torchslicer-driver --timeout=24h
kubectl logs job/torchslicer-driver
kubectl delete -f runs/deploy/torchslicer-k8s/k8s.yaml
```

`k8s.yaml` holds, in `deploy.namespace`:

| object | name | role |
|---|---|---|
| ConfigMap | `<name>-config` | the YAML config, mounted at `/etc/torchslicer/run.yaml` (`EXPERIMENT_CONFIG`) |
| Job (Indexed), `completions = parallelism = n` | `<name>-workers` | the workers / followers |
| headless Service | `<name>-workers` | gives pod *i* the stable DNS name `<name>-workers-<i>.<name>-workers` |
| Job | `<name>-coordinator` or `<name>-driver` | the node that drives training |
| Service | `<name>-coordinator` or `<name>-driver` | its address (workers report to it; P2P gradients return to it) |

**Ordering is deterministic.** Discovery is always static on k8s: the coordinator or driver
gets the worker DNS names in pod-index order, so slice *i* runs on pod *i*. It waits up to
`discovery.timeout` for pods that are still pulling their image. Raise `discovery.timeout`
for large images.

**Why Jobs, not a StatefulSet.** A training run is finite. When it ends, the driving node
shuts the workers down and both Jobs complete. A StatefulSet would restart exited workers
forever. An Indexed Job gives the same stable, ordered DNS names (pod hostname
`<job>-<index>` + `subdomain`). Pods use `restartPolicy: Never` and `backoffLimit: 0`: a
crashed worker fails the run (or, with `fault_tolerance.enabled`, the run continues on the
survivors) instead of coming back empty under the same name. For a long-lived worker pool
shared by many runs, use a StatefulSet with `WORKER_PERSISTENT=1` and
`discovery.backend: coordinator` instead; that is not generated.

**The image must contain the script**, at the same relative path under `/workspace`:

```dockerfile
FROM torchslicer:cpu
COPY . /workspace
```

and `deploy.image` names it. With `deploy.gpus: N` every pod requests `nvidia.com/gpu: N`.
Data is read only by the coordinator or driver pod; mount it there.

Not tested on a cluster: this container has no `kubectl`. The tests check the manifests'
structure (selectors match pod labels, the subdomain is the headless Service, the peer list
is the pod DNS names in index order, ports include the TCP tensor port). The node env is
the one `--target processes` runs and checks.
