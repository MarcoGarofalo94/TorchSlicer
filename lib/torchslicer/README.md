# TorchSlicer

**Split learning for PyTorch** — partition a model across multiple devices or machines and train it end-to-end without any single node holding the full model.

```python
import torchslicer as ts

sliced = ts.slice(model, n=4)
sliced.train(loader, optimizer_cfg, criterion_cfg, epochs=10)
```

## What it does

TorchSlicer splits an `nn.Module` into sequential partitions, assigns each to a device or remote worker, and runs forward and backward passes across them in a pipeline. It handles:

- Gradient flow across partition boundaries
- Mixed-precision training (`bfloat16` autocast)
- GPipe micro-batch pipelining
- Fault tolerance with automatic recovery from worker checkpoints
- Run logging to structured JSONL for offline analysis

## Installation

```bash
# Core (local training, no networking)
pip install torchslicer

# With gRPC distributed support
pip install 'torchslicer[grpc]'

# All extras (gRPC + OpenTelemetry monitoring + LoRA/PEFT)
pip install 'torchslicer[grpc,monitor,peft]'
```

Requires Python ≥ 3.10 and PyTorch (not declared as a dependency — install the build that matches your hardware).

## Execution modes

| Mode | Use case |
|---|---|
| **Local** | Single process, all partitions on one machine. Fast iteration. |
| **Centralized** | Coordinator + gRPC workers. Each worker runs on a separate host/GPU. |
| **P2P** | No coordinator. Driver picks up peers directly. |

Transport layer is configurable: gRPC (default) or raw TCP for lower serialization overhead (~2.4× faster on plain split, ~1.6× faster with GPipe).

## Quick start — local

```python
import torchslicer as ts
import torchvision

model = torchvision.models.resnet18()
sliced = ts.slice(model, n=2)

optimizer_cfg = {"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9}}
criterion_cfg = {"name": "CrossEntropyLoss", "params": {}}

sliced.train(train_loader, optimizer_cfg, criterion_cfg, epochs=5, verbose=True)
```

## Quick start — distributed (coordinator + workers)

```python
# coordinator
from torchslicer import RunConfig
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.discovery import CoordinatorDiscovery
import torchslicer as ts

cfg = RunConfig.load("experiments/resnet18_4gpu.yaml")
executor = DistributedExecutor(
    discovery=CoordinatorDiscovery(run_id=cfg.run_id),
    coordinator_addr="0.0.0.0:50054",
    run_config=cfg,
)
sliced = ts.slice(model, n=cfg.discovery.n_workers, executor=executor)
sliced.train(loader, run_config=cfg)
```

Workers are started separately — see `examples/train/` for full entrypoints and the Docker Compose setup in the [repository](https://github.com/MarcoGarofalo94/TorchSlicer).

## Custom architectures

For models `torch.fx` cannot trace (HuggingFace LLMs, MoE, etc.), supply a `pack` function:

```python
def pack_qwen(model):
    return [
        ts.SimpleEmbedStage(model.model.embed_tokens),
        *[ts.BlockStage(layer) for layer in model.model.layers],
        ts.CausalLMHeadStage(model.model.norm, model.lm_head),
    ]

sliced = ts.slice(model, n=4, pack=pack_qwen)
```

Built-in stage types: `BlockStage`, `GPT2EmbedStage`, `SimpleEmbedStage`, `CausalLMHeadStage`, `AuxInputStage`, `MoEBlockStage`.

## Configuration

All parameters are controlled via `RunConfig` with a 4-level priority: Python kwargs > YAML file > environment variables > defaults.

```python
from torchslicer import RunConfig

cfg = RunConfig.load("experiments/resnet18_4gpu.yaml")  # YAML + env overrides
cfg.to_yaml("runs/my_run/resolved_config.yaml")          # save for reproducibility
```

## Monitoring

When `torchslicer[monitor]` is installed, runs emit OpenTelemetry traces. Point the exporter at any OTLP-compatible backend by setting:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://your-backend:4317
```

Compatible with Arize Phoenix, Jaeger, Grafana Tempo, Honeycomb, and any other OTLP collector. No library changes needed per backend.

Run metrics are also written to `runs/<run_id>/metrics.jsonl` for offline analysis.

## License

GPL-3.0-or-later. See [LICENSE](https://github.com/MarcoGarofalo94/TorchSlicer/blob/main/LICENSE).
