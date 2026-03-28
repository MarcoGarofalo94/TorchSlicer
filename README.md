# TorchSlicer

<img src="https://github.com/MarcoGarofalo94/TorchSlicer/assets/47697181/3b9d71dc-64f0-4adf-ac9b-40576593f532" width="156" height="156" />

**Split learning for PyTorch** — partition a model across multiple devices or machines and train it end-to-end without any single node holding the full model.

```bash
pip install 'torchslicer[grpc]'
```

```python
import torchslicer as ts

sliced = ts.slice(model, n=4)
sliced.train(loader, optimizer_cfg, criterion_cfg, epochs=10)
```

## Benchmark (ResNet-18, 2× RTX 3060, synthetic CIFAR-10, batch=64, 4096 samples)

| Mode | Epoch (s) | vs local |
|---|---|---|
| Local single-process | 2.40 | — |
| gRPC split | 8.48 | 3.5× slower |
| gRPC GPipe n_micro=4 | 7.66 | 3.2× slower |
| TCP split | 3.48 | 1.5× slower |
| TCP GPipe n_micro=4 | 4.91 | 2.0× slower |

TCP transport is ~2.4× faster than gRPC on plain split. The remaining gap vs local is serialization cost inherent to split learning.

## Execution modes

| Mode | Use case |
|---|---|
| **Local** | Single process, all partitions on one machine. Fast iteration. |
| **Centralized** | Coordinator + gRPC workers on separate hosts/GPUs. |
| **P2P** | No coordinator. Driver picks up peers directly. |

## Installation

```bash
pip install torchslicer                            # core only (local training)
pip install 'torchslicer[grpc]'                    # + distributed gRPC support
pip install 'torchslicer[grpc,monitor,peft]'       # + OpenTelemetry + LoRA
```

Requires Python ≥ 3.10. Install PyTorch separately to match your hardware.

## Quick start

```python
import torchslicer as ts
import torchvision

model = torchvision.models.resnet18()
sliced = ts.slice(model, n=2)

optimizer_cfg = {"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9}}
criterion_cfg = {"name": "CrossEntropyLoss", "params": {}}

sliced.train(train_loader, optimizer_cfg, criterion_cfg, epochs=5, verbose=True)
```

## Distributed training (Docker)

```bash
# Build image
make build-cpu

# Run coordinator + 4 workers
make run-centralized CONFIG=experiments/resnet18_4gpu.yaml

# With Phoenix tracing (http://localhost:6006)
make run-phoenix CONFIG=experiments/resnet18_4gpu.yaml
```

## Custom architectures

For models `torch.fx` cannot trace (HuggingFace LLMs, MoE, etc.):

```python
def pack_qwen(model):
    return [
        ts.SimpleEmbedStage(model.model.embed_tokens),
        *[ts.BlockStage(layer) for layer in model.model.layers],
        ts.CausalLMHeadStage(model.model.norm, model.lm_head),
    ]

sliced = ts.slice(model, n=4, pack=pack_qwen)
```

## Configuration

```python
from torchslicer import RunConfig

cfg = RunConfig.load("experiments/resnet18_4gpu.yaml")  # YAML + env overrides
cfg.to_yaml("runs/my_run/resolved_config.yaml")          # save for reproducibility
```

See `experiments/` for YAML examples.

## License

GPL-3.0-or-later
