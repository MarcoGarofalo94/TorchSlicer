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

## Run anywhere: one script, one config

```python
# train.py
import torchslicer as ts
ts.run(build_model, get_loader, strategy="params")   # the role comes from the config / ROLE env
```

```bash
python train.py                                                   # every node in this process
python -m torchslicer launch train.py -c run.yaml --target processes   # one process per node
python -m torchslicer launch train.py -c run.yaml --target compose     # or k8s, baremetal
```

`deploy.topology` (`local`, `centralized`, `p2p`) in the YAML picks the topology. The same pair
trains identically on every target. See [docs/deploy](docs/deploy/README.md).

## Distributed training (Docker)

```bash
# Build image
make build-cpu

# Run coordinator + 4 workers
make run-centralized CONFIG=experiments/resnet18_4gpu.yaml

# With Phoenix tracing (http://localhost:6006)
make run-phoenix CONFIG=experiments/resnet18_4gpu.yaml
```

## Choosing where to cut

```python
print(ts.trace(model).summary())                      # nodes, and where a cut is allowed
ts.slice(model, cuts=["layer2", "layer3.0"])          # named cuts (dotted names trace inside)
ts.slice(model, strategy="params", n=4)               # balance parameters
ts.slice(model, strategy="flops", n=4, example_input=x, expand="all")   # balance FLOPs, finer graph
ts.slice(model, strategy="profiled", n=4, example_input=x)              # balance measured time
```

Write your own strategy in a few lines: see [docs/tutorials/custom_strategy.md](docs/tutorials/custom_strategy.md).

## Custom architectures

If `torch.fx` can't trace a model (HuggingFace LLMs, MoE, etc.), `ts.slice` raises `ts.TraceError`
rather than guessing. Pass a `pack` function instead ([docs/tutorials/pack.md](docs/tutorials/pack.md)):

```python
def pack_qwen(model):
    return [
        ts.SimpleEmbedStage(model.model.embed_tokens),
        *[ts.BlockStage(layer) for layer in model.model.layers],
        ts.CausalLMHeadStage(model.model.norm, model.lm_head),
    ]

sliced = ts.slice(model, n=4, pack=pack_qwen)
```

**Tied weights** (one parameter used by several layers, e.g. the token embedding and the
LM head of GPT-2, Llama-3.2-1B, Qwen2.5-0.5B, Gemma) are handled automatically and train
exactly like the unsplit model. The lowest partition that uses the parameter owns it and is
the only one that updates it. Locally, all uses share one tensor. Across workers, the other
holders send their gradient to the owner before it steps, and the owner sends the new value
back before the batch completes. That costs one parameter-sized message in each direction
per batch.

## Configuration

```python
from torchslicer import RunConfig

cfg = RunConfig.load("experiments/resnet18_4gpu.yaml")  # YAML + env overrides
cfg.to_yaml("runs/my_run/resolved_config.yaml")          # save for reproducibility
```

See `experiments/` for YAML examples.

## License

GPL-3.0-or-later
