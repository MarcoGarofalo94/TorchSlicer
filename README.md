# TorchSlicer

<img src="https://github.com/MarcoGarofalo94/TorchSlicer/assets/47697181/3b9d71dc-64f0-4adf-ac9b-40576593f532" width="156" height="156" />

**Split learning for PyTorch** — partition a model across devices or machines and train it end-to-end without any single node holding the full model.

```bash
pip install 'torchslicer[grpc]'
```

```python
import torchslicer as ts

history = ts.run(model, train_loader, n=4, epochs=10, optimizer="adamw")
```

## Why TorchSlicer

Split learning is a privacy-preserving training paradigm: raw data never leaves the client, only intermediate activations are exchanged. TorchSlicer aims to be the canonical framework for split learning research and production deployment — the Flower of split learning.

- **One-liner training** — `ts.run()` with string optimizer/criterion shortcuts
- **Bring your own model** — works with any `nn.Module` via `torch.fx` tracing or a custom `pack` function
- **Research-ready** — pipeline schedules, topology variants, and activation hooks are first-class primitives
- **HuggingFace support** — `ts.from_pretrained()` with auto-detected packs for GPT-2, BERT, LLaMA, and more
- **Distributed** — local, centralized (gRPC/TCP coordinator + workers), and P2P modes

## Installation

```bash
pip install torchslicer                            # local training only
pip install 'torchslicer[grpc]'                    # + distributed gRPC/TCP
pip install 'torchslicer[grpc,monitor,peft]'       # + OpenTelemetry tracing + LoRA
```

Requires Python ≥ 3.10. Install PyTorch separately to match your hardware.

## Quick start

### One-liner

```python
import torchslicer as ts

history = ts.run(model, train_loader, n=4, epochs=5, optimizer="adamw", criterion="cross_entropy")
# history = [{"loss": 1.42}, {"loss": 0.91}, ...]
```

Optimizer and criterion accept either a string shorthand (`"adam"`, `"sgd"`, `"mse"`, etc.) or a full config dict for when you need custom parameters:

```python
history = ts.run(model, train_loader, n=4, optimizer={"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9}})
```

### Manual slicing

```python
sliced = ts.slice(model, n=4)
sliced.train(train_loader, optimizer="adamw", criterion="cross_entropy", epochs=10, verbose=True)
```

## Partitioning strategies

```python
# Default: split layers uniformly
sliced = ts.slice(model, n=4, strategy="uniform")

# Balance by trainable parameter count
sliced = ts.slice(model, n=4, strategy="param_balanced")

# Explicit layer boundaries (indices into the layer list)
from torchslicer.strategies.explicit import ExplicitSplitter
from torchslicer.strategies.registry import register
sliced = ts.slice(model, n=3, strategy="explicit")   # with boundaries=[2, 5] set via splitter directly

# Custom strategy
@register("my_strategy")
class MySplitter(ts.BaseSplitter):
    def split(self, model_graph, n_partitions):
        ...

sliced = ts.slice(model, n=4, strategy="my_strategy")
```

## Custom architectures (pack functions)

For models `torch.fx` cannot trace — HuggingFace LLMs, MoE, models with dynamic control flow — supply a `pack` function that returns a list of stages:

```python
def pack_qwen(model):
    return [
        ts.SimpleEmbedStage(model.model.embed_tokens),
        *[ts.BlockStage(layer) for layer in model.model.layers],
        ts.CausalLMHeadStage(model.model.norm, model.lm_head),
    ]

sliced = ts.slice(model, n=4, pack=pack_qwen)
history = ts.run(model, loader, n=4, pack=pack_qwen, optimizer="adamw")
```

Built-in stage types: `BlockStage`, `SimpleEmbedStage`, `GPT2EmbedStage`, `CausalLMHeadStage`, `AuxInputStage`, `MoEBlockStage`.

The `pack` function is the primary extension point. TorchSlicer ships built-in packs for common HuggingFace architectures as a convenience, but you are expected to write your own for any architecture that needs custom handling.

## HuggingFace integration

```python
# Load by name — pack auto-detected from model type
sliced = ts.from_pretrained("bert-base-uncased", n=4, task="classification")

# Already-loaded model — same API
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
sliced = ts.from_pretrained(model, n=4)

# Always override with your own pack — it always takes priority
sliced = ts.from_pretrained(model, n=4, pack=my_pack)
```

PEFT/LoRA models are automatically unwrapped before slicing.

Built-in auto-detected architectures: `gpt2`, `bert`, `roberta`, `distilbert`, `llama`, `mistral`, `qwen2`, `mixtral` (MoE).

## Pipeline schedules

```python
from torchslicer.pipeline import StandardSchedule, GPipeSchedule
from torchslicer.executors.local import LocalExecutor

# Standard: forward → backward → step per batch
executor = LocalExecutor(schedule=StandardSchedule())

# GPipe: all-forward / all-backward with M micro-batches
executor = LocalExecutor(schedule=GPipeSchedule(n_micro=4))

history = ts.run(model, loader, n=4, executor=executor, optimizer="adamw")
```

Implement `BasePipelineSchedule` to plug in any custom schedule (1F1B, async, etc.).

## Topology variants

Topology objects describe how partitions map to roles. They are informational by default — you use them to route partitions to the right devices/processes in a distributed setup.

```python
from torchslicer.topology import PipelineTopology, UShapedTopology

sliced = ts.slice(model, n=3)

# Standard pipeline: all partitions are "stage"
roles = PipelineTopology().assign(sliced.partitions)
# {"stage": [p0, p1, p2]}

# U-shaped split learning (Vepakomma et al. 2018):
# client holds bottom + head, server holds middle
roles = UShapedTopology().assign(sliced.partitions)
# {"client": [p0, p2], "server": [p1]}
```

## Activation hooks

Hooks intercept smashed activations at partition boundaries. They are passed to a schedule and applied during the forward pass.

```python
from torchslicer.hooks import DPNoiseHook, NoPeekHook
from torchslicer.pipeline import StandardSchedule
from torchslicer.executors.local import LocalExecutor

# Gaussian noise for differential privacy
schedule = StandardSchedule(hooks=[DPNoiseHook(sigma=0.1)])

# Distance-correlation penalty to reduce information leakage
# (NoPeek, Vepakomma et al. ICDMW 2020)
schedule = StandardSchedule(hooks=[NoPeekHook(lambda_=0.05)])

# Combine freely
schedule = StandardSchedule(hooks=[DPNoiseHook(sigma=0.05), NoPeekHook(lambda_=0.05)])

executor = LocalExecutor(schedule=schedule)
history = ts.run(model, loader, n=3, executor=executor, optimizer="adamw")
```

Implement `ActivationHook` to add any custom intervention — gradient masking, compression, watermarking, etc.

```python
from torchslicer.hooks import ActivationHook

class MyHook(ActivationHook):
    def on_forward_smash(self, smashed, raw_input, partition_idx):
        # modify or monitor smashed activations
        return smashed

    def pop_aux_loss(self):
        # optional: return a scalar loss term added to the main loss
        return None
```

## Distributed training

### Centralized (coordinator + workers)

```bash
make build-cpu
make run-centralized CONFIG=experiments/resnet18_4gpu.yaml
```

### P2P (no coordinator)

```bash
make run-p2p CONFIG=experiments/resnet18_2gpu_p2p.yaml
```

### Local simulation (no Docker)

```python
# Spawn N worker subprocesses locally — no config file, no Docker
history = ts.simulate(model, loader, n=4, devices=["cuda:0", "cuda:1", "cpu", "cpu"])
```

## Monitoring

```bash
# With Phoenix tracing (OpenTelemetry — http://localhost:6006)
make run-phoenix CONFIG=experiments/resnet18_4gpu.yaml
```

Run logs land in `./runs/<timestamp>/` — `metrics.jsonl`, per-worker profiling, `run_manifest.json`.

Restore a full model from distributed checkpoints:

```python
model = ts.restore_model_from_run("runs/2024-01-15_12-30-00", epoch=5)
```

## Configuration

```python
from torchslicer import RunConfig

cfg = RunConfig.load("experiments/resnet18_4gpu.yaml")  # YAML + env var overrides
cfg.to_yaml("runs/my_run/resolved_config.yaml")          # save for reproducibility
```

## Benchmark

ResNet-18 · 2× RTX 3060 · synthetic CIFAR-10 · batch=64 · 4096 samples

| Mode | Epoch (s) | vs local |
|---|---|---|
| Local single-process | 2.40 | — |
| gRPC split | 8.48 | 3.5× |
| gRPC GPipe n_micro=4 | 7.66 | 3.2× |
| TCP split | 3.48 | 1.5× |
| TCP GPipe n_micro=4 | 4.91 | 2.0× |

TCP is ~2.4× faster than gRPC for plain split. The remaining gap vs local is tensor serialization inherent to split learning.

## License

GPL-3.0-or-later
