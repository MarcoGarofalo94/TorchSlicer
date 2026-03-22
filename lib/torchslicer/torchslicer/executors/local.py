import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseExecutor
from ..core.split_layer import SplitLayer
from ..monitor import tracer


class LocalExecutor(BaseExecutor):
    def __init__(self):
        self.split_layers: list[SplitLayer] = []
        self.criterion = None

    def setup(self, model_graph, partitions, optimizer_cfg: dict, criterion_cfg: dict) -> None:
        layers = model_graph.get_layers()
        n = len(partitions)
        self.split_layers = []
        for i, partition in enumerate(partitions):
            is_last = (i == n - 1)
            partition_layers = [layers[j] for j in partition.layer_indices]
            preds = partition.predecessors if partition.predecessors else None
            sl = SplitLayer(partition_layers, is_last=is_last, predecessors=preds)
            opt = getattr(optim, optimizer_cfg["name"])(sl.parameters(), **optimizer_cfg["params"])
            sl.set_optimizer(opt)
            self.split_layers.append(sl)
        self.criterion = getattr(nn, criterion_cfg["name"])(**criterion_cfg["params"])

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False) -> dict:
        total_loss = 0.0
        n_batches = 0
        n_total = len(data_loader)

        with tracer.span("torchslicer.epoch", epoch=epoch) as epoch_span:
            for inputs, labels in data_loader:
                batch_id = epoch * n_total + n_batches

                with tracer.span(
                    "torchslicer.batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    batch_index=n_batches,
                    input_shape=str(tuple(inputs.shape)),
                ) as batch_span:

                    # ── forward ──────────────────────────────────────────────
                    outputs = []
                    x = inputs
                    for i, sl in enumerate(self.split_layers):
                        with tracer.span(
                            "torchslicer.partition.forward",
                            partition=i,
                            batch_id=batch_id,
                            input_shape=str(tuple(x.shape)),
                        ) as fwd_span:
                            x = sl(x)
                            if fwd_span:
                                fwd_span.set_attribute("output_shape", str(tuple(x.shape)))
                        outputs.append(x)

                    loss = self.criterion(outputs[-1], labels)
                    total_loss += loss.item()
                    n_batches += 1
                    if batch_span:
                        batch_span.set_attribute("loss", loss.item())

                    # ── backward ─────────────────────────────────────────────
                    grad = None
                    with tracer.span("torchslicer.partition.backward", partition=len(self.split_layers) - 1, batch_id=batch_id):
                        grad = self.split_layers[-1].backward(loss=loss)
                        self.split_layers[-1].optimize()

                    for i in range(len(self.split_layers) - 2, -1, -1):
                        with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id):
                            grad = self.split_layers[i].backward(prev_g=grad, out=outputs[i])
                            self.split_layers[i].optimize()

                if verbose:
                    print(f"  [epoch {epoch} | batch {n_batches}/{n_total}] loss={loss.item():.4f}")

        avg = total_loss / n_batches if n_batches > 0 else 0.0
        if verbose:
            print(f"[epoch {epoch}] avg_loss={avg:.4f}")
        return {"loss": avg}

    def teardown(self) -> None:
        self.split_layers = []
        self.criterion = None
