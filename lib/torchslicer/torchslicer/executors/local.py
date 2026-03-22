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
        self._n_micro = 1

    def setup(self, model_graph, partitions, optimizer_cfg: dict, criterion_cfg: dict, mixed_precision: bool = False, n_micro_batches: int = 1) -> None:
        layers = model_graph.get_layers()
        n = len(partitions)
        self.split_layers = []
        for i, partition in enumerate(partitions):
            is_last = (i == n - 1)
            partition_layers = [layers[j] for j in partition.layer_indices]
            preds = partition.predecessors if partition.predecessors else None
            sl = SplitLayer(partition_layers, is_last=is_last, predecessors=preds, mixed_precision=mixed_precision)
            opt = getattr(optim, optimizer_cfg["name"])(sl.parameters(), **optimizer_cfg["params"])
            sl.set_optimizer(opt)
            self.split_layers.append(sl)
        self.criterion = getattr(nn, criterion_cfg["name"])(**criterion_cfg["params"])
        self._n_micro = max(1, n_micro_batches)

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

                    if self._n_micro > 1:
                        loss_val = self._gpipe_batch(inputs, labels, batch_id)
                    else:
                        loss_val = self._standard_batch(inputs, labels, batch_id)

                    total_loss += loss_val
                    n_batches += 1
                    if batch_span:
                        batch_span.set_attribute("loss", loss_val)

                if verbose:
                    print(f"  [epoch {epoch} | batch {n_batches}/{n_total}] loss={loss_val:.4f}")

        avg = total_loss / n_batches if n_batches > 0 else 0.0
        if verbose:
            print(f"[epoch {epoch}] avg_loss={avg:.4f}")
        return {"loss": avg}

    # ── standard (non-GPipe) batch ─────────────────────────────────────────────

    def _standard_batch(self, inputs, labels, batch_id: int) -> float:
        # forward
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

        # backward
        with tracer.span("torchslicer.partition.backward", partition=len(self.split_layers) - 1, batch_id=batch_id):
            grad = self.split_layers[-1].backward(loss=loss)
            self.split_layers[-1].optimize()

        for i in range(len(self.split_layers) - 2, -1, -1):
            with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id):
                grad = self.split_layers[i].backward(prev_g=grad, out=outputs[i])
                self.split_layers[i].optimize()

        return loss.item()

    # ── GPipe micro-batch batch ────────────────────────────────────────────────

    def _gpipe_batch(self, inputs, labels, batch_id: int) -> float:
        """
        GPipe-style micro-batch pipelining:
          1. Forward all M micro-batches, saving cut-point tensors (x_refs).
          2. Backward all M micro-batches, accumulating gradients (no zero_grad between).
          3. Single optimizer step across all split layers.

        Each micro-batch loss is scaled by 1/M so accumulated gradients match
        the magnitude of a full-batch gradient.
        """
        M = self._n_micro
        n_sl = len(self.split_layers)
        micro_inputs = inputs.chunk(M)
        micro_labels = labels.chunk(M)

        # Step 1: forward all micro-batches
        # micro_outs[m][i]  = output of split_layer i for micro-batch m
        # x_refs[m][i]      = cut-point tensor (split_layer i's self.x) for micro-batch m
        micro_outs: list[list[torch.Tensor]] = []
        x_refs:     list[list[torch.Tensor]] = []

        for m in range(M):
            outs: list[torch.Tensor] = []
            refs: list[torch.Tensor] = []
            x = micro_inputs[m]
            for i, sl in enumerate(self.split_layers):
                with tracer.span(
                    "torchslicer.partition.forward",
                    partition=i,
                    batch_id=batch_id * M + m,
                    input_shape=str(tuple(x.shape)),
                ):
                    x = sl(x)
                refs.append(sl.x)   # save before next micro-batch overwrites sl.x
                outs.append(x)
            micro_outs.append(outs)
            x_refs.append(refs)

        # Step 2: backward all micro-batches, accumulate gradients
        total_loss = 0.0
        for m in range(M):
            loss_m = self.criterion(micro_outs[m][-1], micro_labels[m])
            total_loss += loss_m.item()
            scaled = loss_m / M     # scale so accumulated gradient ≈ full-batch gradient

            # Last split layer: backward from loss
            with tracer.span("torchslicer.partition.backward", partition=n_sl - 1, batch_id=batch_id * M + m):
                scaled.backward()
                # x_refs[m][-1].grad is now set

            # Earlier split layers: propagate gradient via saved outputs
            for i in range(n_sl - 2, -1, -1):
                with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id * M + m):
                    # x_refs[m][i+1].grad = gradient at input of layer i+1
                    #                     = gradient at output of layer i
                    micro_outs[m][i].backward(x_refs[m][i + 1].grad)
                    # x_refs[m][i].grad now accumulated

        # Step 3: single optimizer step for all split layers
        for sl in self.split_layers:
            sl.optimizer.step()
            sl.optimizer.zero_grad()
            sl.x = None     # release cut-point tensor

        return total_loss / M   # average micro-batch loss ≈ full-batch loss

    def teardown(self) -> None:
        self.split_layers = []
        self.criterion = None
