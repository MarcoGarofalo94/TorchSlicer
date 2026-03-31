"""
Built-in pipeline schedules.

StandardSchedule  — plain single-batch forward → backward → step.
GPipeSchedule     — GPipe all-forward / all-backward with M micro-batches.
"""

import torch

from ..monitor import tracer
from .base import BasePipelineSchedule


class StandardSchedule(BasePipelineSchedule):
    """Plain forward-then-backward on a single full batch (no micro-batches)."""

    def step(self, split_layers, inputs, labels, aux, criterion, profilers, batch_id) -> float:
        outputs = []
        x = inputs
        for i, sl in enumerate(split_layers):
            with tracer.span(
                "torchslicer.partition.forward",
                partition=i,
                batch_id=batch_id,
                input_shape=str(tuple(x.shape)),
            ) as fwd_span:
                with profilers[i].phase("forward"):
                    x = sl(x, **aux) if (i == 0 and aux) else sl(x)
                if fwd_span:
                    fwd_span.set_attribute("output_shape", str(tuple(x.shape)))
            outputs.append(x)

        loss = criterion(outputs[-1], labels)

        with tracer.span("torchslicer.partition.backward",
                         partition=len(split_layers) - 1, batch_id=batch_id):
            with profilers[-1].phase("backward"):
                moe = split_layers[-1].pop_moe_aux_loss()
                total_loss = loss + moe if moe is not None else loss
                grad = split_layers[-1].backward(loss=total_loss)
            with profilers[-1].phase("optimizer"):
                split_layers[-1].optimize()

        for i in range(len(split_layers) - 2, -1, -1):
            with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id):
                with profilers[i].phase("backward"):
                    moe = split_layers[i].pop_moe_aux_loss()
                    if moe is not None:
                        outputs[i].backward(gradient=grad, retain_graph=True)
                        moe.backward()
                        sl_x = split_layers[i].x
                        grad = sl_x.grad if sl_x is not None else None
                    else:
                        grad = split_layers[i].backward(prev_g=grad, out=outputs[i])
                with profilers[i].phase("optimizer"):
                    split_layers[i].optimize()

        return loss.item()


class GPipeSchedule(BasePipelineSchedule):
    """GPipe all-forward / all-backward with M micro-batches.

    Step 1: forward all M micro-batches, saving cut-point tensors.
    Step 2: backward all M micro-batches, accumulating gradients.
    Step 3: single optimizer step across all split layers.

    Each micro-batch loss is scaled by 1/M so accumulated gradients match
    the magnitude of a full-batch gradient.
    """

    def __init__(self, n_micro: int):
        self.n_micro = max(1, n_micro)

    def step(self, split_layers, inputs, labels, aux, criterion, profilers, batch_id) -> float:
        M    = self.n_micro
        n_sl = len(split_layers)
        micro_mains  = inputs.chunk(M)
        micro_labels = labels.chunk(M)
        micro_aux = [
            {k: v.chunk(M)[m] if v.dim() > 0 and v.shape[0] == inputs.shape[0] else v
             for k, v in aux.items()}
            for m in range(M)
        ]

        # Step 1: forward all micro-batches
        micro_outs: list[list[torch.Tensor]] = []
        x_refs:     list[list[torch.Tensor]] = []
        micro_moe:  list[list]               = []

        for m in range(M):
            outs: list[torch.Tensor] = []
            x = micro_mains[m]
            for i, sl in enumerate(split_layers):
                with tracer.span(
                    "torchslicer.partition.forward",
                    partition=i,
                    batch_id=batch_id * M + m,
                    input_shape=str(tuple(x.shape)),
                ):
                    with profilers[i].phase("forward"):
                        x = sl(x, **micro_aux[m]) if (i == 0 and micro_aux[m]) else sl(x)
                outs.append(x)
            micro_outs.append(outs)
            x_refs.append([sl.x for sl in split_layers])
            micro_moe.append([sl.pop_moe_aux_loss() for sl in split_layers])

        # Step 2: backward all micro-batches, accumulate gradients
        total_loss = 0.0
        for m in range(M):
            loss_m = criterion(micro_outs[m][-1], micro_labels[m])
            total_loss += loss_m.item()
            scaled = loss_m / M

            moe_last = micro_moe[m][-1]
            if moe_last is not None:
                scaled = scaled + moe_last / M

            with tracer.span("torchslicer.partition.backward",
                             partition=n_sl - 1, batch_id=batch_id * M + m):
                with profilers[-1].phase("backward"):
                    scaled.backward()

            for i in range(n_sl - 2, -1, -1):
                with tracer.span("torchslicer.partition.backward",
                                 partition=i, batch_id=batch_id * M + m):
                    with profilers[i].phase("backward"):
                        moe_i = micro_moe[m][i]
                        if moe_i is not None:
                            micro_outs[m][i].backward(x_refs[m][i + 1].grad,
                                                      retain_graph=True)
                            (moe_i / M).backward()
                        else:
                            micro_outs[m][i].backward(x_refs[m][i + 1].grad)

        # Step 3: optimizer step
        for i, sl in enumerate(split_layers):
            with profilers[i].phase("optimizer"):
                sl.optimizer.step()
                sl.optimizer.zero_grad()
            sl.x = None

        return total_loss / M
