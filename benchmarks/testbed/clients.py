"""examples/data/train.py's model and data, printing every epoch's metrics as RESULT lines.

    python -m torchslicer launch benchmarks/testbed/clients.py -c benchmarks/testbed/clients.yaml \\
        --target baremetal --hosts COORD,SERVER,CLIENT0,...,CLIENT7 --workdir /path/to/TorchSlicer
    TOPOLOGY=local SL_CLIENT_NODES=0 python benchmarks/testbed/clients.py -c ...   # the in-process twin

One RESULT per epoch: the history entry (loss, duration_s, bytes_*, compute_s,
client_compute_s, clients_lost / clients_joined, ...), so a testbed run and its in-process
twin compare field by field.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "data"))
from train import SHAPES, SmallCNN, npz_shape   # noqa: E402

import torchslicer as ts  # noqa: E402


class _Epochs(ts.TrainingCallback):
    def __init__(self):
        self.metrics = []

    def on_epoch_end(self, epoch, metrics):
        self.metrics.append(dict(metrics))
        return metrics


def _plain(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return str(v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=None)
    args = ap.parse_args()
    cfg = ts.RunConfig.load(args.config) if args.config else ts.RunConfig.load()
    torch.manual_seed(cfg.data.seed)
    in_ch, n_classes, side = SHAPES.get(cfg.data.dataset) or npz_shape(cfg.data.params["path"])
    epochs = _Epochs()
    history = ts.run(lambda: SmallCNN(in_ch, n_classes, side), run_config=cfg,
                     cuts=["pool1", "pool2"][:cfg.discovery.n_workers - 1],   # 2 server workers: also at pool2
                     callbacks=[epochs])
    if history:
        # the callback's metrics (duration_s, epoch) + what the executor adds after it
        # (clients_lost, clients_joined, batches_skipped, bytes_state)
        for m in ({**c, **h} for c, h in zip(epochs.metrics, history)):
            print("RESULT " + json.dumps(_plain(m)), flush=True)
        print("HISTORY " + json.dumps([h["loss"] for h in history]), flush=True)
