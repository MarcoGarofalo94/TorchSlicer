from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from pathlib import Path
from typing import Callable

import torch

from .core.model_graph import ModelGraph
from .core.split_layer import SplitLayer


_CHECKPOINT_RE = re.compile(r"worker_(\d+)_epoch_(\d+)\.pt$")


def _build_layers(model, pack: Callable | None = None) -> list:
    if pack is not None:
        return list(pack(model))
    return ModelGraph.from_module(model).get_layers()


def _checkpoint_files(run_dir: str) -> dict[int, set[int]]:
    files: dict[int, set[int]] = {}
    for name in os.listdir(run_dir):
        match = _CHECKPOINT_RE.match(name)
        if not match:
            continue
        worker_index = int(match.group(1))
        epoch = int(match.group(2))
        files.setdefault(epoch, set()).add(worker_index)
    return files


def _load_manifest_layout(run_dir: str) -> dict[int, list[dict]]:
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    parts = manifest.get("split", {}).get("partitions", [])
    if not parts:
        raise ValueError(f"run manifest at {manifest_path} does not contain split metadata")
    return {0: parts}


def _load_layouts(run_dir: str) -> dict[int, list[dict]]:
    layout_path = os.path.join(run_dir, "checkpoint_layouts.json")
    if not os.path.exists(layout_path):
        return _load_manifest_layout(run_dir)

    with open(layout_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_epochs = payload.get("epochs", {})
    layouts: dict[int, list[dict]] = {}
    for epoch_text, parts in raw_epochs.items():
        layouts[int(epoch_text)] = parts
    if not layouts:
        raise ValueError(f"checkpoint layouts file at {layout_path} is empty")
    return layouts


def resolve_checkpoint_epoch(run_dir: str, epoch: int | None = None) -> int:
    layouts = _load_layouts(run_dir)
    files = _checkpoint_files(run_dir)

    if epoch is not None:
        layout = layouts.get(epoch)
        if layout is None and 0 in layouts:
            layout = layouts[0]
        if layout is None:
            raise ValueError(f"no checkpoint layout metadata found for epoch {epoch}")
        required = {int(part["worker_index"]) for part in layout}
        missing = sorted(required - files.get(epoch, set()))
        if missing:
            raise FileNotFoundError(
                f"missing checkpoint files for epoch {epoch}: workers {missing}"
            )
        return epoch

    candidates = sorted(files.keys(), reverse=True)
    for candidate in candidates:
        layout = layouts.get(candidate)
        if layout is None and 0 in layouts:
            layout = layouts[0]
        if layout is None:
            continue
        required = {int(part["worker_index"]) for part in layout}
        if required.issubset(files.get(candidate, set())):
            return candidate

    raise FileNotFoundError(f"no restorable checkpoint set found in {run_dir}")


def restore_model_from_run(
    model,
    run_dir: str,
    *,
    epoch: int | None = None,
    pack: Callable | None = None,
    strict: bool = True,
):
    """Load worker slice checkpoints from ``run_dir`` back into ``model``.

    ``model`` must be a fresh instance of the original training architecture.
    When the run used ``pack(model)``, pass the same ``pack`` callable here.
    The function restores parameters in place and returns ``model``.
    """
    resolved_epoch = resolve_checkpoint_epoch(run_dir, epoch)
    layouts = _load_layouts(run_dir)
    layout = layouts.get(resolved_epoch) or layouts.get(0)
    if layout is None:
        raise ValueError(f"no layer layout metadata found for epoch {resolved_epoch}")

    layers = _build_layers(model, pack=pack)

    for part in sorted(layout, key=lambda item: int(item["worker_index"])):
        worker_index = int(part["worker_index"])
        layer_indices = list(part["layer_indices"])
        checkpoint_path = os.path.join(
            run_dir, f"worker_{worker_index}_epoch_{resolved_epoch}.pt"
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        slice_layer = SplitLayer([layers[i] for i in layer_indices])
        slice_layer.load_state_dict(state["layer_state_dict"], strict=strict)

    return model


def _load_symbol(spec: str):
    if ":" not in spec:
        raise ValueError(f"expected module:path format, got {spec!r}")
    module_name, symbol_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f"{symbol_name!r} not found in module {module_name!r}") from exc


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recombine TorchSlicer worker checkpoints into a full model."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing worker checkpoints")
    parser.add_argument("--builder", required=True, help="Model builder in module:callable form")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch to restore")
    parser.add_argument("--pack", default=None, help="Optional pack function in module:callable form")
    parser.add_argument(
        "--format",
        choices=("state_dict", "model"),
        default="state_dict",
        help="Save only the combined state_dict or the full model object",
    )
    args = parser.parse_args(argv)

    builder = _load_symbol(args.builder)
    pack = _load_symbol(args.pack) if args.pack else None

    model = builder()
    resolved_epoch = resolve_checkpoint_epoch(args.run_dir, args.epoch)
    restore_model_from_run(model, args.run_dir, epoch=resolved_epoch, pack=pack)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": args.run_dir,
        "epoch": resolved_epoch,
        "state_dict": model.state_dict(),
    }
    if args.format == "model":
        payload["model"] = model
    torch.save(payload, output_path)
    print(f"saved restored {args.format} to {output_path} (epoch={resolved_epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
