#!/usr/bin/env python3
import json
import os
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset


def _build_model() -> nn.Module:
    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def _make_loader(batch_size: int, n_samples: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n_samples,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, num_workers=0)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    run_id = os.environ.get("RUN_ID", "bench_local_resnet_synth")
    out_dir = os.environ.get("LOG_DIR", "./runs")
    epochs = int(os.environ.get("EPOCHS", "2"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    n_samples = int(os.environ.get("N_TRAIN", "4096"))
    lr = float(os.environ.get("LR", "0.05"))
    momentum = float(os.environ.get("MOMENTUM", "0.9"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.0005"))
    mixed_precision = _bool_env("MIXED_PRECISION", False)
    seed = int(os.environ.get("SEED", "1234"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")

    loader = _make_loader(batch_size=batch_size, n_samples=n_samples, seed=seed)
    model = _build_model().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision and device.type == "cuda"))
    autocast_enabled = mixed_precision and device.type == "cuda"

    total_samples = 0
    total_batches = 0
    epoch_rows = []

    with open(metrics_path, "w", encoding="utf-8") as handle:
        global_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(loader, start=1):
                step_start = time.perf_counter()
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                step_ms = (time.perf_counter() - step_start) * 1000.0
                batch_loss = float(loss.detach().item())
                epoch_loss += batch_loss
                total_samples += inputs.shape[0]
                total_batches += 1

                handle.write(json.dumps({
                    "phase": "batch",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": round(batch_loss, 6),
                    "step_ms": round(step_ms, 3),
                    "samples": int(inputs.shape[0]),
                }) + "\n")

            epoch_duration_s = time.perf_counter() - epoch_start
            avg_loss = epoch_loss / max(1, batch_idx)
            row = {
                "phase": "epoch",
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "duration_s": round(epoch_duration_s, 3),
                "samples_per_s": round((batch_idx * batch_size) / epoch_duration_s, 3),
            }
            epoch_rows.append(row)
            handle.write(json.dumps(row) + "\n")

        total_duration_s = time.perf_counter() - global_start

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    summary = {
        "run_id": run_id,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "n_samples": n_samples,
        "mixed_precision": mixed_precision,
        "total_duration_s": round(total_duration_s, 3),
        "total_batches": total_batches,
        "total_samples": total_samples,
        "throughput_samples_per_s": round(total_samples / total_duration_s, 3),
        "throughput_batches_per_s": round(total_batches / total_duration_s, 3),
        "peak_mem_mb": round(peak_mem_mb, 3),
        "last_epoch_loss": epoch_rows[-1]["loss"] if epoch_rows else None,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
