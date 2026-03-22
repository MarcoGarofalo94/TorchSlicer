"""
TorchSlicer Monitor Dashboard
------------------------------
Reads traces from Jaeger's HTTP query API and serves a custom dashboard
that visualises the training topology, per-batch swimlane, and loss history.

Environment variables:
  JAEGER_HTTP        Jaeger query base URL  (default: http://localhost:16686)
  DASHBOARD_PORT     Port to listen on      (default: 8080)
  POLL_INTERVAL      WebSocket push period  (default: 2.0 seconds)
"""

import asyncio
import json
import os

import requests as _requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

JAEGER = os.environ.get("JAEGER_HTTP", "http://localhost:16686")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "2"))

app = FastAPI(title="TorchSlicer Monitor")

_here = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_here, "static")), name="static")


@app.get("/")
async def index():
    with open(os.path.join(_here, "static", "index.html")) as f:
        return HTMLResponse(f.read())


# ── Jaeger helpers ─────────────────────────────────────────────────────────────

def _get(path: str, **params) -> dict:
    try:
        r = _requests.get(f"{JAEGER}{path}", params=params, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _tags(span: dict) -> dict:
    return {t["key"]: t["value"] for t in span.get("tags", [])}


def _ms(span: dict) -> float:
    return round(span.get("duration", 0) / 1_000, 2)   # µs → ms


# ── topology ───────────────────────────────────────────────────────────────────

def _fetch_topology() -> list[dict]:
    data = _get("/api/traces", service="torchslicer-worker",
                operation="torchslicer.worker.init", limit=20)
    workers: dict[str, dict] = {}
    for trace in data.get("data", []):
        for span in trace.get("spans", []):
            if span.get("operationName") != "torchslicer.worker.init":
                continue
            tags = _tags(span)
            host = tags.get("worker", "unknown")
            if host not in workers:
                raw = tags.get("layers", "")
                layers = [l.strip().strip("'\"") for l in raw.split(",") if l.strip()]
                workers[host] = {
                    "hostname": host,
                    "layers": layers,
                    "n_layers": int(tags.get("n_layers", len(layers))),
                    "is_last": bool(tags.get("is_last", False)),
                    "prev": tags.get("prev_worker") or None,
                    "next": tags.get("next_worker") or None,
                }
    # Order by chain: find the first worker (no prev)
    ordered = []
    by_host = dict(workers)
    cur = next((w for w in by_host.values() if not w["prev"]), None)
    while cur:
        ordered.append(cur)
        nxt = cur.get("next")
        cur = by_host.get(nxt) if nxt else None
    # Append any disconnected workers not found in chain
    for w in by_host.values():
        if w not in ordered:
            ordered.append(w)
    return ordered


# ── coordinator batches ────────────────────────────────────────────────────────

def _fetch_coordinator_batches(limit: int = 60) -> list[dict]:
    data = _get("/api/traces", service="torchslicer-coordinator",
                operation="torchslicer.epoch", limit=10)
    batches = []
    for trace in data.get("data", []):
        for span in trace.get("spans", []):
            if span.get("operationName") != "torchslicer.batch":
                continue
            tags = _tags(span)
            batches.append({
                "batch_id": int(tags.get("batch_id", -1)),
                "epoch": int(tags.get("epoch", 0)),
                "loss": float(tags.get("loss", 0.0)),
                "total_ms": _ms(span),
                "start_us": span.get("startTime", 0),
            })
    batches.sort(key=lambda b: (b["epoch"], b["batch_id"]))
    return batches[-limit:]


# ── worker spans ───────────────────────────────────────────────────────────────

def _fetch_worker_spans(limit: int = 300) -> dict[int, dict]:
    """Returns {batch_id: {hostname: {fwd_ms, bwd_ms, output_shape}}}"""
    result: dict[int, dict] = {}
    for op, key in [
        ("torchslicer.worker.forward",  "fwd_ms"),
        ("torchslicer.worker.backward", "bwd_ms"),
    ]:
        data = _get("/api/traces", service="torchslicer-worker",
                    operation=op, limit=limit)
        for trace in data.get("data", []):
            for span in trace.get("spans", []):
                if span.get("operationName") != op:
                    continue
                tags = _tags(span)
                bid = int(tags.get("batch_id", -1))
                host = tags.get("worker", "unknown")
                if bid < 0:
                    continue
                node = result.setdefault(bid, {}).setdefault(host, {})
                node[key] = _ms(span)
                if key == "fwd_ms":
                    node["input_shape"] = tags.get("input_shape", "")
                    node["output_shape"] = tags.get("output_shape", "")
                    node["is_last"] = bool(tags.get("is_last", False))
                if key == "bwd_ms" and tags.get("loss"):
                    node["loss"] = float(tags.get("loss", 0.0))
    return result


# ── state assembly ─────────────────────────────────────────────────────────────

def _build_state() -> dict:
    topology = _fetch_topology()
    coord_batches = _fetch_coordinator_batches()
    worker_spans = _fetch_worker_spans()

    batches_out = []
    for b in coord_batches:
        bid = b["batch_id"]
        b["workers"] = worker_spans.get(bid, {})
        batches_out.append(b)

    epoch_map: dict[int, list[float]] = {}
    for b in batches_out:
        if b["loss"] > 0:
            epoch_map.setdefault(b["epoch"], []).append(b["loss"])
    epochs_out = [
        {"epoch": e, "avg_loss": round(sum(v) / len(v), 4)}
        for e, v in sorted(epoch_map.items())
    ]

    return {
        "topology": topology,
        "batches": batches_out[-30:],
        "epochs": epochs_out,
    }


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            state = await asyncio.to_thread(_build_state)
            await ws.send_text(json.dumps(state))
            await asyncio.sleep(POLL_INTERVAL)
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
