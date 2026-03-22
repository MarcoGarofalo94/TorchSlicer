"""
TorchSlicer Monitor — server-side accumulator edition.

Jaeger data is merged into an in-memory store on every poll so no history is
ever lost between requests. The WebSocket sends the full accumulated state.

Env vars:
  JAEGER_HTTP      Jaeger base URL     (default: http://localhost:16686)
  DASHBOARD_PORT   Port to listen on   (default: 8080)
  POLL_INTERVAL    WS push period (s)  (default: 2.0)
"""

import asyncio
import json
import os
import threading

import requests as _requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

JAEGER        = os.environ.get("JAEGER_HTTP",    "http://localhost:16686")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "2"))

app   = FastAPI(title="TorchSlicer Monitor")
_here = os.path.dirname(os.path.abspath(__file__))

# ── In-memory accumulator ──────────────────────────────────────────────────────
_lock          = threading.Lock()
_batches: dict[int, dict] = {}   # batch_id → batch dict (with workers merged in)
_topology: list[dict]     = []
_topology_ready           = False


# ── Jaeger helpers ─────────────────────────────────────────────────────────────

def _get(path: str, **params) -> dict:
    try:
        r = _requests.get(f"{JAEGER}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def _tags(span: dict) -> dict:
    return {t["key"]: t["value"] for t in span.get("tags", [])}

def _ms(span: dict) -> float:
    return round(span.get("duration", 0) / 1_000, 2)


# ── Fetchers ───────────────────────────────────────────────────────────────────

def _fetch_topology() -> list[dict]:
    data = _get("/api/traces", service="torchslicer-worker",
                operation="torchslicer.worker.init", limit=50)
    workers: dict[str, dict] = {}
    for trace in data.get("data", []):
        for span in trace.get("spans", []):
            if span.get("operationName") != "torchslicer.worker.init":
                continue
            tags = _tags(span)
            host = tags.get("worker", "unknown")
            if host not in workers:
                raw    = tags.get("layers", "")
                layers = [l.strip().strip("'\"") for l in raw.split(",") if l.strip()]
                workers[host] = {
                    "hostname":     host,
                    "layers":       layers,
                    "n_layers":     int(tags.get("n_layers", len(layers))),
                    "is_last":      tags.get("is_last") in (True, "True", "true"),
                    "prev":         tags.get("prev_worker") or None,
                    "next":         tags.get("next_worker") or None,
                    "param_mb":     float(tags.get("param_mb", 0.0)),
                    "cuda_alloc_mb": float(tags.get("cuda_alloc_mb", 0.0)),
                }

    # Order chain: start from worker with no prev
    ordered, visited = [], set()
    by_host = dict(workers)
    cur = next((w for w in by_host.values() if not w["prev"]), None)
    while cur and cur["hostname"] not in visited:
        ordered.append(cur)
        visited.add(cur["hostname"])
        if cur["is_last"]:
            break
        # find next unvisited worker whose prev matches cur's next addr
        nxt_addr = cur.get("next") or ""
        nxt = next((w for w in by_host.values()
                    if w["hostname"] not in visited
                    and w.get("prev") == nxt_addr), None)
        if not nxt:  # fallback: any unvisited
            nxt = next((w for w in by_host.values() if w["hostname"] not in visited), None)
        cur = nxt
    for w in by_host.values():
        if w not in ordered:
            ordered.append(w)
    return ordered


def _fetch_coordinator_batches() -> list[dict]:
    # limit=200 epoch-level traces → each trace holds all batch child spans
    data = _get("/api/traces", service="torchslicer-coordinator",
                operation="torchslicer.epoch", limit=200)
    batches = []
    for trace in data.get("data", []):
        for span in trace.get("spans", []):
            if span.get("operationName") != "torchslicer.batch":
                continue
            tags = _tags(span)
            batches.append({
                "batch_id": int(tags.get("batch_id", -1)),
                "epoch":    int(tags.get("epoch", 0)),
                "loss":     float(tags.get("loss", 0.0)),
                "total_ms": _ms(span),
                "start_us": span.get("startTime", 0),
            })
    return batches


def _fetch_worker_spans() -> dict[int, dict]:
    """Returns {batch_id: {hostname: {fwd_ms, bwd_ms, ...}}}"""
    result: dict[int, dict] = {}
    for op, key in [
        ("torchslicer.worker.forward",  "fwd_ms"),
        ("torchslicer.worker.backward", "bwd_ms"),
    ]:
        data = _get("/api/traces", service="torchslicer-worker",
                    operation=op, limit=2000)
        for trace in data.get("data", []):
            for span in trace.get("spans", []):
                if span.get("operationName") != op:
                    continue
                tags = _tags(span)
                bid  = int(tags.get("batch_id", -1))
                host = tags.get("worker", "unknown")
                if bid < 0:
                    continue
                node = result.setdefault(bid, {}).setdefault(host, {})
                node[key] = _ms(span)
                if key == "fwd_ms":
                    node["input_shape"]  = tags.get("input_shape", "")
                    node["output_shape"] = tags.get("output_shape", "")
                    node["is_last"]      = tags.get("is_last") in (True, "True", "true")
    return result


# ── Merge into accumulator ─────────────────────────────────────────────────────

def _poll_and_merge() -> None:
    global _topology, _topology_ready

    if not _topology_ready:
        topo = _fetch_topology()
        if topo:
            with _lock:
                _topology = topo
                _topology_ready = True

    coord_batches = _fetch_coordinator_batches()
    worker_spans  = _fetch_worker_spans()

    with _lock:
        for b in coord_batches:
            bid = b["batch_id"]
            existing_workers = _batches.get(bid, {}).get("workers", {})
            # Merge: new worker spans override old; keep keys missing from new fetch
            merged_workers = {**existing_workers, **worker_spans.get(bid, {})}
            b["workers"] = merged_workers
            _batches[bid] = b


def _get_full_state() -> dict:
    _poll_and_merge()
    with _lock:
        batches = sorted(_batches.values(), key=lambda b: (b["epoch"], b["batch_id"]))
        epoch_map: dict[int, list[float]] = {}
        for b in batches:
            if b.get("loss", 0) > 0:
                epoch_map.setdefault(b["epoch"], []).append(b["loss"])
        epochs = [
            {"epoch": e, "avg_loss": round(sum(v) / len(v), 4), "n_batches": len(v)}
            for e, v in sorted(epoch_map.items())
        ]
        return {"topology": _topology, "batches": batches, "epochs": epochs}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/state")
async def api_state():
    state = await asyncio.to_thread(_get_full_state)
    return JSONResponse(state)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            state = await asyncio.to_thread(_get_full_state)
            await ws.send_text(json.dumps(state))
            await asyncio.sleep(POLL_INTERVAL)
    except (WebSocketDisconnect, Exception):
        pass


# Serve the Vite build — must be LAST so API/WS routes take priority
app.mount("/", StaticFiles(directory=os.path.join(_here, "static"), html=True), name="spa")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
