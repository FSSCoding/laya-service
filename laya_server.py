#!/usr/bin/env python3
"""
Laya GPU Inference Server
=========================

ONNX (onnxruntime-gpu) Laya inference exposed over HTTP on 0.0.0.0:11448.
Topology mirrors /MasterFOLDER/Services/embed-server: single process, model resident on a pinned
3090, a request batcher that coalesces concurrent calls into one forward pass, an inference lock,
a two-tier result cache, and OpenAI-style error envelope.

Endpoints:
  POST /system_one   — Laya systemOne({state, questions}) -> {model, answers, usage}
  GET  /health       — liveness + active providers + uptime
  GET  /v1/stats     — request/latency/cache/provider stats
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("laya-service")

_SERVER_ROOT = Path(__file__).parent
_CONFIG_PATH = Path(os.environ.get("LAYA_CONFIG_PATH", str(_SERVER_ROOT / "config" / "laya_config.json")))


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


CONFIG = _load_config()
SERVER_HOST: str = CONFIG.get("host", "0.0.0.0")
SERVER_PORT: int = CONFIG.get("port", 11448)
MODEL_DIR: str = CONFIG.get("model_dir", "/MASTERFOLDER/Services/laya/onnx")
_BATCHER_CFG = CONFIG.get("batcher", {})
_BATCHER_MAX = _BATCHER_CFG.get("max_batch_size", 512)
_BATCHER_WAIT = _BATCHER_CFG.get("max_wait_ms", 5.0)
_CACHE_CFG = CONFIG.get("cache", {})
_CACHE_ENABLED = _CACHE_CFG.get("enabled", True)
_CACHE_DB = _CACHE_CFG.get("db_path", "~/.laya-service/result_cache.db")
_CACHE_MAX = _CACHE_CFG.get("max_memory_entries", 10000)
_CACHE_TTL = _CACHE_CFG.get("ttl_days", 30)
_IDLE_CFG = CONFIG.get("idle", {})
_IDLE_UNLOAD_S: int = _IDLE_CFG.get("unload_after_s", 300)
_IDLE_ENABLED: bool = _IDLE_CFG.get("enabled", True)
_GPU_CFG = CONFIG.get("gpu", {})
_GPU_PROVIDERS = [ (p["name"], p.get("options", {})) for p in _GPU_CFG.get("providers", []) ] or [
    ("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {}),
]

# Make src/ importable as a package
import sys

sys.path.insert(0, str(_SERVER_ROOT))

from src.batcher import RequestBatcher  # noqa: E402
from src.result_cache import ResultCache  # noqa: E402
from src.worker import LayaWorker  # noqa: E402

_START_TIME = time.time()
_worker: Optional[LayaWorker] = None
_worker_lock = threading.Lock()
_cache: Optional[ResultCache] = None
_batcher: Optional[RequestBatcher] = None
_stats_lock = threading.Lock()
_stats: Dict[str, Any] = {
    "requests_total": 0,
    "errors_total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "batch_coalesced": 0,
    "latency_ms": deque(maxlen=1000),
}
# Idle-tracking for auto-unload (mirrors the Chatterbox model-daemon TTL pattern).
_last_activity = time.time()
_activity_lock = threading.Lock()


def _touch_activity() -> None:
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def _elapsed_idle() -> float:
    with _activity_lock:
        return time.time() - _last_activity


def get_worker() -> LayaWorker:
    """Lazy, thread-safe singleton worker. Loaded on demand; reloaded after an idle-unload."""
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                logger.info(f"Loading Laya ONNX worker from {MODEL_DIR} ...")
                t0 = time.time()
                _worker = LayaWorker(MODEL_DIR, execution_providers=_GPU_PROVIDERS)
                logger.info(f"Worker ready in {time.time() - t0:.1f}s, providers={_worker.active_providers}")
    return _worker


def _process_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Blocking batcher callback: optional cache lookup + one systemOne call."""
    global _cache
    state = payload.get("state")
    questions = payload.get("questions") or {}
    if not questions:
        raise ValueError("questions: at least one question is required")

    _touch_activity()
    ck = None
    if _CACHE_ENABLED and _cache is not None:
        ck = ResultCache.key(state, questions)
        hit = _cache.get(ck)
        if hit is not None:
            with _stats_lock:
                _stats["cache_hits"] += 1
            return hit
        with _stats_lock:
            _stats["cache_misses"] += 1
    w = get_worker()
    res = w.system_one(state, questions)
    if _CACHE_ENABLED and _cache is not None and ck is not None:
        _cache.put(ck, res)
    return res


def unload_worker() -> None:
    """Release the ONNX session and free VRAM (idle-unload). Re-loaded lazily on next request."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.release()
            _worker = None
            logger.info("Worker unloaded (idle) — VRAM freed.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SystemOneRequest(BaseModel):
    state: Any = Field(..., description="Ticket/email/JSON state to reason about")
    questions: Dict[str, Any] = Field(..., min_length=1, description="qid -> question (choice/score/noul)")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
async def _idle_reaper() -> None:
    """Periodically unload the model after it has been idle for _IDLE_UNLOAD_S.

    Mirrors the Chatterbox model-daemon TTL: frees the ~2.3 GB VRAM when the service is not in
    use, and the model is transparently re-loaded (lazy) on the next request.
    """
    if not _IDLE_ENABLED or _IDLE_UNLOAD_S <= 0:
        return
    while True:
        try:
            await asyncio.sleep(15)
            if _worker is not None and _elapsed_idle() >= _IDLE_UNLOAD_S:
                await asyncio.get_running_loop().run_in_executor(None, unload_worker)
        except asyncio.CancelledError:
            return


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _cache, _batcher
    if _CACHE_ENABLED:
        _cache = ResultCache(_CACHE_DB, max_memory_entries=_CACHE_MAX, ttl_days=_CACHE_TTL)
    loop = asyncio.get_running_loop()
    _batcher = RequestBatcher(max_batch_size=_BATCHER_MAX, max_wait_ms=_BATCHER_WAIT, worker=_process_payload)
    _batcher.start(loop)
    reaper = loop.create_task(_idle_reaper())
    # Pre-warm the model so the first real request does not pay the load penalty.
    logger.info("Pre-warming model ...")
    try:
        await loop.run_in_executor(None, get_worker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Pre-warm failed (will retry on first request): {exc}")
    yield
    reaper.cancel()
    if _batcher is not None:
        _batcher._task.cancel()
    if _cache is not None:
        _cache.close()
    if _worker is not None:
        _worker.release()


app = FastAPI(title="Laya GPU Inference Server", version="1.0.0", lifespan=_lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": {"message": str(exc.errors()), "type": "invalid_request_error", "param": None, "code": None}},
    )


@app.post("/system_one")
async def system_one(req: SystemOneRequest, request: Request):
    req_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    _touch_activity()
    with _stats_lock:
        _stats["requests_total"] += 1
    try:
        res = await _batcher.submit({"state": req.state, "questions": req.questions})
    except Exception as exc:  # noqa: BLE001
        with _stats_lock:
            _stats["errors_total"] += 1
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "server_error", "param": None, "code": None}},
        )
    dt = (time.time() - t0) * 1000
    with _stats_lock:
        _stats["latency_ms"].append(dt)
    resp_headers = {"X-Request-Id": req_id, "X-Latency-Ms": f"{dt:.1f}"}
    return JSONResponse(content=res, headers=resp_headers)


@app.get("/health")
async def health():
    w = _worker
    idle = _elapsed_idle()
    return {
        "status": "ok",
        "model": "laya",
        "providers": (w.active_providers if w else []),
        "model_loaded": w is not None,
        "idle_s": round(idle, 1),
        "idle_unload_after_s": _IDLE_UNLOAD_S if _IDLE_ENABLED else 0,
        "uptime_s": round(time.time() - _START_TIME, 1),
    }


@app.post("/control/unload")
async def control_unload():
    """Explicitly unload the model and free VRAM (no request needed). Idempotent."""
    await asyncio.get_running_loop().run_in_executor(None, unload_worker)
    return {"status": "unloaded", "model_loaded": _worker is not None}


@app.get("/v1/stats")
async def stats():
    with _stats_lock:
        lat = list(_stats["latency_ms"])
        s = {k: (list(v) if isinstance(v, deque) else v) for k, v in _stats.items()}
    lat_sorted = sorted(lat)
    n = len(lat_sorted)
    p50 = lat_sorted[n // 2] if n else 0.0
    p99 = lat_sorted[min(n - 1, int(n * 0.99))] if n else 0.0
    return {
        "requests_total": s["requests_total"],
        "errors_total": s["errors_total"],
        "cache": {"hits": s["cache_hits"], "misses": s["cache_misses"], "enabled": _CACHE_ENABLED},
        "batch_coalesced": s["batch_coalesced"],
        "latency_ms": {"p50": round(p50, 1), "p99": round(p99, 1), "last": round(lat[-1], 1) if n else 0.0},
        "providers": (_worker.active_providers if _worker else []),
        "model_loaded": _worker is not None,
        "idle_s": round(_elapsed_idle(), 1),
        "uptime_s": round(time.time() - _START_TIME, 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
