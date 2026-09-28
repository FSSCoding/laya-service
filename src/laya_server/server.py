"""FastAPI application factory and server runner for laya-service.

This module exposes two public functions:
- create_app(cfg) — build a FastAPI app instance (all state on app.state)
- run_server(config_path=None) — console entry point: load config, start uvicorn

Single uvicorn worker is intentional: the GPU model is pinned to one card via
CUDA_VISIBLE_DEVICES, and running multiple workers would duplicate the ~2.3 GB
VRAM footprint per process.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from ._config import AppConfig, load_config
from .batcher import RequestBatcher
from .result_cache import ResultCache
from .worker import LayaWorker

logger = logging.getLogger("laya-service")


class SystemOneRequest(BaseModel):
    state: Any = Field(..., description="Ticket/email/JSON state to reason about")
    questions: Dict[str, Any] = Field(
        ..., min_length=1, description="qid -> question (choice/score/noul)"
    )


def _make_lifespan(cfg: AppConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st = app.state
        st.start_time = time.time()
        st.worker = None
        st.cache = None
        st.batcher = None
        st.stats = {
            "total_requests": 0,
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "last_latency_ms": None,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_coalesced": 0,
            "latency_window": deque(maxlen=100),
        }

        # Pre-warm the GPU model at startup (avoids first-request latency spike).
        try:
            t0 = time.time()
            st.worker = LayaWorker(
                cfg.model_dir,
                execution_providers=[(p.name, p.options) for p in cfg.gpu.providers],
            )
            logger.info(
                f"Worker ready in {time.time() - t0:.1f}s, providers={st.worker.active_providers}"
            )
        except Exception:
            logger.exception(
                "Failed to initialise model at startup; /system_one will fail until fixed"
            )

        cache_cfg = cfg.cache
        if cache_cfg.enabled:
            st.cache = ResultCache(
                cache_cfg.db_path,
                max_memory_entries=cache_cfg.max_memory_entries,
                ttl_days=cache_cfg.ttl_days,
            )

        loop = asyncio.get_running_loop()
        st.batcher = RequestBatcher(
            max_batch_size=cfg.batcher.max_batch_size,
            max_wait_ms=cfg.batcher.max_wait_ms,
            worker=_make_process_payload(app),
        )
        st.batcher.start(loop)

        reaper_task = None
        if cfg.idle.enabled:
            reaper_task = loop.create_task(_idle_reaper(app, cfg))

        yield

        if reaper_task:
            reaper_task.cancel()
        if st.batcher and st.batcher._task:
            st.batcher._task.cancel()
        if st.cache:
            st.cache.close()
        if st.worker:
            st.worker.release()
            logger.info("Model unloaded from GPU (shutdown)")

    return lifespan


def _make_process_payload(app: FastAPI):
    """Return a function that processes a single payload via the app's worker."""

    def _process(payload: Dict[str, Any]) -> Dict[str, Any]:
        w = app.state.worker
        if w is None:
            raise RuntimeError("Model not loaded")
        return w.system_one(payload["state"], payload["questions"])

    return _process


async def _idle_reaper(app: FastAPI, cfg: AppConfig):
    """Background task: unload the model after idle timeout."""
    while True:
        await asyncio.sleep(60)
        st = app.state
        if not cfg.idle.enabled:
            continue
        w = st.worker
        if w is None:
            continue
        last = st.stats.get("last_activity", "")
        try:
            last_dt = datetime.fromisoformat(last)
            idle = (datetime.now(timezone.utc) - last_dt).total_seconds()
        except (ValueError, TypeError):
            continue
        if idle > cfg.idle.unload_after_s:
            w.release()
            st.worker = None
            logger.info(f"Model unloaded after {int(idle)}s idle (> {cfg.idle.unload_after_s}s)")


def create_app(cfg: AppConfig) -> FastAPI:
    """Build the FastAPI app. All mutable state lives on app.state."""
    app = FastAPI(
        title="Laya GPU Inference Server",
        version=__version__,
        description=(
            "System-1 inference over HTTP using onnxruntime-gpu. "
            "One forward pass returns calibrated probabilities for every typed question."
        ),
        lifespan=_make_lifespan(cfg),
    )

    @app.get("/health")
    async def health():
        st = app.state
        uptime = time.time() - getattr(st, "start_time", time.time())
        w = getattr(st, "worker", None)
        status = "ok" if w is not None else "degraded"
        return {
            "status": status,
            "model_loaded": w is not None,
            "providers": w.active_providers if w else [],
            "uptime_s": round(uptime, 2),
            "version": __version__,
        }

    @app.post("/system_one")
    async def system_one(req: SystemOneRequest):
        st = app.state
        st.stats["total_requests"] += 1
        st.stats["last_activity"] = datetime.now(timezone.utc).isoformat()
        t0 = time.time()

        # Lazy reload: if the model was unloaded by the idle reaper, reload on demand.
        if st.worker is None:
            logger.info("Model unloaded — reloading for incoming request")
            st.worker = LayaWorker(
                cfg.model_dir,
                execution_providers=[(p.name, p.options) for p in cfg.gpu.providers],
            )
            logger.info(
                f"Worker reloaded, providers={st.worker.active_providers}"
            )

        key = ResultCache.key(req.state, req.questions)
        cached = st.cache and st.cache.get(key) if st.cache else None
        if cached is not None:
            st.stats["cache_hits"] += 1
            latency = (time.time() - t0) * 1000
            st.stats["last_latency_ms"] = latency
            st.stats["latency_window"].append(latency)
            return {
                "model": "laya",
                "answers": cached["answers"],
                "usage": cached["usage"],
                "cached": True,
            }

        st.stats["cache_misses"] += 1
        result = await st.batcher.submit(
            {"state": req.state, "questions": req.questions}
        )
        latency = (time.time() - t0) * 1000
        st.stats["last_latency_ms"] = latency
        st.stats["latency_window"].append(latency)

        answers = result["answers"]
        usage = result["usage"]
        if st.cache:
            st.cache.put(key, {"answers": answers, "usage": usage})

        return {
            "model": "laya",
            "answers": answers,
            "usage": usage,
            "cached": False,
        }

    @app.get("/v1/stats")
    async def stats():
        st = app.state
        s = st.stats
        lat = list(s["latency_window"])
        n = len(lat)
        p50 = sorted(lat)[n // 2] if n else 0.0
        p99 = sorted(lat)[max(0, int(n * 0.99) - 1)] if n else 0.0
        w = getattr(st, "worker", None)
        return {
            "total_requests": s["total_requests"],
            "last_activity": s["last_activity"],
            "last_latency_ms": s["last_latency_ms"],
            "cache": {
                "hits": s["cache_hits"],
                "misses": s["cache_misses"],
                "enabled": cfg.cache.enabled,
            },
            "batch_coalesced": s["batch_coalesced"],
            "latency_ms": {
                "p50": round(p50, 1),
                "p99": round(p99, 1),
                "last": round(lat[-1], 1) if n else 0.0,
            },
            "providers": (w.active_providers if w else []),
            "model_loaded": w is not None,
            "uptime_s": round(time.time() - getattr(st, "start_time", time.time()), 2),
            "version": __version__,
        }

    @app.post("/control/unload")
    async def unload():
        st = app.state
        w = st.worker
        if w is None:
            return {"status": "not_loaded"}
        w.release()
        st.worker = None
        logger.info("Model unloaded from GPU (manual)")
        return {"status": "unloaded", "gpu_freed_mb": 2400}

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled error on {request.method} {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    return app


def run_server(config_path: str | None = None) -> None:
    """Console entry point: load config, configure logging, start uvicorn.

    Accepts an optional positional config path (for the console script) or
    --config /path/to/config.json (for python -m laya_server).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Laya GPU Inference Server")
    parser.add_argument(
        "--config", dest="config_path", default=config_path,
        help="Path to config JSON (default: LAYA_CONFIG_PATH env or config/laya_config.json)",
    )
    parser.add_argument(
        "--log-level", dest="log_level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level (default: info)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = load_config(Path(args.config_path) if args.config_path else None)
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=args.log_level)


if __name__ == "__main__":
    run_server()
