"""App state initialization — single source of truth for app.state shape.

Used by production lifespan (server.py) and test fixtures. If a new state
field is added here, both paths pick it up automatically.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone

from fastapi import FastAPI

from ._config import AppConfig
from .batcher import RequestBatcher
from .result_cache import ResultCache
from .worker import LayaWorker


def initialize_state(app: FastAPI, cfg: AppConfig, create_worker: bool = True) -> None:
    """Populate app.state for a running Laya service instance.

    Called by the production lifespan (with real LayaWorker) and by test
    fixtures (with a mocked worker).

    create_worker: when True (test fixtures) and app.state.worker is not yet
    set, a real LayaWorker is created. The production lifespan always
    pre-sets app.state.worker (real worker, or None when the startup load
    failed) and passes create_worker=False so a failed load is NOT silently
    retried here — that would re-run the exact construction that just raised.
    """
    st = app.state
    st.start_time = time.time()

    # Worker: created only when the caller opted in AND no worker is set yet.
    if create_worker and (not hasattr(st, "worker") or st.worker is None):
        st.worker = LayaWorker(
            cfg.model_dir,
            execution_providers=[(p.name, p.options) for p in cfg.gpu.providers],
        )

    # Cache
    cache_cfg = cfg.cache
    st.cache = None
    if cache_cfg.enabled:
        st.cache = ResultCache(
            cache_cfg.db_path,
            max_memory_entries=cache_cfg.max_memory_entries,
            ttl_days=cache_cfg.ttl_days,
        )

    # Batcher — needs a process function that reads the worker from app.state
    def _on_coalesce(n: int) -> None:
        st.stats["batch_coalesced"] += n

    st.batcher = RequestBatcher(
        max_batch_size=cfg.batcher.max_batch_size,
        max_wait_ms=cfg.batcher.max_wait_ms,
        worker=_make_process_fn(app),
        worker_many=_make_process_many_fn(app),
        on_coalesce=_on_coalesce,
    )

    # Stats
    st.stats = {
        "total_requests": 0,
        "last_activity": datetime.now(timezone.utc).isoformat(),
        "last_latency_ms": None,
        "cache_hits": 0,
        "cache_misses": 0,
        "batch_coalesced": 0,
        "latency_window": deque(maxlen=100),
    }


def _make_process_fn(app: FastAPI):
    """Return a sync callable that processes a single payload via app.state.worker."""

    def _process(payload: dict) -> dict:
        w = app.state.worker
        if w is None:
            raise RuntimeError("Model not loaded")
        return w.system_one(payload["state"], payload["questions"])

    return _process


def _make_process_many_fn(app: FastAPI):
    """Return a sync callable that processes multiple payloads in one forward pass."""

    def _process_many(pairs: list) -> list:
        w = app.state.worker
        if w is None:
            raise RuntimeError("Model not loaded")
        return w.system_many(pairs)

    return _process_many
