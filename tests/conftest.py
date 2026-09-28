"""Shared fixtures for laya-service tests."""

import os
from unittest.mock import MagicMock

import pytest
from fixtures import QUESTIONS, STATE  # noqa: F401 — re-exported for test modules

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/MASTERFOLDER/Services/laya/onnx")


@pytest.fixture(scope="module")
def worker():
    """LayaWorker with CUDA (skips if GPU unavailable)."""
    from laya_server.worker import LayaWorker

    try:
        w = LayaWorker(MODEL_DIR)
    except (ImportError, FileNotFoundError, OSError) as e:
        pytest.skip(f"CUDA/model unavailable: {e}")
    yield w
    w.release()


def _make_mock_worker() -> MagicMock:
    """Create a mock worker that mimics LayaWorker's interface."""
    mock_w = MagicMock()
    mock_w.active_providers = ["CPUExecutionProvider"]
    mock_w.system_one.return_value = {
        "answers": {
            "test_q": {
                "type": "noul",
                "noul": 0.5,
                "rl_agent": {
                    "act_probability": 1.0,
                    "escalate_probability": 0.0,
                    "experimental": True,
                },
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    return mock_w


def _make_test_cfg():
    """Build an AppConfig for tests (bypasses model_dir existence check)."""
    # Use a real directory that exists so the validator passes
    import tempfile

    from laya_server._config import AppConfig, CacheConfig

    cfg = AppConfig(
        model_dir=tempfile.gettempdir(),
        port=11448,
        host="127.0.0.1",
    )
    cfg.cache = CacheConfig(enabled=False)
    return cfg


def _make_mock_batcher(worker: MagicMock):
    """Mock batcher with async submit that calls the worker directly."""

    class _MockBatcher:
        def __init__(self, w):
            self._w = w
            self._task = None

        async def submit(self, payload: dict) -> dict:
            return self._w.system_one(payload["state"], payload["questions"])

    return _MockBatcher(worker)


@pytest.fixture
def mock_app(monkeypatch):
    """App with mocked worker and batcher for HTTP contract tests.

    No GPU, no ONNX, no real lifespan. State initialized via initialize_state()
    helper with pre-set mocks. TestClient used for routing only.
    This tests the HTTP layer (routes, validation, status codes, response shape).
    It does NOT test batcher coalescing or async request path.
    """

    from laya_server import server as server_mod
    from laya_server import worker as worker_mod
    from laya_server._state import initialize_state
    from laya_server.server import create_app

    mock_w = _make_mock_worker()
    mock_b = _make_mock_batcher(mock_w)

    # Monkeypatch LayaWorker in BOTH modules that reference it
    original_worker = worker_mod.LayaWorker
    original_server_worker = server_mod.LayaWorker
    worker_mod.LayaWorker = lambda *a, **kw: mock_w
    server_mod.LayaWorker = lambda *a, **kw: mock_w
    try:
        cfg = _make_test_cfg()
        app = create_app(cfg)

        # Pre-set the mock worker; create_worker=False so initialize_state
        # does not attempt to build a real LayaWorker (the mock is authoritative).
        app.state.worker = mock_w
        app.state.batcher = mock_b
        initialize_state(app, cfg, create_worker=False)

        # Override batcher with our mock (initialize_state created a real one)
        app.state.batcher = mock_b

        yield app
    finally:
        worker_mod.LayaWorker = original_worker
        server_mod.LayaWorker = original_server_worker


@pytest.fixture
def client(mock_app):
    """TestClient bound to the mock app."""
    from fastapi.testclient import TestClient

    with TestClient(mock_app) as c:
        yield c
