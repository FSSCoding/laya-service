"""Shared fixtures for laya-service tests."""

import os

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
