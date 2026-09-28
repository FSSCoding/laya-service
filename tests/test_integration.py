"""Near-end-to-end integration test.

Verifies the real app wiring: create_app → lifespan → state init → batcher →
request handling → response. Uses a mocked LayaWorker (no GPU) but exercises
the actual async request path through the real RequestBatcher.

This is the test that catches "tests pass but the app wiring is broken."
"""

import pytest
from fastapi.testclient import TestClient
from fixtures import QUESTIONS, STATE


@pytest.fixture
def integration_app(monkeypatch):
    """App with real lifespan and real batcher, but mocked worker.

    TestClient as context manager triggers the lifespan, which:
    - sets app.state.worker (mock via monkeypatch)
    - calls initialize_state() (creates real cache/batcher/stats)
    - starts the real RequestBatcher task
    """
    from conftest import _make_mock_worker, _make_test_cfg

    from laya_server import server as server_mod
    from laya_server import worker as worker_mod
    from laya_server.server import create_app

    mock_w = _make_mock_worker()
    original_worker = worker_mod.LayaWorker
    original_server_worker = server_mod.LayaWorker
    worker_mod.LayaWorker = lambda *a, **kw: mock_w
    server_mod.LayaWorker = lambda *a, **kw: mock_w
    try:
        cfg = _make_test_cfg()
        app = create_app(cfg)
        yield app
    finally:
        worker_mod.LayaWorker = original_worker
        server_mod.LayaWorker = original_server_worker


class TestIntegration:
    def test_full_request_path(self, integration_app):
        """One /system_one request through the real lifespan and real batcher."""
        with TestClient(integration_app) as client:
            # Health should show the model is loaded (mock worker set by lifespan)
            health = client.get("/health").json()
            assert health["model_loaded"] is True

            # Send a real inference request through the full async path
            resp = client.post("/system_one", json={"state": STATE, "questions": QUESTIONS})
            assert resp.status_code == 200
            body = resp.json()
            assert body["model"] == "laya"
            assert "answers" in body
            assert body["cached"] is False

            # Routing signal carried through the full path
            for qid, answer in body["answers"].items():
                rs = answer["rl_agent"]
                assert "act_probability" in rs
                assert "escalate_probability" in rs
                assert rs["experimental"] is True

            # Stats should reflect the request
            stats = client.get("/v1/stats").json()
            assert stats["total_requests"] >= 1
            assert stats["model_loaded"] is True

    def test_lifespan_initializes_state(self, integration_app):
        """Verify lifespan properly initialized all state fields."""
        with TestClient(integration_app):
            st = integration_app.state
            assert hasattr(st, "start_time")
            assert hasattr(st, "worker")
            assert st.worker is not None
            assert hasattr(st, "batcher")
            assert st.batcher is not None
            assert hasattr(st, "stats")
            assert "total_requests" in st.stats
            assert "latency_window" in st.stats


class TestWorkerInitFailure:
    """Regression: worker load failing at startup must not re-instantiate or
    leave state unassigned (which would raise AttributeError on shutdown)."""

    @pytest.fixture
    def failure_app(self, monkeypatch):
        """App whose LayaWorker constructor raises on every call.

        The monkeypatch is held for the whole test so the lifespan (which runs
        inside TestClient) actually sees the failing constructor.
        Returns (app, attempts) where attempts counts constructor invocations.
        """
        from conftest import _make_test_cfg

        from laya_server import server as server_mod
        from laya_server import worker as worker_mod
        from laya_server.server import create_app

        attempts = []

        def _boom(*a, **kw):
            attempts.append(1)
            raise RuntimeError("simulated GPU/model load failure")

        monkeypatch.setattr(worker_mod, "LayaWorker", _boom)
        monkeypatch.setattr(server_mod, "LayaWorker", _boom)

        cfg = _make_test_cfg()
        app = create_app(cfg)
        return app, attempts

    def test_failed_load_not_retried_and_shutdown_clean(self, failure_app):
        app, attempts = failure_app
        # Entering the context runs the lifespan; exiting runs shutdown.
        # With the fix: the worker is constructed exactly ONCE (the failed
        # startup attempt), state (cache/batcher/stats) is still initialized,
        # and shutdown completes without AttributeError.
        with TestClient(app):
            st = app.state
            # Worker load failed, so it stays None (not retried).
            assert st.worker is None
            # The rest of state was still initialized so shutdown can read it.
            assert hasattr(st, "batcher")
            assert hasattr(st, "cache")
            assert hasattr(st, "stats")
        # LayaWorker was attempted exactly once — the failure was not retried
        # by initialize_state().
        assert len(attempts) == 1

    def test_health_reports_degraded_when_worker_failed(self, failure_app):
        app, _ = failure_app
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert health["status"] == "degraded"
            assert health["model_loaded"] is False
