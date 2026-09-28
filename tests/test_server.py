"""HTTP contract tests for laya-service endpoints.

Layer: Route/contract tests. No GPU, no real lifespan, no batcher coalescing.
These verify routing, validation, status codes, and response shape.
Full wiring is covered by test_integration.py.
"""


from fixtures import QUESTIONS, STATE


class TestHealth:
    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert isinstance(body["providers"], list)
        assert len(body["providers"]) > 0
        assert "uptime_s" in body
        assert "version" in body

    def test_health_version_matches_package(self, client):
        from laya_server import __version__

        resp = client.get("/health")
        assert resp.json()["version"] == __version__


class TestSystemOne:
    def test_system_one_returns_answers(self, client):
        resp = client.post("/system_one", json={"state": STATE, "questions": QUESTIONS})
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "laya"
        assert "answers" in body
        assert "usage" in body
        assert body["cached"] is False

    def test_system_one_routing_signal_present(self, client):
        resp = client.post("/system_one", json={"state": STATE, "questions": QUESTIONS})
        body = resp.json()
        for qid, answer in body["answers"].items():
            rs = answer["rl_agent"]
            assert "act_probability" in rs
            assert "escalate_probability" in rs
            assert rs["experimental"] is True

    def test_system_one_rejects_empty_questions(self, client):
        resp = client.post("/system_one", json={"state": STATE, "questions": {}})
        assert resp.status_code == 422

    def test_system_one_rejects_missing_state(self, client):
        resp = client.post("/system_one", json={"questions": QUESTIONS})
        assert resp.status_code == 422

    def test_system_one_mock_worker_called(self, mock_app, client):
        client.post("/system_one", json={"state": STATE, "questions": QUESTIONS})
        # The mock worker's system_one should have been called via the mock batcher
        worker = mock_app.state.worker
        worker.system_one.assert_called()


class TestStats:
    def test_stats_structure(self, client):
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_requests" in body
        assert "last_activity" in body
        assert "cache" in body
        assert "batch_coalesced" in body
        assert "latency_ms" in body
        assert "providers" in body
        assert "model_loaded" in body
        assert "version" in body

    def test_stats_increments_after_request(self, client):
        before = client.get("/v1/stats").json()["total_requests"]
        client.post("/system_one", json={"state": STATE, "questions": QUESTIONS})
        after = client.get("/v1/stats").json()["total_requests"]
        assert after == before + 1


class TestUnload:
    def test_unload_sets_worker_none(self, client, mock_app):
        resp = client.post("/control/unload")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unloaded"
        assert mock_app.state.worker is None

    def test_unload_idempotent(self, client):
        # First unload
        resp1 = client.post("/control/unload")
        assert resp1.json()["status"] == "unloaded"
        # Second unload — worker already None
        resp2 = client.post("/control/unload")
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "not_loaded"
