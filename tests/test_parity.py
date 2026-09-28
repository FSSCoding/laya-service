"""Parity gate: GPU (onnxruntime-gpu) outputs match the CPU reference.

Acceptance bar:
  * input_tokens must be EXACT (deterministic integer) -> 267
  * the model DECISION is invariant: same choice, same argmax
  * per-probability values within TOL (CPU vs CUDA FP32 non-determinism ~1e-4)
"""
import pytest
from conftest import QUESTIONS, STATE

TOL = 1e-3


class TestParity:
    def test_input_tokens_exact(self, worker):
        r = worker.system_one(STATE, QUESTIONS)
        assert r["usage"]["input_tokens"] == 267

    def test_department_choice_invariant(self, worker):
        r = worker.system_one(STATE, QUESTIONS)
        a = r["answers"]
        assert a["department"]["choice"] == "billing"

    def test_department_probabilities(self, worker):
        r = worker.system_one(STATE, QUESTIONS)
        a = r["answers"]["department"]["probabilities"]
        assert a["billing"] == pytest.approx(0.9415, abs=TOL)
        assert a["support"] == pytest.approx(0.031, abs=TOL)
        assert a["sales"] == pytest.approx(0.0275, abs=TOL)

    def test_department_confidence(self, worker):
        r = worker.system_one(STATE, QUESTIONS)
        assert r["answers"]["department"]["confidence"] == pytest.approx(0.7603, abs=TOL)

    def test_urgency_score(self, worker):
        r = worker.system_one(STATE, QUESTIONS)
        assert r["answers"]["urgency"]["score"] == pytest.approx(1.3886, abs=TOL)

    def test_urgency_probabilities(self, worker):
        r = worker.system_one(STATE, QUESTIONS)
        p = r["answers"]["urgency"]["probabilities"]
        assert p["0"] == pytest.approx(0.1752, abs=TOL)
        assert p["1"] == pytest.approx(0.2947, abs=TOL)
        assert p["2"] == pytest.approx(0.4962, abs=TOL)
        assert p["3"] == pytest.approx(0.0338, abs=TOL)

    def test_churn_noul(self, worker):
        r = worker.system_one(STATE, QUESTIONS)
        assert r["answers"]["churn_risk"]["noul"] == pytest.approx(0.0988, abs=TOL)
