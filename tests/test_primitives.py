"""Primitives tests — native wrappers, batch, and composed helpers.

All tests mock ``client.system_one`` to avoid needing a running service.
The parity gate (test_parity.py) covers the real GPU path.
"""
from __future__ import annotations

from unittest.mock import patch

from laya_server.primitives import (
    BatchResult,
    ChoiceResult,
    NoulResult,
    RoutingSignal,
    ScoreResult,
    classify,
    laya_batch,
    laya_choice,
    laya_noul,
    laya_score,
    multi_select,
    rank,
)

MOCK_ROUTING = {
    "act_probability": 0.95,
    "escalate_probability": 0.05,
    "experimental": True,
}


def _mock_resp(answers: dict) -> dict:
    return {
        "model": "laya",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 0},
        "cached": False,
    }


def _choice_answer(**overrides) -> dict:
    base = {
        "type": "choice",
        "choice": "A",
        "probabilities": {"A": 0.7, "B": 0.2, "C": 0.1},
        "confidence": 0.55,
        "rl_agent": dict(MOCK_ROUTING),
    }
    base.update(overrides)
    return base


def _score_answer(**overrides) -> dict:
    base = {
        "type": "score",
        "score": 1.5,
        "legend": {"0": "low", "1": "med", "2": "high"},
        "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
        "confidence": 0.3,
        "rl_agent": dict(MOCK_ROUTING),
    }
    base.update(overrides)
    return base


def _noul_answer(noul: float = 0.73, **overrides) -> dict:
    base = {
        "type": "noul",
        "noul": noul,
        "rl_agent": dict(MOCK_ROUTING),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Native wrappers
# ---------------------------------------------------------------------------

class TestLayaChoice:
    def test_returns_choice_result(self):
        resp = _mock_resp({"q": _choice_answer()})
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = laya_choice({"x": 1}, "pick", {"A": "first", "B": "second", "C": "third"})
        assert isinstance(result, ChoiceResult)
        assert result.choice == "A"
        assert result.probabilities == {"A": 0.7, "B": 0.2, "C": 0.1}
        assert result.confidence == 0.55

    def test_routing_signal_present_and_experimental(self):
        resp = _mock_resp({"q": _choice_answer()})
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = laya_choice({"x": 1}, "pick", ["A", "B", "C"])
        assert isinstance(result.routing_signal, RoutingSignal)
        assert result.routing_signal.experimental is True
        assert result.routing_signal.act_probability == 0.95
        assert result.routing_signal.escalate_probability == 0.05

    def test_list_options_converted_to_dict(self):
        resp = _mock_resp({"q": _choice_answer()})
        with patch("laya_server.primitives.system_one", return_value=resp) as mock:
            laya_choice({"x": 1}, "pick", ["A", "B"])
        call_args = mock.call_args
        questions = call_args[0][2]
        criteria = questions["q"]["criteria"]
        assert criteria == {"A": None, "B": None}


class TestLayaScore:
    def test_returns_score_result(self):
        resp = _mock_resp({"q": _score_answer()})
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = laya_score({"x": 1}, "rate", ["low", "med", "high"])
        assert isinstance(result, ScoreResult)
        assert result.score == 1.5
        assert result.legend == {"0": "low", "1": "med", "2": "high"}

    def test_routing_signal_experimental(self):
        resp = _mock_resp({"q": _score_answer()})
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = laya_score({"x": 1}, "rate", ["low", "med", "high"])
        assert result.routing_signal.experimental is True


class TestLayaNoul:
    def test_returns_noul_result(self):
        resp = _mock_resp({"q": _noul_answer(0.82)})
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = laya_noul({"x": 1}, "Is this true?")
        assert isinstance(result, NoulResult)
        assert result.noul == 0.82

    def test_routing_signal_experimental(self):
        resp = _mock_resp({"q": _noul_answer()})
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = laya_noul({"x": 1}, "Is this true?")
        assert result.routing_signal.experimental is True
        assert result.routing_signal.act_probability == 0.95

    def test_optional_descs(self):
        resp = _mock_resp({"q": _noul_answer()})
        with patch("laya_server.primitives.system_one", return_value=resp) as mock:
            laya_noul({"x": 1}, "Is this true?", true_desc="yes it is", false_desc="no it isnt")
        call_args = mock.call_args
        criteria = call_args[0][2]["q"]["criteria"]
        assert criteria["true"] == "yes it is"
        assert criteria["false"] == "no it isnt"


# ---------------------------------------------------------------------------
# Batch primitive
# ---------------------------------------------------------------------------

class TestLayaBatch:
    def test_mixed_types(self):
        resp = _mock_resp({
            "c": _choice_answer(),
            "s": _score_answer(),
            "n": _noul_answer(),
        })
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = laya_batch({"x": 1}, {
                "c": {"type": "choice", "instructions": "pick", "criteria": {"A": None}},
                "s": {"type": "score", "instructions": "rate", "criteria": ["lo", "hi"]},
                "n": {"type": "noul", "instructions": "is it?"},
            })
        assert isinstance(result, BatchResult)
        assert isinstance(result.answers["c"], ChoiceResult)
        assert isinstance(result.answers["s"], ScoreResult)
        assert isinstance(result.answers["n"], NoulResult)
        assert result.usage == {"input_tokens": 100, "output_tokens": 0}
        assert result.cached is False

    def test_single_question(self):
        resp = _mock_resp({"q": _noul_answer(0.6)})
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = laya_batch({"x": 1}, {"q": {"type": "noul", "instructions": "?"}})
        assert len(result.answers) == 1
        assert isinstance(result.answers["q"], NoulResult)
        assert result.answers["q"].noul == 0.6


# ---------------------------------------------------------------------------
# Composed helpers
# ---------------------------------------------------------------------------

class TestClassify:
    def test_is_sugar_over_choice(self):
        resp = _mock_resp({"q": _choice_answer(choice="urgent")})
        with patch("laya_server.primitives.system_one", return_value=resp) as mock:
            result = classify({"x": 1}, "Classify this", ["urgent", "normal", "low"])
        assert isinstance(result, ChoiceResult)
        assert result.choice == "urgent"
        call_args = mock.call_args
        questions = call_args[0][2]
        assert questions["q"]["type"] == "choice"


class TestMultiSelect:
    def test_returns_probability_dict(self):
        resp = _mock_resp({
            "c0": _noul_answer(0.9),
            "c1": _noul_answer(0.3),
            "c2": _noul_answer(0.7),
        })
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = multi_select(
                {"x": 1}, "Which apply?", ["fast", "cheap", "good"],
            )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"fast", "cheap", "good"}
        assert result["fast"] == 0.9
        assert result["cheap"] == 0.3
        assert result["good"] == 0.7

    def test_threshold_not_applied(self):
        resp = _mock_resp({
            "c0": _noul_answer(0.9),
            "c1": _noul_answer(0.1),
        })
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = multi_select(
                {"x": 1}, "Which?", ["a", "b"], threshold=0.5,
            )
        assert len(result) == 2
        assert "b" in result
        assert result["b"] == 0.1

    def test_batched_in_one_call(self):
        resp = _mock_resp({
            "c0": _noul_answer(0.8),
            "c1": _noul_answer(0.6),
        })
        with patch("laya_server.primitives.system_one", return_value=resp) as mock:
            multi_select({"x": 1}, "Which?", ["a", "b"])
        call_args = mock.call_args
        questions = call_args[0][2]
        assert len(questions) == 2
        assert questions["c0"]["type"] == "noul"
        assert questions["c1"]["type"] == "noul"


class TestRank:
    def test_returns_sorted_pairs(self):
        resp = _mock_resp({
            "item_0": _score_answer(score=0.5, probabilities={"0": 0.8, "1": 0.2}),
            "item_1": _score_answer(score=2.0, probabilities={"0": 0.1, "1": 0.9}),
            "item_2": _score_answer(score=1.0, probabilities={"0": 0.3, "1": 0.7}),
        })
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = rank(
                {"x": 1}, "Urgency", ["alpha", "beta", "gamma"],
                rubric=["low", "high"],
            )
        assert isinstance(result, list)
        assert len(result) == 3
        items = [pair[0] for pair in result]
        assert items == ["beta", "gamma", "alpha"]
        scores = [pair[1].score for pair in result]
        assert scores == [2.0, 1.0, 0.5]

    def test_preserves_full_distribution(self):
        resp = _mock_resp({
            "item_0": _score_answer(
                score=1.0,
                legend={"0": "low", "1": "high"},
                probabilities={"0": 0.3, "1": 0.7},
            ),
        })
        with patch("laya_server.primitives.system_one", return_value=resp):
            result = rank({"x": 1}, "Rate", ["only"], rubric=["low", "high"])
        item, score_result = result[0]
        assert item == "only"
        assert isinstance(score_result, ScoreResult)
        assert score_result.probabilities == {"0": 0.3, "1": 0.7}
        assert score_result.legend == {"0": "low", "1": "high"}

    def test_shared_rubric_in_instructions(self):
        resp = _mock_resp({
            "item_0": _score_answer(),
            "item_1": _score_answer(),
        })
        with patch("laya_server.primitives.system_one", return_value=resp) as mock:
            rank({"x": 1}, "Urgency", ["alpha", "beta"], rubric=["low", "med", "high"])
        call_args = mock.call_args
        questions = call_args[0][2]
        assert questions["item_0"]["instructions"] == "Urgency — alpha"
        assert questions["item_1"]["instructions"] == "Urgency — beta"
        assert questions["item_0"]["criteria"] == ["low", "med", "high"]
        assert questions["item_1"]["criteria"] == ["low", "med", "high"]


# ---------------------------------------------------------------------------
# Inversion comparison (documented pattern, no helper)
# ---------------------------------------------------------------------------

class TestInversionComparison:
    def test_choice_vs_noul_inversion(self):
        """Pose the same problem two ways: multiclass choice vs. many binary noul.

        Both return valid results; the test does NOT assert they are identical
        (they are not probabilistically equivalent).
        """
        choice_resp = _mock_resp({"q": _choice_answer(choice="A")})
        noul_resp = _mock_resp({
            "c0": _noul_answer(0.8),
            "c1": _noul_answer(0.2),
        })

        with patch("laya_server.primitives.system_one", return_value=choice_resp):
            choice_result = laya_choice({"x": 1}, "Which?", ["A", "B"])
        with patch("laya_server.primitives.system_one", return_value=noul_resp):
            noul_result = multi_select({"x": 1}, "Which?", ["A", "B"])

        assert isinstance(choice_result, ChoiceResult)
        assert choice_result.choice == "A"
        assert isinstance(noul_result, dict)
        assert set(noul_result.keys()) == {"A", "B"}
        for v in noul_result.values():
            assert 0.0 <= v <= 1.0
