"""CLI contract tests — verifies stable JSON output schemas.

These tests do NOT require a running service. They use the click test runner
with mocked client calls to verify output structure.
"""

import json
from unittest.mock import patch

from click.testing import CliRunner

from laya_server.cli import cli

MOCK_ROUTING = {
    "act_probability": 0.95,
    "escalate_probability": 0.05,
    "experimental": True,
}


class TestVersion:
    def test_version_output(self):
        from laya_server import __version__

        result = CliRunner().invoke(cli, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestStatusJsonSchema:
    """Verify the frozen --json schema for `laya status`.

    The schema is:
    {
      "systemd": "active" | "inactive" | "unknown",
      "health": {...} | null,
      "endpoint": str
    }
    """

    def test_status_json_top_level_keys(self, monkeypatch):
        from unittest.mock import patch

        # Mock systemctl to return "inactive" (no service running)
        with patch("laya_server.cli._run_systemctl", return_value=(3, "inactive")):
            result = CliRunner().invoke(cli, ["status", "--json"])
            # Exit code 3 = stopped
            assert result.exit_code == 3
            data = json.loads(result.output)
            assert set(data.keys()) == {"systemd", "health", "endpoint"}
            assert data["systemd"] == "inactive"
            assert data["health"] is None
            assert data["endpoint"].startswith("http://")

    def test_status_json_systemd_values(self, monkeypatch):
        from unittest.mock import patch

        # Mock systemctl to return "unknown"
        with patch("laya_server.cli._run_systemctl", return_value=(-1, "systemctl not found")):
            with patch("laya_server.cli.health", side_effect=ConnectionError("unreachable")):
                result = CliRunner().invoke(cli, ["status", "--json"])
                assert result.exit_code == 2
                data = json.loads(result.output)
                assert data["systemd"] in {"active", "inactive", "unknown"}

    def test_status_json_health_null_when_unreachable(self, monkeypatch):
        from unittest.mock import patch

        # Active but health fails
        with patch("laya_server.cli._run_systemctl", return_value=(0, "active")):
            with patch("laya_server.cli.health", side_effect=ConnectionError("refused")):
                result = CliRunner().invoke(cli, ["status", "--json"])
                assert result.exit_code == 2
                data = json.loads(result.output)
                assert data["systemd"] == "active"
                assert data["health"] is None


class TestHealthExitCodes:
    def test_health_exit_0_when_ok(self, monkeypatch):
        from unittest.mock import patch

        fake_health = {"status": "ok", "model_loaded": True, "providers": ["CUDAExecutionProvider"]}
        with patch("laya_server.cli.health", return_value=fake_health):
            result = CliRunner().invoke(cli, ["health", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["status"] == "ok"

    def test_health_exit_1_when_degraded(self, monkeypatch):
        from unittest.mock import patch

        fake_health = {"status": "degraded", "model_loaded": False, "providers": []}
        with patch("laya_server.cli.health", return_value=fake_health):
            result = CliRunner().invoke(cli, ["health", "--json"])
            assert result.exit_code == 1

    def test_health_exit_2_when_unreachable(self, monkeypatch):
        from unittest.mock import patch

        with patch("laya_server.cli.health", side_effect=ConnectionError("refused")):
            result = CliRunner().invoke(cli, ["health", "--json"])
            assert result.exit_code == 2


MOCK_CHOICE_RESP = {
    "model": "laya",
    "answers": {
        "q": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "support": 0.1},
            "confidence": 0.6,
            "rl_agent": MOCK_ROUTING,
        }
    },
    "usage": {"input_tokens": 50, "output_tokens": 0},
    "cached": False,
}

MOCK_SCORE_RESP = {
    "model": "laya",
    "answers": {
        "q": {
            "type": "score",
            "score": 1.5,
            "legend": {"0": "low", "1": "med", "2": "high"},
            "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
            "confidence": 0.3,
            "rl_agent": MOCK_ROUTING,
        }
    },
    "usage": {"input_tokens": 50, "output_tokens": 0},
    "cached": False,
}

MOCK_NOUL_RESP = {
    "model": "laya",
    "answers": {
        "q": {
            "type": "noul",
            "noul": 0.73,
            "rl_agent": MOCK_ROUTING,
        }
    },
    "usage": {"input_tokens": 50, "output_tokens": 0},
    "cached": False,
}


class TestChoiceCommand:
    def test_choice_json_output(self):
        with patch("laya_server.primitives.system_one", return_value=MOCK_CHOICE_RESP):
            result = CliRunner().invoke(cli, [
                "choice",
                "--state", '{"subject":"Refund"}',
                "--instructions", "Which team?",
                "--options", '{"billing":"payments","support":"help"}',
                "--json",
            ])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["choice"] == "billing"
            assert data["routing_signal"]["experimental"] is True
            assert data["routing_signal"]["act_probability"] == 0.95
            assert data["routing_signal"]["escalate_probability"] == 0.05

    def test_choice_text_output(self):
        with patch("laya_server.primitives.system_one", return_value=MOCK_CHOICE_RESP):
            result = CliRunner().invoke(cli, [
                "choice",
                "--state", '{"subject":"Refund"}',
                "--instructions", "Which team?",
                "--options", '{"billing":"payments","support":"help"}',
            ])
            assert result.exit_code == 0
            assert "billing" in result.output
            assert "routing:" in result.output
            assert "experimental" in result.output

    def test_choice_list_options(self):
        with patch("laya_server.primitives.system_one", return_value=MOCK_CHOICE_RESP):
            result = CliRunner().invoke(cli, [
                "choice",
                "--state", '{"x":1}',
                "--instructions", "pick",
                "--options", '["billing","support"]',
                "--json",
            ])
            assert result.exit_code == 0


class TestScoreCommand:
    def test_score_json_output(self):
        with patch("laya_server.primitives.system_one", return_value=MOCK_SCORE_RESP):
            result = CliRunner().invoke(cli, [
                "score",
                "--state", '{"subject":"Refund"}',
                "--instructions", "How urgent?",
                "--levels", '["low","med","high"]',
                "--json",
            ])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["score"] == 1.5
            assert data["routing_signal"]["experimental"] is True


class TestNoulCommand:
    def test_noul_json_output(self):
        with patch("laya_server.primitives.system_one", return_value=MOCK_NOUL_RESP):
            result = CliRunner().invoke(cli, [
                "noul",
                "--state", '{"subject":"Refund"}',
                "--statement", "Is the customer likely to cancel?",
                "--json",
            ])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["noul"] == 0.73
            assert data["routing_signal"]["experimental"] is True

    def test_noul_with_descs(self):
        with patch("laya_server.primitives.system_one", return_value=MOCK_NOUL_RESP):
            result = CliRunner().invoke(cli, [
                "noul",
                "--state", '{"x":1}',
                "--statement", "Is this true?",
                "--true-desc", "yes",
                "--false-desc", "no",
                "--json",
            ])
            assert result.exit_code == 0


class TestBatchCommand:
    def test_batch_json_output(self):
        batch_resp = {
            "model": "laya",
            "answers": {
                "c": {
                    "type": "choice",
                    "choice": "A",
                    "probabilities": {"A": 0.8, "B": 0.2},
                    "confidence": 0.5,
                    "rl_agent": MOCK_ROUTING,
                },
                "n": {
                    "type": "noul",
                    "noul": 0.6,
                    "rl_agent": MOCK_ROUTING,
                },
            },
            "usage": {"input_tokens": 100, "output_tokens": 0},
            "cached": False,
        }
        with patch("laya_server.primitives.system_one", return_value=batch_resp):
            result = CliRunner().invoke(cli, [
                "batch",
                "--state", '{"x":1}',
                "--questions", '{"c":{"type":"choice","instructions":"pick","criteria":{"A":null,"B":null}},"n":{"type":"noul","instructions":"?"}}',
                "--json",
            ])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "answers" in data
            assert data["answers"]["c"]["choice"] == "A"
            assert data["answers"]["n"]["noul"] == 0.6
            assert data["answers"]["c"]["routing_signal"]["experimental"] is True
