"""CLI contract tests — verifies stable JSON output schemas.

These tests do NOT require a running service. They use the click test runner
with mocked client calls to verify output structure.
"""

import json

from click.testing import CliRunner

from laya_server.cli import cli


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
