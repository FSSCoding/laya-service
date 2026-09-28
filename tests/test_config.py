"""Config validation and resolution tests."""

import json

import pytest

from laya_server._config import AppConfig, BatcherConfig, resolve_config_path


class TestConfigValidation:
    def test_rejects_invalid_port_high(self):
        with pytest.raises(Exception, match="port"):
            AppConfig(port=99999)

    def test_rejects_invalid_port_negative(self):
        with pytest.raises(Exception, match="port"):
            AppConfig(port=-1)

    def test_rejects_zero_batcher_size(self):
        with pytest.raises(Exception, match="max_batch_size"):
            BatcherConfig(max_batch_size=0)

    def test_rejects_negative_ttl(self):
        from laya_server._config import CacheConfig

        with pytest.raises(Exception, match="ttl_days"):
            CacheConfig(ttl_days=0)

    def test_expands_db_path(self):
        from laya_server._config import CacheConfig

        cfg = CacheConfig(db_path="~/.laya-service/result_cache.db")
        assert cfg.db_path.startswith("/home/") or cfg.db_path.startswith("/root/")
        assert "~" not in cfg.db_path


class TestConfigResolution:
    def test_missing_config_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LAYA_CONFIG_PATH", raising=False)
        with pytest.raises(FileNotFoundError, match="No config found"):
            resolve_config_path()

    def test_env_var_resolution(self, tmp_path, monkeypatch):
        config_file = tmp_path / "custom_config.json"
        config_file.write_text(json.dumps({"port": 9999}))
        monkeypatch.setenv("LAYA_CONFIG_PATH", str(config_file))
        p = resolve_config_path()
        assert p == config_file

    def test_env_var_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAYA_CONFIG_PATH", str(tmp_path / "nonexistent.json"))
        with pytest.raises(FileNotFoundError, match="LAYA_CONFIG_PATH"):
            resolve_config_path()

    def test_cwd_relative_resolution(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "laya_config.json").write_text(json.dumps({"port": 8080}))
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LAYA_CONFIG_PATH", raising=False)
        p = resolve_config_path()
        assert p.name == "laya_config.json"
        assert p.parent.name == "config"
        assert p.is_file()


class TestCreateAppIsolation:
    def test_no_module_level_app(self):
        import laya_server.server as srv

        assert not hasattr(srv, "app"), "server.py must not define a module-level app instance"

    def test_two_apps_independent(self, tmp_path):
        from laya_server._config import AppConfig
        from laya_server.server import create_app

        cfg1 = AppConfig(
            model_dir="/MASTERFOLDER/Services/laya/onnx",
            port=11448,
            host="0.0.0.0",
        )
        cfg2 = AppConfig(
            model_dir="/MASTERFOLDER/Services/laya/onnx",
            port=9999,
            host="127.0.0.1",
        )
        app1 = create_app(cfg1)
        app2 = create_app(cfg2)
        assert app1 is not app2
        assert len(app1.routes) == len(app2.routes)

    def test_version_in_app_metadata(self):
        from laya_server import __version__
        from laya_server._config import AppConfig
        from laya_server.server import create_app

        cfg = AppConfig(model_dir="/MASTERFOLDER/Services/laya/onnx")
        app = create_app(cfg)
        assert app.version == __version__
