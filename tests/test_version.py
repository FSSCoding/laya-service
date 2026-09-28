"""Version drift check: pyproject.toml and __init__.py must agree."""

from importlib.metadata import version

import laya_server


def test_version_matches_metadata():
    assert version("laya-service") == laya_server.__version__
