"""Typed configuration for laya-service.

Configuration resolution order (strict — no silent fallback):
1. LAYA_CONFIG_PATH env var, if set
2. config/laya_config.json relative to CWD, if it exists
3. Raise FileNotFoundError
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class GPUProvider(BaseModel):
    name: str
    options: dict[str, Any] = {}


class GPUConfig(BaseModel):
    visible_devices: str = "0"
    providers: list[GPUProvider] = Field(default_factory=list)


class BatcherConfig(BaseModel):
    max_batch_size: int = Field(default=512, gt=0)
    max_wait_ms: float = Field(default=5.0, ge=0)


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_days: int = Field(default=30, gt=0)
    max_memory_entries: int = Field(default=10000, gt=0)
    db_path: str = "~/.laya-service/result_cache.db"

    @field_validator("db_path")
    @classmethod
    def expand_user(cls, v: str) -> str:
        return str(Path(v).expanduser())


class IdleConfig(BaseModel):
    unload_after_s: int = Field(default=300, ge=0)
    enabled: bool = True


class AppConfig(BaseModel):
    model_dir: str = "/MASTERFOLDER/Services/laya/onnx"
    host: str = "0.0.0.0"
    port: int = Field(default=11448, ge=1, le=65535)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    batcher: BatcherConfig = Field(default_factory=BatcherConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    idle: IdleConfig = Field(default_factory=IdleConfig)

    @field_validator("model_dir")
    @classmethod
    def model_dir_exists(cls, v: str) -> str:
        p = Path(v).expanduser()
        if not p.is_dir():
            raise ValueError(f"model_dir does not exist: {p}")
        return str(p)


def resolve_config_path() -> Path:
    """Resolve the config path. Raises FileNotFoundError if not found."""
    env = os.environ.get("LAYA_CONFIG_PATH")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"LAYA_CONFIG_PATH={p} but file does not exist")
        return p
    cwd_config = Path("config/laya_config.json")
    if cwd_config.is_file():
        return cwd_config
    raise FileNotFoundError(
        "No config found. Set LAYA_CONFIG_PATH or ensure "
        "config/laya_config.json exists relative to CWD."
    )


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate the config file. No side effects."""
    if path is None:
        path = resolve_config_path()
    with open(path) as f:
        return AppConfig.model_validate(json.load(f))
