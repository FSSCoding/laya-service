"""Lightweight HTTP client for the laya-service API.

Used by the CLI. Not used by the MCP server (which hits HTTP directly from JS).

Endpoint resolution order:
1. Explicit endpoint argument (from --endpoint flag)
2. LAYA_ENDPOINT env var
3. Default http://127.0.0.1:11448
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

DEFAULT_ENDPOINT = "http://127.0.0.1:11448"


def resolve_endpoint(endpoint: str | None = None) -> str:
    """Resolve the service endpoint URL."""
    if endpoint:
        return endpoint
    env = os.environ.get("LAYA_ENDPOINT")
    if env:
        return env
    return DEFAULT_ENDPOINT


def _request(
    method: str,
    url: str,
    timeout: float,
    json_body: dict | None = None,
) -> dict:
    """Execute an HTTP request with retry on transient read errors.

    Retry policy:
    - Retry (1x, 1s backoff): httpx.ReadError, httpx.ReadTimeout
    - No retry: httpx.ConnectError, httpx.ConnectTimeout, any 4xx/5xx
    """
    retries = 1
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.request(method, url, json=json_body)
            resp.raise_for_status()
            return resp.json()
        except httpx.ReadError as e:
            if attempt < retries:
                time.sleep(1)
                continue
            raise ConnectionError(f"Read error from {url}: {e}") from e
        except httpx.ReadTimeout as e:
            if attempt < retries:
                time.sleep(1)
                continue
            raise TimeoutError(f"Read timeout after {timeout}s from {url}") from e
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to {url} — is the service running?"
            ) from e
        except httpx.ConnectTimeout as e:
            raise TimeoutError(
                f"Connect timeout to {url} — service may be down"
            ) from e
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                body = e.response.json()
                if "detail" in body:
                    detail = str(body["detail"])
                elif "error" in body:
                    detail = str(body["error"])
            except Exception:
                pass
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {detail or e.response.status_code}"
            ) from e


def system_one(
    endpoint: str,
    state: Any,
    questions: dict,
    timeout: float = 30.0,
) -> dict:
    """POST /system_one. Returns {model, answers, usage, cached}."""
    return _request(
        "POST",
        f"{endpoint}/system_one",
        timeout,
        json_body={"state": state, "questions": questions},
    )


def health(endpoint: str, timeout: float = 5.0) -> dict:
    """GET /health. Returns {status, model_loaded, providers, uptime_s, version}."""
    return _request("GET", f"{endpoint}/health", timeout)


def stats(endpoint: str, timeout: float = 5.0) -> dict:
    """GET /v1/stats. Returns full stats object."""
    return _request("GET", f"{endpoint}/v1/stats", timeout)


def unload(endpoint: str, timeout: float = 5.0) -> dict:
    """POST /control/unload. Returns {status}."""
    return _request("POST", f"{endpoint}/control/unload", timeout)
