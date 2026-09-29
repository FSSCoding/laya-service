"""Shared utilities for hardening scripts.

Provides HTTP helpers, timing, stats, and output management.
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ENDPOINT = "http://127.0.0.1:11448"
RESULTS_DIR = Path(__file__).parent.parent / "results"
HARDENING_DIR = RESULTS_DIR / "hardening"


def get_output_dir() -> Path:
    """Get today's hardening output directory."""
    d = HARDENING_DIR / datetime.now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def post_system_one(
    payload: dict,
    timeout: float = 30.0,
    endpoint: str = ENDPOINT,
) -> dict:
    """POST /system_one, return parsed JSON (or error dict on failure)."""
    try:
        r = httpx.post(f"{endpoint}/system_one", json=payload, timeout=timeout)
        return {"status_code": r.status_code, "body": r.json()}
    except httpx.HTTPStatusError as e:
        return {"status_code": e.response.status_code, "body": e.response.text}
    except Exception as e:
        return {"status_code": 0, "body": str(e)}


def get_health(endpoint: str = ENDPOINT) -> dict:
    r = httpx.get(f"{endpoint}/health", timeout=10.0)
    return r.json()


def get_stats(endpoint: str = ENDPOINT) -> dict:
    r = httpx.get(f"{endpoint}/v1/stats", timeout=10.0)
    return r.json()


def post_unload(endpoint: str = ENDPOINT) -> dict:
    r = httpx.post(f"{endpoint}/control/unload", timeout=10.0)
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}


def timed_post(payload: dict, timeout: float = 30.0, endpoint: str = ENDPOINT) -> dict:
    """POST /system_one with precise timing. Returns (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = post_system_one(payload, timeout=timeout, endpoint=endpoint)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


def latency_stats(ms_list: list[float]) -> dict:
    """Compute min/median/p95/p99/mean from a list of latencies in ms."""
    if not ms_list:
        return {"n": 0, "min": None, "median": None, "p95": None, "p99": None, "mean": None}
    s = sorted(ms_list)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(int(p / 100.0 * n), n - 1)
        return s[idx]

    return {
        "n": n,
        "min": round(s[0], 2),
        "median": round(statistics.median(s), 2),
        "p95": round(pct(95), 2),
        "p99": round(pct(99), 2),
        "mean": round(statistics.mean(s), 2),
    }


def warm_up(n: int = 5, endpoint: str = ENDPOINT) -> None:
    """Fire n throwaway requests to ensure model is loaded and warm."""
    payload = {
        "state": {"subject": "warmup", "body": "warming up"},
        "questions": {"q": {"type": "noul", "instructions": "Is this warm?"}},
    }
    for _ in range(n):
        post_system_one(payload, endpoint=endpoint)


def verify_provider(endpoint: str = ENDPOINT) -> dict:
    """Verify CUDA EP is active. Returns health dict. Raises if CPU-only."""
    health = get_health(endpoint)
    providers = health.get("providers", [])
    if not providers or "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            f"CUDA EP not active! providers={providers}. "
            f"Check LD_LIBRARY_PATH and nvidia-smi."
        )
    if "CUDAExecutionProvider" != providers[0]:
        print(f"  ⚠️  WARNING: CUDA is not the primary provider: {providers}")
    print(f"  Provider verified: {providers[0]} (card-pinned)")
    return health


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def make_choice_state(n_options: int, state_body: str = "Which category does this belong to?") -> dict:
    """Build a state + choice question with n_options options."""
    criteria = {f"option_{i}": f"Category {i} description" for i in range(n_options)}
    return {
        "state": {"subject": f"Test with {n_options} options", "body": state_body},
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "Which option best matches this case?",
                "criteria": criteria,
            }
        },
    }


def make_score_state(n_levels: int) -> dict:
    """Build a state + score question with n_levels levels."""
    criteria = [f"Level {i+1}" for i in range(n_levels)]
    return {
        "state": {"subject": "Urgency assessment", "body": "A production service is experiencing intermittent failures."},
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this incident?",
                "criteria": criteria,
            }
        },
    }


def make_noul_state(proposition: str, state_body: str = "") -> dict:
    """Build a state + noul question."""
    body = state_body or "The server was restarted at 14:00 UTC by the operations team."
    return {
        "state": {"subject": "Factual check", "body": body},
        "questions": {
            "q": {"type": "noul", "instructions": proposition},
        },
    }


def concurrent_burst(
    n_requests: int,
    payload_factory,
    timeout: float = 60.0,
    endpoint: str = ENDPOINT,
) -> dict:
    """Fire n_requests concurrently, collect per-request latencies and results.

    payload_factory(i) -> dict (the request payload for request i).
    """
    results = []
    latencies = []

    def do_one(i: int) -> tuple:
        payload = payload_factory(i)
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{endpoint}/system_one", json=payload, timeout=timeout)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return i, r.status_code, r.json(), elapsed
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return i, 0, str(e), elapsed

    with ThreadPoolExecutor(max_workers=min(n_requests, 512)) as pool:
        futures = [pool.submit(do_one, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            i, status, body, elapsed = fut.result()
            latencies.append(elapsed)
            results.append({
                "index": i,
                "status_code": status,
                "latency_ms": round(elapsed, 2),
                "cached": body.get("cached") if isinstance(body, dict) else None,
                "input_tokens": body.get("usage", {}).get("input_tokens") if isinstance(body, dict) else None,
            })

    # Sort by index for deterministic output
    results.sort(key=lambda x: x["index"])
    return {
        "n_requests": n_requests,
        "latencies_ms": [round(l, 2) for l in latencies],
        "stats": latency_stats(latencies),
        "throughput_rps": round(n_requests / (sum(latencies) / 1000.0), 2) if latencies else 0,
        "results": results,
    }
