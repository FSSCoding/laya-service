"""§5 — Systems and operational behaviour.

Latency distributions, cold/warm paths, cache behaviour, VRAM/RAM,
concurrency sweep, error recovery.

Run: .venv/bin/python -m benchmarks.hardening.systems_ops
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ._common import (
    get_output_dir, save_json, timed_post, warm_up, verify_provider,
    latency_stats, post_system_one, concurrent_burst,
    get_health, get_stats, post_unload, ENDPOINT,
)


def _nvidia_smi() -> dict:
    """Get GPU memory via nvidia-smi (card 1)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5.0, text=True,
        )
        gpus = {}
        for line in out.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 4:
                idx = int(parts[0].strip())
                gpus[idx] = {
                    "memory_used_mb": int(parts[1].strip()),
                    "memory_total_mb": int(parts[2].strip()),
                    "utilization_pct": int(parts[3].strip()),
                }
        return gpus
    except Exception as e:
        return {"error": str(e)}


def _ram_rss_kb() -> int:
    """Get RSS of the laya-service process."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "laya-server"], timeout=5.0, text=True,
        ).strip().split("\n")
        if not out or not out[0]:
            return 0
        pid = out[0]
        statm = subprocess.check_output(["cat", f"/proc/{pid}/statm"], timeout=5.0, text=True).split()
        return int(statm[1])  # in pages (4KB each)
    except Exception:
        return 0


# ── 5.1 Latency distributions ───────────────────────────────────────────────

def test_latency_by_question_count() -> dict:
    """Warm single-call latency: 1q, 3q, 10q, 24q, 50q (50 reps each)."""
    print("§5.1 latency by question count")
    results = []
    for n_q in [1, 3, 10, 24, 50]:
        questions = {}
        for i in range(n_q):
            questions[f"q{i}"] = {
                "type": "choice",
                "instructions": f"Dimension {i}?",
                "criteria": {f"a{j}": f"option {j}" for j in range(4)},
            }
        payload = {
            "state": {"subject": f"latency-{n_q}q", "body": "Standard test case for latency measurement."},
            "questions": questions,
        }
        post_system_one(payload)  # warm

        latencies = []
        for _ in range(50):
            _, ms = timed_post(payload)
            latencies.append(ms)

        stats = latency_stats(latencies)
        results.append({"n_questions": n_q, "latency": stats})
        print(f"  {n_q:>3}q | min={stats['min']} p50={stats['median']} p95={stats['p95']} p99={stats['p99']} mean={stats['mean']}ms")

    return {"test": "latency_by_question_count", "results": results}


# ── 5.2 Cold/warm paths ─────────────────────────────────────────────────────

def test_cold_warm() -> dict:
    """After unload: first request (cold) vs subsequent (warm)."""
    print("§5.2 cold/warm paths")
    results = {}

    payload = {
        "state": {"subject": "cold-warm-test", "body": "Testing cold reload behaviour."},
        "questions": {"q": {"type": "noul", "instructions": "Is this a test?"}},
    }

    # Unload
    print("  Unloading model...")
    post_unload()
    time.sleep(1.0)
    health = get_health()
    print(f"  After unload: model_loaded={health.get('model_loaded')}")

    # Cold: 5 reps
    cold_latencies = []
    for i in range(5):
        if i > 0:
            # Subsequent cold requests should be warm (model already loaded)
            pass
        result, ms = timed_post(payload)
        cold_latencies.append(ms)
    # Only the FIRST is truly cold; the rest are warm
    cold_first = cold_latencies[0] if cold_latencies else None
    warm_after_cold = cold_latencies[1:] if len(cold_latencies) > 1 else []

    # More warm reps
    warm_latencies = []
    for _ in range(50):
        _, ms = timed_post(payload)
        warm_latencies.append(ms)

    results["cold_first_ms"] = round(cold_first, 2) if cold_first else None
    results["warm_after_cold"] = latency_stats(warm_after_cold)
    results["warm_50_reps"] = latency_stats(warm_latencies)
    results["health_before"] = health

    print(f"  Cold first: {cold_first:.1f}ms")
    print(f"  Warm (50 reps): p50={latency_stats(warm_latencies)['median']}ms p99={latency_stats(warm_latencies)['p99']}ms")

    return {"test": "cold_warm", "results": results}


# ── 5.3 Cache behaviour ─────────────────────────────────────────────────────

def test_cache_behaviour() -> dict:
    """Cache hit vs miss, key sensitivity, persistence."""
    print("§5.3 cache behaviour")
    results = {}

    base_payload = {
        "state": {"subject": "cache-test", "body": "A specific state for cache testing."},
        "questions": {"q": {"type": "noul", "instructions": "Was this tested?"}},
    }

    # Warm up the cache entry
    post_system_one(base_payload)

    # Hit: 50 reps of the same payload
    hit_latencies = []
    hit_cached_flags = []
    for _ in range(50):
        result, ms = timed_post(base_payload)
        hit_latencies.append(ms)
        body = result.get("body", {})
        hit_cached_flags.append(body.get("cached") if isinstance(body, dict) else None)

    # Miss: 20 different payloads
    miss_latencies = []
    for i in range(20):
        miss_payload = {
            "state": {"subject": f"cache-miss-{i}", "body": f"Unique state number {i} for miss testing."},
            "questions": {"q": {"type": "noul", "instructions": f"Was case {i} tested?"}},
        }
        _, ms = timed_post(miss_payload)
        miss_latencies.append(ms)

    # Key sensitivity: 1-char change
    modified_payload = dict(base_payload)
    modified_payload["state"] = {"subject": "cache-test", "body": "A specific state for cache testingx"}  # +1 char
    result_mod, ms_mod = timed_post(modified_payload)
    body_mod = result_mod.get("body", {})
    mod_cached = body_mod.get("cached") if isinstance(body_mod, dict) else None

    results["hit"] = {
        "stats": latency_stats(hit_latencies),
        "n_cached_true": sum(1 for c in hit_cached_flags if c),
        "n_total": len(hit_cached_flags),
    }
    results["miss"] = {"stats": latency_stats(miss_latencies)}
    results["key_sensitivity"] = {
        "description": "1-char body change",
        "cached": mod_cached,
        "latency_ms": round(ms_mod, 2),
        "expected_miss": not mod_cached,
    }

    print(f"  Hit:  p50={latency_stats(hit_latencies)['median']}ms ({results['hit']['n_cached_true']}/50 cached)")
    print(f"  Miss: p50={latency_stats(miss_latencies)['median']}ms")
    print(f"  1-char change → cached={mod_cached} (miss expected: {not mod_cached})")

    return {"test": "cache_behaviour", "results": results}


# ── 5.4 VRAM/RAM ────────────────────────────────────────────────────────────

def test_vram_ram() -> dict:
    """GPU and memory snapshots."""
    print("§5.4 VRAM/RAM")
    results = {}

    # Unloaded state
    post_unload()
    time.sleep(1.0)
    gpus_unloaded = _nvidia_smi()
    ram_unloaded = _ram_rss_kb()

    # Load (trigger with a request)
    payload = {
        "state": {"subject": "vram-test", "body": "Testing VRAM usage."},
        "questions": {"q": {"type": "noul", "instructions": "Is this loaded?"}},
    }
    post_system_one(payload)
    time.sleep(1.0)
    gpus_loaded = _nvidia_smi()
    ram_loaded = _ram_rss_kb()

    # Under load (concurrent burst)
    def factory(i):
        return {
            "state": {"subject": f"vram-load-{i}", "body": f"Load test {i}."},
            "questions": {"q": {"type": "noul", "instructions": f"Is {i} valid?"}},
        }
    concurrent_burst(64, factory, timeout=60.0)
    gpus_under_load = _nvidia_smi()
    ram_under_load = _ram_rss_kb()

    results["unloaded"] = {"gpus": gpus_unloaded, "ram_rss_kb": ram_unloaded}
    results["loaded"] = {"gpus": gpus_loaded, "ram_rss_kb": ram_loaded}
    results["under_load"] = {"gpus": gpus_under_load, "ram_rss_kb": ram_under_load}

    card1_u = gpus_unloaded.get(1, {})
    card1_l = gpus_loaded.get(1, {})
    card1_ll = gpus_under_load.get(1, {})
    print(f"  Card 1 VRAM: unloaded={card1_u.get('memory_used_mb','?')}MB → "
          f"loaded={card1_l.get('memory_used_mb','?')}MB → "
          f"under_load={card1_ll.get('memory_used_mb','?')}MB")
    print(f"  RAM RSS: unloaded={ram_unloaded}KB → loaded={ram_loaded}KB → under_load={ram_under_load}KB")

    return {"test": "vram_ram", "results": results}


# ── 5.5 Concurrency sweep ───────────────────────────────────────────────────

CONCURRENCY_LEVELS = [1, 4, 16, 64, 128, 256, 512]


def test_concurrency_sweep() -> dict:
    """≥200 requests per level at increasing concurrency."""
    print("§5.5 concurrency sweep")
    results = []
    for n in CONCURRENCY_LEVELS:
        n_req = max(n, 200)

        def factory(i):
            return {
                "state": {"subject": f"conc-{n}-{i}", "body": f"Concurrency test {i}."},
                "questions": {"q": {"type": "noul", "instructions": f"Is {i} a valid case?"}},
            }

        burst = concurrent_burst(n_req, factory, timeout=120.0)
        entry = {
            "n_concurrent": n,
            "n_requests": n_req,
            "throughput_rps": burst["throughput_rps"],
            "latency": burst["stats"],
            "n_errors": sum(1 for r in burst["results"] if r["status_code"] != 200),
        }
        results.append(entry)
        s = burst["stats"]
        err = f" ERR={entry['n_errors']}" if entry["n_errors"] else ""
        print(f"  c={n:>4} ({n_req:>4} reqs) | p50={s['median']}ms p99={s['p99']}ms | "
              f"thru={burst['throughput_rps']} rps{err}")

        time.sleep(2.0)  # let service recover

    return {"test": "concurrency_sweep", "results": results}


# ── 5.6 Error recovery ──────────────────────────────────────────────────────

def test_error_recovery() -> dict:
    """After malformed requests, does the service stay healthy?"""
    print("§5.6 error recovery")
    results = {}

    # Send a batch of malformed requests
    malformed_payloads = [
        {"state": None, "questions": {}},
        {"state": "not a dict", "questions": {}},
        {"state": {"subject": "x", "body": "y"}, "questions": {"q": {"type": "bogus"}}},
        {"state": {"subject": "x", "body": "y"}, "questions": {}},
        {"state": {}, "questions": "not a dict"},
        {"state": [1, 2, 3], "questions": {"q": {"type": "noul"}}},
    ]

    errors = []
    for p in malformed_payloads:
        import httpx
        try:
            r = httpx.post(f"{ENDPOINT}/system_one", json=p, timeout=10.0)
            errors.append({"payload_keys": list(p.keys()), "status": r.status_code, "detail": r.text[:100]})
        except Exception as e:
            errors.append({"payload_keys": list(p.keys()), "status": 0, "detail": str(e)[:100]})

    # Check health after
    time.sleep(1.0)
    health_after = get_health()
    stats_after = get_stats()

    # Verify service still works
    recovery_payload = {
        "state": {"subject": "recovery-check", "body": "Is the service still functional?"},
        "questions": {"q": {"type": "noul", "instructions": "Is this working?"}},
    }
    result, ms = timed_post(recovery_payload)
    body = result.get("body", {})
    recovered = isinstance(body, dict) and "answers" in body

    results["malformed_requests"] = errors
    results["health_after"] = health_after
    results["recovery_check"] = {
        "status_code": result.get("status_code"),
        "latency_ms": round(ms, 2),
        "functional": recovered,
    }

    print(f"  Sent {len(errors)} malformed requests")
    print(f"  Health after: status={health_after.get('status')}, model_loaded={health_after.get('model_loaded')}")
    print(f"  Recovery check: functional={recovered} ({ms:.1f}ms)")

    return {"test": "error_recovery", "results": results}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = get_output_dir()
    verify_provider()
    warm_up(10)

    all_results = {}
    all_results["latency"] = test_latency_by_question_count()
    all_results["cold_warm"] = test_cold_warm()
    all_results["cache"] = test_cache_behaviour()
    all_results["vram_ram"] = test_vram_ram()
    all_results["concurrency"] = test_concurrency_sweep()
    all_results["error_recovery"] = test_error_recovery()

    save_json(out_dir / "systems_ops.json", all_results)
    print(f"\nResults saved to {out_dir / 'systems_ops.json'}")


if __name__ == "__main__":
    main()
