"""Systems benchmark: latency, concurrency, cache, reload, VRAM.

Measures what the service costs. Sample counts per §6.2 of the spec.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from laya_server.client import resolve_endpoint, system_one, health, unload, stats  # noqa: E402

from .runner import collect_run_provenance  # noqa: E402


# Test fixtures (from tests/fixtures.py)
STATE = {
    "subject": "Refund not received",
    "body": "I cancelled my subscription two weeks ago and I still have not received my refund.",
}

QUESTIONS_3 = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "payments", "support": "help", "sales": "new purchases"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
    },
    "churn_risk": {"type": "noul", "instructions": "Is the customer likely to cancel?"},
}


def build_questions(n: int, qtype: str = "choice") -> dict:
    """Build n questions of a given type."""
    qs = {}
    for i in range(n):
        if qtype == "choice":
            qs[f"q{i}"] = {
                "type": "choice",
                "instructions": f"Which option is best for dimension {i}?",
                "criteria": {f"opt{a}": f"option {a}" for a in range(6)},
            }
        elif qtype == "score":
            qs[f"q{i}"] = {
                "type": "score",
                "instructions": f"How urgent is dimension {i}?",
                "criteria": ["low", "med", "high"],
            }
        elif qtype == "noul":
            qs[f"q{i}"] = {
                "type": "noul",
                "instructions": f"Is dimension {i} important?",
            }
    return qs


def measure_latency(fn, reps: int) -> dict[str, float]:
    """Run fn() reps times, return min/median/p95/p99/mean."""
    times = []
    for _ in range(reps):
        t0 = time.time()
        fn()
        times.append((time.time() - t0) * 1000)
    times.sort()
    n = len(times)
    return {
        "min": times[0],
        "median": statistics.median(times),
        "p95": times[int(n * 0.95)] if n >= 20 else times[-1],
        "p99": times[int(n * 0.99)] if n >= 100 else times[-1],
        "mean": statistics.mean(times),
        "n": n,
    }


def bench_warm_single(endpoint: str) -> dict:
    """Warm single-call latency by question-shape (50 reps each)."""
    print("  warm single-call latency...")
    shapes = {
        "1q_choice": build_questions(1, "choice"),
        "3q_mixed": QUESTIONS_3,
        "10q_choice": build_questions(10, "choice"),
        "24q_choice": build_questions(24, "choice"),
    }
    results = {}
    for name, qs in shapes.items():
        def call():
            system_one(endpoint, STATE, qs)
        results[name] = measure_latency(call, reps=50)
    return results


def bench_cold_reload(endpoint: str) -> dict:
    """Cold reload latency (5 reps)."""
    print("  cold reload latency...")
    times = []
    for i in range(5):
        # Unload
        unload(endpoint)
        time.sleep(1)
        # First request (cold)
        t0 = time.time()
        system_one(endpoint, STATE, QUESTIONS_3)
        cold_ms = (time.time() - t0) * 1000
        times.append(cold_ms)
        print(f"    reload {i+1}/5: {cold_ms:.1f} ms")
    times.sort()
    return {
        "min": times[0],
        "median": statistics.median(times),
        "max": times[-1],
        "mean": statistics.mean(times),
        "n": 5,
    }


def bench_concurrency(endpoint: str) -> dict:
    """Concurrency sweep: 1/4/16/64 concurrent requests (≥200 requests per level)."""
    print("  concurrency sweep...")
    levels = [1, 4, 16, 64]
    results = {}
    for conc in levels:
        n_requests = max(200, conc * 10)
        n_batches = n_requests // conc
        times = []
        for _ in range(n_batches):
            def batch_call():
                with ThreadPoolExecutor(max_workers=conc) as pool:
                    futures = [
                        pool.submit(system_one, endpoint, STATE, QUESTIONS_3)
                        for _ in range(conc)
                    ]
                    for f in as_completed(futures):
                        f.result()
            t0 = time.time()
            batch_call()
            batch_ms = (time.time() - t0) * 1000
            per_req_ms = batch_ms / conc
            times.append(per_req_ms)
        times.sort()
        n = len(times)
        results[f"conc_{conc}"] = {
            "min": times[0],
            "median": statistics.median(times),
            "p95": times[int(n * 0.95)] if n >= 20 else times[-1],
            "p99": times[int(n * 0.99)] if n >= 100 else times[-1],
            "mean": statistics.mean(times),
            "throughput_rps": conc * 1000 / statistics.mean(times),
            "n_requests": n * conc,
        }
    return results


def bench_cache(endpoint: str) -> dict:
    """Cache hit vs miss latency (50 hit, 20 miss)."""
    print("  cache hit/miss latency...")
    # Warm the cache
    system_one(endpoint, STATE, QUESTIONS_3)

    # Cache hit (repeat identical payload)
    def hit_call():
        system_one(endpoint, STATE, QUESTIONS_3)
    hit_stats = measure_latency(hit_call, reps=50)

    # Cache miss (1-char state change)
    miss_state = {**STATE, "subject": STATE["subject"] + "!"}
    miss_times = []
    for _ in range(20):
        miss_state = {**miss_state, "subject": miss_state["subject"] + "x"}
        t0 = time.time()
        system_one(endpoint, miss_state, QUESTIONS_3)
        miss_times.append((time.time() - t0) * 1000)
    miss_times.sort()
    n = len(miss_times)
    miss_stats = {
        "min": miss_times[0],
        "median": statistics.median(miss_times),
        "p95": miss_times[int(n * 0.95)] if n >= 20 else miss_times[-1],
        "mean": statistics.mean(miss_times),
        "n": n,
    }

    return {"hit": hit_stats, "miss": miss_stats}


def bench_option_count(endpoint: str) -> dict:
    """Latency vs option count (2 → 10 → 20 → 25+ options)."""
    print("  latency vs option count...")
    option_counts = [2, 5, 10, 20, 25]
    results = {}
    for n_opts in option_counts:
        qs = {
            "q": {
                "type": "choice",
                "instructions": "Which option?",
                "criteria": {f"opt{i}": f"option {i}" for i in range(n_opts)},
            }
        }
        def call():
            resp = system_one(endpoint, STATE, qs)
            return resp
        stats = measure_latency(call, reps=50)
        # Get input_tokens from one call
        resp = call()
        stats["input_tokens"] = resp.get("usage", {}).get("input_tokens", 0)
        results[f"opts_{n_opts}"] = stats
    return results


def get_vram_snapshot() -> dict:
    """Get GPU VRAM usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("\n")[0].split(",")
            return {
                "vram_used_mb": float(parts[0].strip()),
                "vram_total_mb": float(parts[1].strip()),
                "gpu_util_pct": float(parts[2].strip()),
            }
    except Exception:
        pass
    return {"error": "nvidia-smi unavailable"}


def run_systems_benchmark(
    endpoint: str | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run full systems benchmark. Returns results dict."""
    endpoint = resolve_endpoint(endpoint)
    print(f"Systems benchmark against {endpoint}\n")

    # Collect provenance
    prov = collect_run_provenance(endpoint)

    # VRAM before
    vram_before = get_vram_snapshot()
    print(f"VRAM before: {vram_before}\n")

    # Run all benchmarks
    results = {
        "warm_single": bench_warm_single(endpoint),
        "cold_reload": bench_cold_reload(endpoint),
        "concurrency": bench_concurrency(endpoint),
        "cache": bench_cache(endpoint),
        "option_count": bench_option_count(endpoint),
    }

    # VRAM after
    vram_after = get_vram_snapshot()
    print(f"\nVRAM after: {vram_after}")

    # Add metadata
    results["provenance"] = prov
    results["vram_before"] = vram_before
    results["vram_after"] = vram_after

    # Write output
    if output_path is None:
        output_path = Path(__file__).parent.parent / "results" / f"{prov['run_date'][:10]}-systems.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote systems benchmark to {output_path}")
    return results


def main():
    """CLI entry point: python -m benchmarks.harness.systems [--endpoint URL]"""
    import argparse
    parser = argparse.ArgumentParser(description="Run systems benchmark")
    parser.add_argument("--endpoint", type=str, default=None, help="Service endpoint URL")
    parser.add_argument("--output", type=Path, default=None, help="Output path for systems.json")
    args = parser.parse_args()

    run_systems_benchmark(args.endpoint, args.output)


if __name__ == "__main__":
    main()
