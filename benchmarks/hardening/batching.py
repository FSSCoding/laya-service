"""§3 — Batching and coalescing.

5ms window behaviour, batch saturation (1→512→513), heterogeneous batches.

Run: .venv/bin/python -m benchmarks.hardening.batching
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ._common import (
    get_output_dir, save_json, timed_post, warm_up, verify_provider,
    latency_stats, post_system_one, concurrent_burst, ENDPOINT,
)
import httpx


# ── Coalescing window behaviour ──────────────────────────────────────────────

def make_payload(i: int) -> dict:
    """A simple unique noul payload (unique enough to avoid cache)."""
    return {
        "state": {"subject": f"coalesce-{i}", "body": f"Unique case number {i} for coalescing test."},
        "questions": {"q": {"type": "noul", "instructions": f"Is case {i} a billing issue?"}},
    }


def test_coalescing_window() -> dict:
    """Send pairs/triples at controlled offsets. Determine if they share a batch."""
    print("§3.1 coalescing: 5ms window behaviour")
    results = []

    offsets_ms = [0, 2, 4, 6, 8, 12, 20]
    for offset in offsets_ms:
        # Run 5 trials at this offset
        trials = []
        for trial in range(5):
            i = trial * 100 + offset
            t0 = time.perf_counter()

            # Request A
            r1 = httpx.post(f"{ENDPOINT}/system_one", json=make_payload(i), timeout=30.0)
            t1 = time.perf_counter()

            # Wait for offset
            target = t0 + offset / 1000.0
            now = time.perf_counter()
            if now < target:
                time.sleep(target - now)
            actual_offset = (time.perf_counter() - t0) * 1000.0

            # Request B
            r2 = httpx.post(f"{ENDPOINT}/system_one", json=make_payload(i + 1), timeout=30.0)
            t2 = time.perf_counter()

            lat_a = (t1 - t0) * 1000.0
            lat_b = (t2 - t1) * 1000.0

            # If coalesced, B's wall-clock latency should be ~0 (it was dispatched with A)
            # If sequential, B waits for its own forward pass
            coalesced = lat_b < 10.0  # heuristic: <10ms means same batch
            trials.append({
                "offset_ms_requested": offset,
                "offset_ms_actual": round(actual_offset, 2),
                "lat_a_ms": round(lat_a, 2),
                "lat_b_ms": round(lat_b, 2),
                "coalesced": coalesced,
            })

        n_coalesced = sum(1 for t in trials if t["coalesced"])
        results.append({
            "offset_ms": offset,
            "trials": trials,
            "n_coalesced": n_coalesced,
            "n_total": len(trials),
            "mean_lat_a": round(sum(t["lat_a_ms"] for t in trials) / len(trials), 2),
            "mean_lat_b": round(sum(t["lat_b_ms"] for t in trials) / len(trials), 2),
        })
        print(f"  offset={offset:>3}ms | coalesced={n_coalesced}/5 | "
              f"mean A={sum(t['lat_a_ms'] for t in trials)/len(trials):.1f}ms | "
              f"mean B={sum(t['lat_b_ms'] for t in trials)/len(trials):.1f}ms")

    return {"test": "coalescing_window", "results": results}


# ── Batch saturation ─────────────────────────────────────────────────────────

SATURATION_LEVELS = [1, 4, 16, 64, 128, 256, 512, 513]


def test_batch_saturation() -> dict:
    """Sweep concurrency: 1, 4, 16, 64, 128, 256, 512, 513."""
    print("§3.2 batch saturation")
    results = []
    for n in SATURATION_LEVELS:
        # Use ≥200 requests for levels where possible; use n itself for small levels
        n_req = max(n, 200) if n <= 512 else n  # always ≥200 for throughput measurement
        if n > 512:
            n_req = n  # exactly 513 for the overflow test

        def factory(i):
            return make_payload(i * 1000 + n)

        burst = concurrent_burst(n_req, factory, timeout=120.0)
        entry = {
            "n_concurrent": n,
            "n_requests": n_req,
            "throughput_rps": burst["throughput_rps"],
            "latency": burst["stats"],
            "n_errors": sum(1 for r in burst["results"] if r["status_code"] != 200),
            "n_cached": sum(1 for r in burst["results"] if r.get("cached")),
        }
        results.append(entry)
        s = burst["stats"]
        err = f" ERR={entry['n_errors']}" if entry["n_errors"] else ""
        print(f"  n={n:>4} ({n_req:>4} reqs) | p50={s['median']}ms p99={s['p99']}ms | "
              f"throughput={burst['throughput_rps']} rps{err}")

        # Brief pause between levels to let the service recover
        time.sleep(1.0)

    return {"test": "batch_saturation", "results": results}


# ── Heterogeneous batch ──────────────────────────────────────────────────────

def test_heterogeneous_batch() -> dict:
    """Mixed-type questions (choice + score + noul) in one batch."""
    print("§3.3 heterogeneous batch")
    mixed_payloads = []
    for i in range(50):
        mixed_payloads.append({
            "state": {"subject": f"hetero-{i}", "body": f"Case {i}: a customer reported a problem."},
            "questions": {
                "cat": {
                    "type": "choice",
                    "instructions": "Category?",
                    "criteria": {"billing": "money", "support": "help", "sales": "purchases"},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Urgency?",
                    "criteria": ["low", "medium", "high"],
                },
                "churn": {
                    "type": "noul",
                    "instructions": "Likely to churn?",
                },
            },
        })

    # Send all 50 concurrently
    def factory(i):
        return mixed_payloads[i]

    burst = concurrent_burst(50, factory, timeout=60.0)
    entry = {
        "n_requests": 50,
        "throughput_rps": burst["throughput_rps"],
        "latency": burst["stats"],
        "n_errors": sum(1 for r in burst["results"] if r["status_code"] != 200),
    }
    print(f"  50 mixed-type requests | p50={burst['stats']['median']}ms | "
          f"throughput={burst['throughput_rps']} rps | errors={entry['n_errors']}")

    # Also test: one batch with 3 question types vs 3 separate single-type batches
    single_latencies = []
    for qtype in ["choice", "score", "noul"]:
        if qtype == "choice":
            payload = {"state": {"subject": "single", "body": "test"},
                       "questions": {"q": {"type": "choice", "instructions": "?",
                                           "criteria": {"a": "x", "b": "y", "c": "z"}}}}
        elif qtype == "score":
            payload = {"state": {"subject": "single", "body": "test"},
                       "questions": {"q": {"type": "score", "instructions": "?",
                                           "criteria": ["l1", "l2", "l3"]}}}
        else:
            payload = {"state": {"subject": "single", "body": "test"},
                       "questions": {"q": {"type": "noul", "instructions": "?"}}}
        post_system_one(payload)  # warm
        _, ms = timed_post(payload)
        single_latencies.append({"type": qtype, "latency_ms": round(ms, 2)})

    entry["single_type_latencies"] = single_latencies
    return {"test": "heterogeneous_batch", "results": entry}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = get_output_dir()
    verify_provider()
    warm_up(10)

    all_results = {}
    all_results["coalescing"] = test_coalescing_window()
    all_results["saturation"] = test_batch_saturation()
    all_results["heterogeneous"] = test_heterogeneous_batch()

    save_json(out_dir / "batching.json", all_results)
    print(f"\nResults saved to {out_dir / 'batching.json'}")


if __name__ == "__main__":
    main()
