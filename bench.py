#!/usr/bin/env python3
"""
Benchmark: Laya ONNX CPU vs GPU latency — single call and batched/coalesced load.

Measures:
  * model load time (CPU vs CUDA)
  * warm single systemOne (3 questions) latency, CPU vs GPU
  * N concurrent requests coalesced into one forward pass (batcher-style: one executor call over a
    list of payloads) — the realistic "traffic burst" number
  * a heavy single request (many questions) to show per-token scaling

Run with CUDA on the pinned card:
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13 ... .venv/bin/python bench.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import onnxruntime as ort  # noqa: E402

from src.worker import LayaWorker  # noqa: E402

MODEL_DIR = "/MASTERFOLDER/Services/laya/onnx"

STATE = {
    "subject": "Refund not received",
    "body": (
        "I cancelled my subscription two weeks ago and I still have not received my refund. "
        "This is the third time I am writing. If this is not resolved I will dispute the charge with my bank."
    ),
}
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "payments, refunds, invoices", "support": "product help and bugs", "sales": "new purchases and upgrades"},
    },
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
    "churn_risk": {"type": "noul", "instructions": "Is the customer likely to cancel or dispute?"},
}


def build_questions(n: int) -> dict:
    """Build n choice questions to exercise a larger forward pass."""
    qs = {}
    for i in range(n):
        qs[f"q{i}"] = {
            "type": "choice",
            "instructions": f"Which option is best for dimension {i}?",
            "criteria": {f"opt{a}": f"option {a}" for a in range(6)},
        }
    return qs


def bench_one(providers_label: str, providers: list, single_qs: dict, n_concurrent: int, n_heavy: int):
    print(f"\n=== {providers_label} ===")
    t0 = time.time()
    w = LayaWorker(MODEL_DIR, execution_providers=providers)
    load_s = time.time() - t0
    print(f"  load: {load_s:.2f}s   active={w.active_providers}")

    # warm the session / GPU arena
    w.system_one(STATE, QUESTIONS)

    # 1) warm single (3 questions)
    ts = []
    for _ in range(5):
        s = time.time()
        w.system_one(STATE, QUESTIONS)
        ts.append(time.time() - s)
    single = min(ts)
    print(f"  single (3q), best of 5: {single * 1000:.1f} ms   (mean {statistics.mean(ts) * 1000:.1f} ms)")

    # 2) N concurrent payloads coalesced into ONE executor call (batcher behaviour)
    payloads = [{"state": STATE, "questions": QUESTIONS} for _ in range(n_concurrent)]

    def run_batch():
        out = []
        for p in payloads:
            out.append(w.system_one(p["state"], p["questions"]))
        return out

    bs = []
    for _ in range(3):
        s = time.time()
        run_batch()
        bs.append(time.time() - s)
    total = min(bs)
    print(f"  batch {n_concurrent} concurrent (one pass): total {total * 1000:.1f} ms  => {total / n_concurrent * 1000:.1f} ms per request  (mean {statistics.mean(bs) * 1000:.1f} ms)")

    # 3) heavy single request (many questions in one call)
    heavy_qs = build_questions(n_heavy)
    w.system_one(STATE, heavy_qs)  # warm
    hs = []
    for _ in range(3):
        s = time.time()
        w.system_one(STATE, heavy_qs)
        hs.append(time.time() - s)
    print(f"  single heavy ({n_heavy} questions, one pass): {min(hs) * 1000:.1f} ms  (mean {statistics.mean(hs) * 1000:.1f} ms)")

    w.release()
    return {"load_s": load_s, "single_ms": single * 1000, "batch_total_ms": total * 1000, "per_req_ms": total / n_concurrent * 1000, "heavy_ms": min(hs) * 1000}


def main() -> int:
    n_concurrent = 16
    n_heavy = 24
    res = {}

    # CPU only
    res["cpu"] = bench_one("CPU", ["CPUExecutionProvider"], QUESTIONS, n_concurrent, n_heavy)

    # GPU (CUDA if libs resolve, else falls back to CPU and warns)
    gpu = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
    res["gpu"] = bench_one("GPU (CUDA)", gpu, QUESTIONS, n_concurrent, n_heavy)

    print("\n=== SUMMARY (ms, lower is better) ===")
    print(f"  {'metric':<28} {'CPU':>12} {'GPU':>12} {'speedup':>9}")
    rows = [
        ("model load", res["cpu"]["load_s"] * 1000, res["gpu"]["load_s"] * 1000),
        ("single 3q (warm)", res["cpu"]["single_ms"], res["gpu"]["single_ms"]),
        (f"batch {n_concurrent} total", res["cpu"]["batch_total_ms"], res["gpu"]["batch_total_ms"]),
        (f"batch {n_concurrent} per-req", res["cpu"]["per_req_ms"], res["gpu"]["per_req_ms"]),
        (f"single heavy {n_heavy}q", res["cpu"]["heavy_ms"], res["gpu"]["heavy_ms"]),
    ]
    for name, c, g in rows:
        sp = f"{c / g:.2f}x" if g > 0 else "n/a"
        print(f"  {name:<28} {c:>12.1f} {g:>12.1f} {sp:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
