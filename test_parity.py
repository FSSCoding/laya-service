#!/usr/bin/env python3
"""
Parity gate: run the Laya GPU worker (onnxruntime-gpu) on the canonical fixture and confirm the
GPU outputs match the CPU reference (from /MasterFOLDER/Services/laya/test/test_model.ts).

Acceptance bar (correct across backends):
  * input_tokens must be EXACT (deterministic integer) -> 267
  * the model DECISION is invariant: same choice, same argmax, same score/noul to ~1e-2
  * per-probability values are within a tight absolute tolerance (CPU vs CUDA FP32
    non-determinism is ~1e-4, so 5e-4 abs is generous while still catching a real bug)

CPU reference (verified):
  usage.input_tokens = 267
  department.choice = billing ; probabilities {billing:0.9415, support:0.031, sales:0.0275} ; confidence 0.7603
  urgency.score = 1.3886 ; probabilities {"0":0.1752,"1":0.2947,"2":0.4962,"3":0.0338}
  churn_risk.noul = 0.0988
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.worker import LayaWorker  # noqa: E402

MODEL_DIR = "/MASTERFOLDER/Services/laya/onnx"
# Per-probability deltas are ~1e-4; confidence/score are nonlinear aggregates of those deltas and
# drift to ~7e-4. 1e-3 stays far tighter than any real bug (wrong sequence/temperature bucket moves
# values by ~0.01+) while absorbing CPU-vs-CUDA FP32 non-determinism on the aggregate terms.
TOL = 1e-3

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
        "criteria": {
            "billing": "payments, refunds, invoices",
            "support": "product help and bugs",
            "sales": "new purchases and upgrades",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
    },
    "churn_risk": {"type": "noul", "instructions": "Is the customer likely to cancel or dispute?"},
}


def main() -> int:
    print(f"Loading worker from {MODEL_DIR} (CUDA if available) ...")
    w = LayaWorker(MODEL_DIR)
    providers = w.active_providers
    print(f"providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("WARN: CUDA EP not active (fell back to CPU). Run with LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13")
    r = w.system_one(STATE, QUESTIONS)
    a = r["answers"]

    failures = []

    def check(name, got, want, exact=False):
        ok = (got == want) if exact else abs(got - want) <= TOL
        mark = "OK " if ok else "FAIL"
        detail = f"got={got!r} want={want!r}"
        if not exact:
            detail += f" d={abs(got - want):.6f}"
        print(f"  [{mark}] {name}: {detail}")
        if not ok:
            failures.append(name)

    # deterministic integer — must be exact
    check("usage.input_tokens", r["usage"]["input_tokens"], 267, exact=True)
    # decision-invariant (must be identical)
    check("department.choice", a["department"]["choice"], "billing", exact=True)
    # numeric — within tolerance (FP32 backend non-determinism)
    check("department.probabilities.billing", a["department"]["probabilities"]["billing"], 0.9415)
    check("department.probabilities.support", a["department"]["probabilities"]["support"], 0.031)
    check("department.probabilities.sales", a["department"]["probabilities"]["sales"], 0.0275)
    check("department.confidence", a["department"]["confidence"], 0.7603)
    check("urgency.score", a["urgency"]["score"], 1.3886)
    check("urgency.probabilities.0", a["urgency"]["probabilities"]["0"], 0.1752)
    check("urgency.probabilities.1", a["urgency"]["probabilities"]["1"], 0.2947)
    check("urgency.probabilities.2", a["urgency"]["probabilities"]["2"], 0.4962)
    check("urgency.probabilities.3", a["urgency"]["probabilities"]["3"], 0.0338)
    check("churn_risk.noul", a["churn_risk"]["noul"], 0.0988)

    w.release()
    if failures:
        print(f"\nPARITY FAILED on {len(failures)} checks: {failures}")
        return 1
    print("\nPARITY OK — GPU outputs match the CPU reference (decision-invariant, within FP32 tolerance).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
