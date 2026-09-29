"""§4 — Token and truncation limits.

State-length sweep (50→600 tokens), combined budget, error handling at limits.

Run: .venv/bin/python -m benchmarks.hardening.token_limits
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ._common import (
    get_output_dir, save_json, timed_post, warm_up, verify_provider,
    latency_stats, post_system_one, ENDPOINT,
)


# Use varied, non-repetitive sentences so the subword tokeniser doesn't
# over-compress the text. Each sentence is ~12-15 tokens.
_SENTENCES = [
    "The billing department received a complaint from a customer who was charged twice for the same subscription renewal.",
    "Our monitoring dashboard showed an unusual spike in request latency during the morning shift, which correlated with a database migration.",
    "The customer support team escalated the ticket because the client reported that their payment gateway was intermittently returning timeout errors.",
    "Engineering identified a memory leak in the connection pool manager that caused gradual degradation of the API response times over several hours.",
    "The operations team rolled back the deployment to the previous stable version after detecting a 15 percent increase in error rates across all endpoints.",
    "Finance confirmed that the invoice discrepancy was caused by a timezone mismatch between the billing system and the payment processor.",
    "The client requested a full audit of their usage metrics for the previous quarter, which required coordination between the data team and account management.",
    "A security scan flagged three medium-severity vulnerabilities in the authentication flow, requiring a patch to be deployed within the SLA window.",
    "The product team scheduled a sprint review to discuss the new feature rollout plan, which includes a phased approach to avoid service disruption.",
    "Network diagnostics revealed that the DNS resolution delays were caused by an expired TLS certificate on the load balancer infrastructure.",
    "The QA team found a regression in the reporting module where date ranges were calculated incorrectly for users in the Pacific timezone.",
    "Management approved the budget increase for additional GPU capacity to handle the projected growth in inference workloads next quarter.",
]


def make_state_at_tokens(target_tokens: int) -> dict:
    """Generate a state whose body is approximately target_tokens long.
    Uses varied sentences to avoid subword compression."""
    words = []
    total = 0
    si = 0
    while total < target_tokens * 4:  # rough: 4 chars per token
        sent = _SENTENCES[si % len(_SENTENCES)]
        # Vary the text slightly so repetition isn't exact
        modifier = f" (report #{si})" if si >= len(_SENTENCES) else ""
        words.append(sent + modifier)
        total += len(sent) + 1
        si += 1
    body = " ".join(words)
    return {"subject": f"state-{target_tokens}tok", "body": body}


def test_state_length_sweep() -> dict:
    """Sweep state length: 50, 100, 200, 300, 400, 450, 500, 512, 550, 600 tokens."""
    print("§4.1 state-length sweep")
    token_targets = [50, 100, 200, 300, 400, 450, 500, 512, 550, 600]
    results = []

    for target in token_targets:
        state = make_state_at_tokens(target)
        payload = {
            "state": state,
            "questions": {"q": {"type": "noul", "instructions": "Was the system configured correctly?"}},
        }
        # Warm-up
        post_system_one(payload)

        # Timed reps
        latencies = []
        for _ in range(10):
            _, ms = timed_post(payload)
            latencies.append(ms)

        result, _ = timed_post(payload)
        body = result.get("body", {})
        input_tokens = body.get("usage", {}).get("input_tokens") if isinstance(body, dict) else None
        stats = latency_stats(latencies)

        entry = {
            "target_tokens_approx": target,
            "input_tokens_rendered": input_tokens,
            "status_code": result.get("status_code"),
            "truncated": input_tokens is not None and input_tokens > 512,
            "latency": stats,
            "error": body if (isinstance(body, str) or (isinstance(body, dict) and "detail" in body)) else None,
        }
        results.append(entry)
        status = "OK" if entry["status_code"] == 200 else f"ERR({entry['status_code']})"
        trunc = " [TRUNCATED]" if entry["truncated"] else ""
        print(f"  ~{target:>4} tok target | rendered={input_tokens} | p50={stats['median']}ms {status}{trunc}")

    return {"test": "state_length_sweep", "results": results}


def test_combined_budget() -> dict:
    """Interaction between state length and question complexity.
    450-token state + 20-option choice vs 450-token state + noul."""
    print("§4.2 combined budget")
    results = []

    # Test matrix: state_length × question_type
    state_lengths = [200, 300, 400, 450, 500]
    question_configs = [
        ("noul", {"type": "noul", "instructions": "Is this valid?"}),
        ("choice_4", {"type": "choice", "instructions": "Category?",
                      "criteria": {"a": "alpha", "b": "beta", "c": "gamma", "d": "delta"}}),
        ("choice_8", {"type": "choice", "instructions": "Category?",
                      "criteria": {f"o{i}": f"Option {i}" for i in range(8)}}),
        ("choice_16", {"type": "choice", "instructions": "Category?",
                       "criteria": {f"o{i}": f"Option {i}" for i in range(16)}}),
        ("choice_32", {"type": "choice", "instructions": "Category?",
                       "criteria": {f"o{i}": f"Option {i}" for i in range(32)}}),
        ("score_5", {"type": "score", "instructions": "Rate urgency?",
                     "criteria": ["l1", "l2", "l3", "l4", "l5"]}),
    ]

    for slen in state_lengths:
        state = make_state_at_tokens(slen)
        for q_label, q_config in question_configs:
            qid = "q"
            payload = {
                "state": state,
                "questions": {qid: q_config},
            }
            post_system_one(payload)  # warm
            result, ms = timed_post(payload)
            body = result.get("body", {})
            input_tokens = body.get("usage", {}).get("input_tokens") if isinstance(body, dict) else None
            entry = {
                "state_tokens_approx": slen,
                "question_type": q_label,
                "input_tokens_rendered": input_tokens,
                "status_code": result.get("status_code"),
                "latency_ms": round(ms, 2),
                "truncated": input_tokens is not None and input_tokens > 512,
                "error": body if (isinstance(body, str) or (isinstance(body, dict) and "detail" in body)) else None,
            }
            results.append(entry)

        # Print row
        row = f"  state~{slen:>4}tok | "
        for q_label, _ in question_configs:
            match = next((r for r in results if r["state_tokens_approx"] == slen and r["question_type"] == q_label), None)
            if match:
                row += f"{q_label}={match['input_tokens_rendered']} "
        print(row)

    return {"test": "combined_budget", "results": results}


def test_error_handling_at_limits() -> dict:
    """What happens at the hard limits?"""
    print("§4.3 error handling at limits")
    results = []

    tests = [
        ("state_exceeds_max", "State alone > 512 tokens",
         make_state_at_tokens(600),
         {"q": {"type": "noul", "instructions": "Test?"}}),
        ("head_exceeds", "64 options (head might exceed 192)",
         make_state_at_tokens(50),
         {"q": {"type": "choice", "instructions": "Pick one?",
                "criteria": {f"o{i}": f"Option number {i} with some description" for i in range(64)}}}),
        ("head_exceeds_128", "128 options",
         make_state_at_tokens(50),
         {"q": {"type": "choice", "instructions": "Pick one?",
                "criteria": {f"o{i}": f"Option number {i}" for i in range(128)}}}),
        ("empty_state", "Empty state body",
         {"subject": "", "body": ""},
         {"q": {"type": "noul", "instructions": "Is there anything here?"}}),
        ("null_state", "Null-like state",
         {"subject": None, "body": None},
         {"q": {"type": "noul", "instructions": "Test?"}}),
        ("no_questions", "Empty questions object",
         make_state_at_tokens(50),
         {}),
        ("malformed_type", "Invalid question type",
         make_state_at_tokens(50),
         {"q": {"type": "invalid_type", "instructions": "Test?"}}),
        ("missing_instructions", "Choice without instructions",
         make_state_at_tokens(50),
         {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}),
        ("score_no_criteria", "Score without criteria",
         make_state_at_tokens(50),
         {"q": {"type": "score", "instructions": "Rate?"}}),
        ("noul_missing", "Noul without instructions",
         make_state_at_tokens(50),
         {"q": {"type": "noul"}}),
    ]

    for label, desc, state, questions in tests:
        payload = {"state": state, "questions": questions}
        result, ms = timed_post(payload)
        body = result.get("body", {})
        entry = {
            "label": label,
            "description": desc,
            "status_code": result.get("status_code"),
            "latency_ms": round(ms, 2),
            "error_detail": body if (isinstance(body, str) or (isinstance(body, dict) and "detail" in body)) else None,
            "answer": body.get("answers") if isinstance(body, dict) and "answers" in body else None,
        }
        results.append(entry)
        detail = entry["error_detail"]
        detail_str = json.dumps(detail)[:80] if detail else "clean answer"
        print(f"  {label:<22} → {result.get('status_code')} | {detail_str}")

    return {"test": "error_handling_at_limits", "results": results}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = get_output_dir()
    verify_provider()
    warm_up(5)

    all_results = {}
    all_results["state_length_sweep"] = test_state_length_sweep()
    all_results["combined_budget"] = test_combined_budget()
    all_results["error_at_limits"] = test_error_handling_at_limits()

    save_json(out_dir / "token_limits.json", all_results)
    print(f"\nResults saved to {out_dir / 'token_limits.json'}")


if __name__ == "__main__":
    main()
