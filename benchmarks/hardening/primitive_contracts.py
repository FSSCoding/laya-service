"""§2 — Native primitive contract testing.

choice: option-count scaling (2→64+), truncation, ordering sensitivity
score: level-count bounds (1→11), controlled monotonic ladders
noul: mechanical semantics (true/false/missing/negated/edge cases)

Run: .venv/bin/python -m benchmarks.hardening.primitive_contracts
"""
from __future__ import annotations

import json
import sys
import time
from math import log2
from pathlib import Path

from ._common import (
    get_output_dir, save_json, timed_post, warm_up, verify_provider,
    latency_stats, post_system_one, make_choice_state, make_score_state, make_noul_state,
    ENDPOINT,
)


# ── Choice: option-count scaling ─────────────────────────────────────────────

CHOICE_SWEEP = [2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]


def test_choice_option_scaling() -> dict:
    """Sweep option count from 2 to 256. Record tokens, latency, distribution shape."""
    print("§2.1 choice: option-count scaling")
    results = []
    for n_opts in CHOICE_SWEEP:
        payload = make_choice_state(n_opts)
        # Warm-up (skip first call for timing accuracy)
        post_system_one(payload)

        # Timed reps
        latencies = []
        for _ in range(20):
            _, ms = timed_post(payload)
            latencies.append(ms)

        # Get the last response for analysis
        result, _ = timed_post(payload)
        body = result.get("body", {})

        if isinstance(body, dict) and "answers" in body:
            ans = body["answers"].get("dept", {})
            probs = ans.get("probabilities", {})
            max_p = max(probs.values()) if probs else None
            entropy = -sum(p * log2(p) for p in probs.values() if p > 0) if probs else None
            n_opts_actual = len(probs)
        else:
            probs = {}
            max_p = None
            entropy = None
            n_opts_actual = 0

        stats = latency_stats(latencies)
        entry = {
            "n_options": n_opts,
            "status_code": result.get("status_code"),
            "input_tokens": body.get("usage", {}).get("input_tokens") if isinstance(body, dict) else None,
            "n_options_returned": n_opts_actual,
            "max_probability": round(max_p, 6) if max_p is not None else None,
            "entropy_bits": round(entropy, 4) if entropy is not None else None,
            "confidence": ans.get("confidence") if isinstance(body, dict) and "answers" in body else None,
            "truncated": n_opts_actual < n_opts,
            "latency": stats,
            "error": body if (isinstance(body, str) or (isinstance(body, dict) and "detail" in body)) else None,
        }
        results.append(entry)
        status = "OK" if entry["status_code"] == 200 else f"ERR({entry['status_code']})"
        trunc = " [TRUNCATED]" if entry["truncated"] else ""
        print(f"  {n_opts:>4} opts | tokens={entry['input_tokens']} | p50={stats['median']}ms | "
              f"maxP={entry['max_probability']} | entropy={entry['entropy_bits']} {status}{trunc}")

        # Stop if we got a hard rejection
        if result.get("status_code") in (400, 422):
            print(f"  → Hard limit reached at {n_opts} options")
            break

    return {"test": "choice_option_scaling", "results": results}


def test_choice_ordering_sensitivity() -> dict:
    """Same options, reversed order → same answer?"""
    print("§2.1b choice: ordering sensitivity")
    base_options = [f"Option {i}" for i in range(8)]
    payload_fwd = {
        "state": {"subject": "Ordering test", "body": "A billing dispute about a refund."},
        "questions": {"q": {
            "type": "choice",
            "instructions": "Which category?",
            "criteria": {f"opt{i}": o for i, o in enumerate(base_options)},
        }},
    }
    payload_rev = {
        "state": payload_fwd["state"],
        "questions": {"q": {
            "type": "choice",
            "instructions": "Which category?",
            "criteria": {f"opt{i}": o for i, o in enumerate(reversed(base_options))},
        }},
    }
    r1, _ = timed_post(payload_fwd)
    r2, _ = timed_post(payload_rev)
    a1 = r1.get("body", {}).get("answers", {}).get("q", {})
    a2 = r2.get("body", {}).get("answers", {}).get("q", {})
    choice1 = a1.get("choice")
    choice2 = a2.get("choice")
    same = choice1 == choice2
    print(f"  forward: {choice1} | reversed: {choice2} | same={same}")
    return {
        "test": "choice_ordering_sensitivity",
        "forward_choice": choice1,
        "reversed_choice": choice2,
        "same_answer": same,
    }


# ── Score: level-count bounds ────────────────────────────────────────────────

SCORE_BOUNDARY_LEVELS = [1, 2, 3, 4, 5, 7, 10, 11, 0, -1]


def test_score_level_bounds() -> dict:
    """Test score with 0, -1, 1, 2, 3, 4, 5, 7, 10, 11 levels."""
    print("§2.2 score: level-count bounds")
    results = []
    for n_levels in SCORE_BOUNDARY_LEVELS:
        if n_levels <= 0:
            # Empty criteria list (closest valid shape for 0/negative)
            payload = {
                "state": {"subject": "Score bounds", "body": "Test state."},
                "questions": {"q": {
                    "type": "score",
                    "instructions": "Rate this.",
                    "criteria": [],
                }},
            }
        else:
            criteria = [f"Level {i+1}" for i in range(n_levels)]
            payload = {
                "state": {"subject": "Score bounds", "body": "A production service is experiencing failures."},
                "questions": {"q": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": criteria,
                }},
            }

        result, ms = timed_post(payload)
        body = result.get("body", {})
        # Extract the score value if present
        score_val = None
        if isinstance(body, dict) and "answers" in body:
            ans = body["answers"].get("q", {})
            score_val = ans.get("score")
        entry = {
            "n_levels": n_levels,
            "status_code": result.get("status_code"),
            "score_value": score_val,
            "latency_ms": round(ms, 2),
            "error": body if (isinstance(body, str) or (isinstance(body, dict) and ("detail" in body or "error" in body))) else None,
        }
        results.append(entry)
        status = "OK" if entry["status_code"] == 200 else f"REJECT({entry['status_code']})"
        detail = ""
        if isinstance(body, dict):
            if "detail" in body:
                detail = f" [{body['detail'][:60]}]"
            elif "error" in body:
                detail = f" [{body['error'].get('message', '')[:60]}]"
        elif isinstance(body, str) and entry["status_code"] != 200:
            detail = f" [{body[:60]}]"
        score_str = f" score={score_val}" if score_val is not None else ""
        print(f"  {n_levels:>3} levels → {status}{score_str}{detail}")

    return {"test": "score_level_bounds", "results": results}


def test_score_monotonic_ladder() -> dict:
    """Controlled urgency ladder: 6 states from routine to safety-critical.
    Run at 3, 5, 7, 10 levels. Check monotonicity of scores."""
    print("§2.2b score: monotonic ladder")
    ladder_states = [
        ("routine FYI", "FYI: The weekly report has been published. No action needed."),
        ("no deadline", "Please review the attached document when you get a chance. No rush."),
        ("deadline next week", "The client needs the deliverable by next Friday. We are on track but need to confirm scope."),
        ("deadline today", "URGENT: The client's production site is down. We need a fix within 2 hours."),
        ("service degraded", "CRITICAL: 40% of customers cannot access checkout. Revenue is actively lost. Multiple teams mobilised."),
        ("safety-critical", "SAFETY: A data corruption event may have affected user financial records. Legal is involved. Regulatory notification may be required within 24h."),
    ]

    level_counts = [3, 5, 7, 10]
    results = {}
    for n_levels in level_counts:
        criteria = [f"Level {i+1}" for i in range(n_levels)]
        scores = []
        for label, body in ladder_states:
            payload = {
                "state": {"subject": label, "body": body},
                "questions": {"urgency": {
                    "type": "score",
                    "instructions": "How urgent is this ticket? 1 = lowest, highest = most urgent.",
                    "criteria": criteria,
                }},
            }
            result, _ = timed_post(payload)
            ans = result.get("body", {}).get("answers", {}).get("urgency", {})
            scores.append({"label": label, "score": ans.get("score"), "confidence": ans.get("confidence")})

        # Check monotonicity
        raw_scores = [s["score"] for s in scores if s["score"] is not None]
        monotonic = all(a <= b for a, b in zip(raw_scores, raw_scores[1:])) if len(raw_scores) >= 2 else None
        results[str(n_levels)] = {
            "scores": scores,
            "monotonic": monotonic,
        }
        print(f"  {n_levels} levels: {[s['score'] for s in scores]} → monotonic={monotonic}")

    return {"test": "score_monotonic_ladder", "results": results}


# ── Noul: mechanical semantics ───────────────────────────────────────────────


def test_noul_mechanical() -> dict:
    """Test noul with controlled true/false/edge inputs."""
    print("§2.3 noul: mechanical semantics")
    state_true = "The server was restarted at 14:00 UTC by the operations team. The restart was confirmed in the monitoring dashboard."
    state_false = "The server was last restarted on Monday morning at 09:15 by an automated deploy pipeline."

    tests = [
        ("true_prop", state_true, "Was the server restarted at 14:00 UTC?"),
        ("false_prop", state_true, "Was the server restarted at 10:00 UTC?"),
        ("true_from_false_state", state_false, "Was the server restarted by an automated deploy pipeline?"),
        ("false_from_false_state", state_false, "Was the server restarted at 14:00 UTC?"),
        ("negated_true", state_true, "Was the server NOT restarted at 14:00 UTC?"),
        ("negated_false", state_true, "Was the server NOT restarted at 10:00 UTC?"),
        ("missing_evidence", "No information available.", "Was the server restarted at 14:00 UTC?"),
        ("contradictory", "The server was restarted at 14:00 UTC. The server was not restarted today.", "Was the server restarted at 14:00 UTC?"),
        ("empty_state", "", "Is anything documented here?"),
        ("unicode_state", "Überprüfung des Servers à 14h00 — confirmado ✅", "Was the server checked at 14:00?"),
        ("structured_json", json.dumps({"event": "restart", "time": "14:00 UTC", "by": "ops-team"}), "Was the restart performed by ops-team?"),
    ]

    results = []
    for label, state_body, proposition in tests:
        payload = {
            "state": {"subject": f"noul-{label}", "body": state_body},
            "questions": {"q": {"type": "noul", "instructions": proposition}},
        }
        result, ms = timed_post(payload)
        body = result.get("body", {})
        ans = body.get("answers", {}).get("q", {}) if isinstance(body, dict) else {}
        entry = {
            "label": label,
            "status_code": result.get("status_code"),
            "p_true": ans.get("noul"),
            "latency_ms": round(ms, 2),
            "error": body if (isinstance(body, str) or (isinstance(body, dict) and "detail" in body)) else None,
        }
        results.append(entry)
        p = entry["p_true"]
        p_str = f"P(true)={p:.4f}" if p is not None else f"ERR({entry['status_code']})"
        print(f"  {label:<24} → {p_str}")

    return {"test": "noul_mechanical", "results": results}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = get_output_dir()
    verify_provider()
    warm_up(5)

    all_results = {}
    all_results["choice_scaling"] = test_choice_option_scaling()
    all_results["choice_ordering"] = test_choice_ordering_sensitivity()
    all_results["score_bounds"] = test_score_level_bounds()
    all_results["score_ladder"] = test_score_monotonic_ladder()
    all_results["noul_mechanical"] = test_noul_mechanical()

    save_json(out_dir / "primitive_contracts.json", all_results)
    print(f"\nResults saved to {out_dir / 'primitive_contracts.json'}")


if __name__ == "__main__":
    main()
