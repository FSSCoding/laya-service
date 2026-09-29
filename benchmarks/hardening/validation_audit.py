"""§7 — Validation and error-handling audit.

Aggregates findings from all hardening tests into the safe operating envelope.
Also runs additional validation-specific probes.

Run: .venv/bin/python -m benchmarks.hardening.validation_audit
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from ._common import (
    get_output_dir, save_json, timed_post, warm_up, verify_provider,
    post_system_one, get_stats, ENDPOINT,
)


def test_input_validation() -> dict:
    """Probe the service's input validation behaviour comprehensively."""
    print("§7 validation: input validation probes")
    results = []

    probes = [
        # (label, payload, expected)
        ("valid_minimal",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "noul", "instructions": "x"}}},
         "200"),
        ("missing_state_key",
         {"questions": {"q": {"type": "noul", "instructions": "x"}}},
         "4xx"),
        ("missing_questions_key",
         {"state": {"subject": "s", "body": "b"}},
         "4xx"),
        ("state_not_dict",
         {"state": "string state", "questions": {"q": {"type": "noul", "instructions": "x"}}},
         "4xx or 200"),
        ("questions_not_dict",
         {"state": {"subject": "s", "body": "b"}, "questions": "string"},
         "4xx"),
        ("empty_questions",
         {"state": {"subject": "s", "body": "b"}, "questions": {}},
         "4xx or 200"),
        ("invalid_type",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "banana", "instructions": "x"}}},
         "4xx"),
        ("choice_no_criteria",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "choice", "instructions": "Pick?"}}},
         "4xx"),
        ("choice_empty_criteria",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "choice", "instructions": "Pick?", "criteria": {}}}},
         "4xx"),
        ("score_no_criteria",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "score", "instructions": "Rate?"}}},
         "4xx"),
        ("score_empty_criteria",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "score", "instructions": "Rate?", "criteria": []}}},
         "4xx"),
        ("noul_no_instructions",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "noul"}}},
         "4xx or 200"),
        ("unicode_in_state",
         {"state": {"subject": "Ünïcödé ☺", "body": "test — à é ï"},
          "questions": {"q": {"type": "noul", "instructions": "OK?"}}},
         "200"),
        ("very_long_instruction",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "noul", "instructions": "x" * 5000}}},
         "200 or truncated"),
        ("duplicate_qids",
         {"state": {"subject": "s", "body": "b"},
          "questions": {"q": {"type": "noul", "instructions": "a"},
                        "q": {"type": "choice", "instructions": "b", "criteria": {"x": "1"}}}},
         "last wins or error"),
    ]

    for label, payload, expected in probes:
        try:
            r = httpx.post(f"{ENDPOINT}/system_one", json=payload, timeout=15.0)
            status = r.status_code
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
            detail = ""
            if isinstance(body, dict):
                if "detail" in body:
                    detail = body["detail"][:80]
                elif "answers" in body:
                    detail = f"answers={list(body['answers'].keys())}"
            elif isinstance(body, str):
                detail = body[:80]
            entry = {"label": label, "status": status, "expected": expected, "detail": detail}
        except Exception as e:
            entry = {"label": label, "status": 0, "expected": expected, "detail": str(e)[:80]}
        results.append(entry)
        print(f"  {label:<24} → {status} (expected {expected}) | {entry['detail']}")

    return {"test": "input_validation", "results": results}


def test_response_shape_consistency() -> dict:
    """Verify response shape is consistent across question types."""
    print("§7b validation: response shape consistency")
    results = {}

    # Choice
    r, _ = timed_post({
        "state": {"subject": "shape", "body": "test"},
        "questions": {"c": {"type": "choice", "instructions": "?",
                            "criteria": {"a": "x", "b": "y"}}},
    })
    ans = r.get("body", {}).get("answers", {}).get("c", {})
    results["choice_fields"] = sorted(ans.keys()) if ans else []

    # Score
    r, _ = timed_post({
        "state": {"subject": "shape", "body": "test"},
        "questions": {"s": {"type": "score", "instructions": "?",
                            "criteria": ["l1", "l2", "l3"]}},
    })
    ans = r.get("body", {}).get("answers", {}).get("s", {})
    results["score_fields"] = sorted(ans.keys()) if ans else []

    # Noul
    r, _ = timed_post({
        "state": {"subject": "shape", "body": "test"},
        "questions": {"n": {"type": "noul", "instructions": "?"}},
    })
    ans = r.get("body", {}).get("answers", {}).get("n", {})
    results["noul_fields"] = sorted(ans.keys()) if ans else []

    # Mixed batch
    r, _ = timed_post({
        "state": {"subject": "shape", "body": "test"},
        "questions": {
            "c": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}},
            "s": {"type": "score", "instructions": "?", "criteria": ["l1", "l2"]},
            "n": {"type": "noul", "instructions": "?"},
        },
    })
    body = r.get("body", {})
    results["mixed_keys"] = sorted(body.get("answers", {}).keys()) if isinstance(body, dict) else []
    results["usage_present"] = "usage" in body if isinstance(body, dict) else False
    results["cached_present"] = "cached" in body if isinstance(body, dict) else False

    for k, v in results.items():
        print(f"  {k}: {v}")

    return {"test": "response_shape", "results": results}


# ── Safe operating envelope assembly ─────────────────────────────────────────

def assemble_envelope(hardening_dir: Path) -> dict:
    """Read all hardening results and produce the safe operating envelope table."""
    print("\n§7c assembling safe operating envelope...")
    envelope = {}

    # Load primitive contracts
    pc_path = hardening_dir / "primitive_contracts.json"
    if pc_path.exists():
        pc = json.loads(pc_path.read_text())
        # Choice: find where it starts failing
        scaling = pc.get("choice_scaling", {}).get("results", [])
        max_working = 0
        first_error = None
        first_truncation = None
        for entry in scaling:
            if entry.get("status_code") == 200:
                max_working = entry["n_options"]
            if entry.get("status_code") in (400, 422) and first_error is None:
                first_error = entry
            if entry.get("truncated") and first_truncation is None:
                first_truncation = entry
        envelope["choice"] = {
            "max_options_working": max_working,
            "first_error": first_error,
            "first_truncation": first_truncation,
            "ordering_sensitive": not pc.get("choice_ordering", {}).get("same_answer", True),
        }

        # Score bounds
        bounds = pc.get("score_bounds", {}).get("results", [])
        valid_levels = [b["n_levels"] for b in bounds if b.get("status_code") == 200]
        rejected_levels = [b["n_levels"] for b in bounds if b.get("status_code") != 200]
        envelope["score"] = {
            "valid_levels": valid_levels,
            "rejected_levels": rejected_levels,
        }

    # Load token limits
    tl_path = hardening_dir / "token_limits.json"
    if tl_path.exists():
        tl = json.loads(tl_path.read_text())
        sweep = tl.get("state_length_sweep", {}).get("results", [])
        max_rendered = max((e.get("input_tokens_rendered") or 0) for e in sweep)
        truncation_points = [e for e in sweep if e.get("truncated")]
        envelope["state_tokens"] = {
            "max_rendered_tokens": max_rendered,
            "truncation_points": truncation_points,
        }

        # Error handling
        errors = tl.get("error_at_limits", {}).get("results", [])
        envelope["error_behaviours"] = [
            {"label": e["label"], "status": e["status_code"], "detail": e.get("error_detail")}
            for e in errors
        ]

    # Load batching
    bat_path = hardening_dir / "batching.json"
    if bat_path.exists():
        bat = json.loads(bat_path.read_text())
        sat = bat.get("saturation", {}).get("results", [])
        max_level = max((s["n_concurrent"] for s in sat), default=0)
        overflow = [s for s in sat if s["n_concurrent"] > 512]
        envelope["batch"] = {
            "max_concurrent_tested": max_level,
            "overflow_behaviour": overflow,
        }

        coalesce = bat.get("coalescing", {}).get("results", [])
        envelope["coalescing_window"] = {
            "offsets_tested": [c["offset_ms"] for c in coalesce],
            "window_size_ms": 5,
        }

    # Load systems ops
    so_path = hardening_dir / "systems_ops.json"
    if so_path.exists():
        so = json.loads(so_path.read_text())
        cw = so.get("cold_warm", {}).get("results", {})
        cache = so.get("cache", {}).get("results", {})
        conc = so.get("concurrency", {}).get("results", [])
        envelope["cold_start_ms"] = cw.get("cold_first_ms")
        envelope["cache"] = {
            "hit_latency_ms": cache.get("hit", {}).get("stats", {}).get("median"),
            "miss_latency_ms": cache.get("miss", {}).get("stats", {}).get("median"),
        }
        if conc:
            max_throughput = max(c["throughput_rps"] for c in conc)
            envelope["max_throughput_rps"] = max_throughput

    # Validation audit
    va_path = hardening_dir / "validation_audit.json"
    if va_path.exists():
        va = json.loads(va_path.read_text())
        probes = va.get("input_validation", {}).get("results", [])
        clean_rejections = [p for p in probes if p["status"] in (400, 422)]
        envelope["validation"] = {
            "n_probes": len(probes),
            "n_clean_rejections": len(clean_rejections),
            "rejection_details": clean_rejections,
        }

    return envelope


def main() -> None:
    out_dir = get_output_dir()
    verify_provider()
    warm_up(5)

    all_results = {}
    all_results["validation"] = test_input_validation()
    all_results["response_shape"] = test_response_shape_consistency()

    save_json(out_dir / "validation_audit.json", all_results)
    print(f"\nValidation results saved to {out_dir / 'validation_audit.json'}")

    # Assemble envelope
    envelope = assemble_envelope(out_dir)
    save_json(out_dir / "safe_operating_envelope.json", envelope)
    print(f"Safe operating envelope saved to {out_dir / 'safe_operating_envelope.json'}")

    # Print summary
    print("\n=== SAFE OPERATING ENVELOPE (summary) ===")
    if "choice" in envelope:
        c = envelope["choice"]
        print(f"  choice: max_options={c['max_options_working']}, "
              f"ordering_sensitive={c['ordering_sensitive']}")
    if "score" in envelope:
        s = envelope["score"]
        print(f"  score: valid_levels={s['valid_levels']}, rejected={s['rejected_levels']}")
    if "state_tokens" in envelope:
        st = envelope["state_tokens"]
        print(f"  state: max_rendered={st['max_rendered_tokens']}, "
              f"truncation_points={len(st['truncation_points'])}")
    if "batch" in envelope:
        b = envelope["batch"]
        print(f"  batch: max_tested={b['max_concurrent_tested']}")
    if "cold_start_ms" in envelope:
        print(f"  cold_start: {envelope['cold_start_ms']}ms")
    if "max_throughput_rps" in envelope:
        print(f"  max_throughput: {envelope['max_throughput_rps']} rps")
    if "validation" in envelope:
        v = envelope["validation"]
        print(f"  validation: {v['n_clean_rejections']}/{v['n_probes']} clean rejections")


if __name__ == "__main__":
    main()
