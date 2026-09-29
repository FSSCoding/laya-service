"""27B reference runner — comparator only, never adjudicator.

Runs the same cases through the main local 27B model with forced structured output.
Both models are scored against the same human gold; the 27B is a capability/cost
comparator, not a ground-truth source.

Endpoint and model are supplied via config — no ports or model names hardcoded.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .runner import collect_run_provenance, build_request
from .schemas import Case, load_cases


def run_reference(
    cases_path: Path,
    reference_endpoint: str,
    reference_model: str,
    output_path: Path | None = None,
    reps: int = 3,
) -> list[dict]:
    """Run all cases against the 27B reference. Returns list of raw records.

    The 27B endpoint must accept a JSON body and return structured output matching
    the question type. The exact protocol is endpoint-specific — this runner sends
    a generic prompt + schema and records whatever comes back.

    Args:
        cases_path: Path to cases.jsonl
        reference_endpoint: URL of the 27B model endpoint
        reference_model: Model identifier (for provenance)
        output_path: Where to write reference-raw.jsonl
        reps: Number of repetitions per case (≥3 for stochastic model)
    """
    cases = load_cases(cases_path)
    run_date = datetime.now(timezone.utc).isoformat()

    records = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id} ({case.family})")

        request = build_request(case)
        all_responses = []

        for rep in range(reps):
            t0 = time.time()
            try:
                # Generic reference call — endpoint-specific protocol
                # The 27B runner must be adapted to the actual endpoint's API
                resp = _call_reference(reference_endpoint, reference_model, request)
                timing_ms = (time.time() - t0) * 1000
                all_responses.append({"response": resp, "timing_ms": timing_ms, "error": None})
            except Exception as e:
                timing_ms = (time.time() - t0) * 1000
                all_responses.append({"response": None, "timing_ms": timing_ms, "error": str(e)})

        record = {
            "case_id": case.id,
            "split": case.split,
            "family": case.family,
            "tier": case.tier,
            "difficulty": case.difficulty,
            "request": request,
            "repetitions": all_responses,
            "reference_model": reference_model,
            "reference_endpoint": reference_endpoint,
            "run_date": run_date,
        }
        records.append(record)

    # Write output
    if output_path is None:
        output_path = cases_path.parent.parent / "results" / f"{run_date[:10]}-reference-raw.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(records)} reference records to {output_path}")
    return records


def _call_reference(endpoint: str, model: str, request: dict) -> dict:
    """Call the 27B reference endpoint.

    This is a placeholder — the actual protocol depends on the endpoint.
    Common patterns:
    - OpenAI-compatible: POST /v1/chat/completions with response_format
    - vLLM: POST /v1/completions with guided_json
    - Custom: POST /generate with prompt + schema

    Adapt this function to the actual endpoint being used.
    """
    # Placeholder implementation — returns a stub response
    # In production, this would format the request for the specific endpoint
    # and parse the structured response back into the expected shape.
    raise NotImplementedError(
        f"Reference endpoint protocol not implemented for {endpoint}. "
        "Adapt _call_reference() to the actual 27B endpoint API."
    )


def main():
    """CLI: python -m benchmarks.harness.reference <cases.jsonl> --endpoint URL --model NAME"""
    import argparse
    parser = argparse.ArgumentParser(description="Run 27B reference comparison")
    parser.add_argument("cases", type=Path, help="Path to cases.jsonl")
    parser.add_argument("--endpoint", type=str, required=True, help="27B model endpoint URL")
    parser.add_argument("--model", type=str, required=True, help="Model identifier")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per case (default: 3)")
    args = parser.parse_args()

    run_reference(args.cases, args.endpoint, args.model, args.output, args.reps)


if __name__ == "__main__":
    main()
