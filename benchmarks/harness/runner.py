"""HTTP runner: executes cases against the live service, records raw responses + provenance.

Outputs raw.jsonl with one record per case: case metadata, request, response, timing, provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Model bundle location (for provenance capture); defaults to ./onnx.
_MODEL_DIR = Path(os.environ.get("LAYA_MODEL_DIR", "./onnx"))

# Add service src to path so we can import laya_server modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from laya_server.client import resolve_endpoint, system_one, health  # noqa: E402

from .schemas import Case, load_cases  # noqa: E402


# Model constraints (from onnx/laya_config.json)
MAX_LEN = 512
HEAD_MAX_LEN = 192


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens using the Laya tokenizer."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def get_tokenizer():
    """Load the Laya tokenizer (cached on first call)."""
    try:
        from transformers import AutoTokenizer
        tokenizer_path = str(_MODEL_DIR / "tokenizer")
        return AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Warning: could not load tokenizer: {e}", file=sys.stderr)
        return None


def build_request(case: Case) -> dict[str, Any]:
    """Build the /system_one request from a case's formulation."""
    form = case.formulation
    if form.type in ("choice", "score", "noul"):
        # Direct mapping: formulation.questions is already the request shape
        return {"state": case.state, "questions": form.questions}
    elif form.type == "multi_select":
        # multi_select is composed: one noul per candidate
        # formulation.questions should have the candidate list
        instructions = form.questions.get("instructions", "")
        candidates = form.questions.get("candidates", [])
        questions = {}
        for cand in candidates:
            questions[cand] = {
                "type": "noul",
                "instructions": f"{instructions} — {cand}",
            }
        return {"state": case.state, "questions": questions}
    elif form.type == "rank":
        # rank is composed: one score per item against shared rubric
        instructions = form.questions.get("instructions", "")
        items = form.questions.get("items", [])
        rubric = form.questions.get("rubric", [])
        questions = {}
        for item in items:
            questions[item] = {
                "type": "score",
                "instructions": f"{instructions} — {item}",
                "criteria": rubric,
            }
        return {"state": case.state, "questions": questions}
    else:
        raise ValueError(f"Unknown formulation type: {form.type}")


def compute_provenance(
    case: Case,
    response: dict,
    tokenizer,
) -> dict[str, Any]:
    """Compute token accounting and truncation detection."""
    # State tokens (if tokenizer available)
    state_text = json.dumps(case.state)
    state_tokens = count_tokens(state_text, tokenizer) if tokenizer else None

    # Response metadata
    usage = response.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)

    # Question metadata
    questions = response.get("_request_questions", {})  # injected by runner
    n_questions = len(questions)
    max_options = 0
    for q in questions.values():
        if q["type"] == "choice":
            max_options = max(max_options, len(q.get("criteria", {})))
        elif q["type"] == "score":
            max_options = max(max_options, len(q.get("criteria", [])))

    # Truncation detection
    truncated = input_tokens >= MAX_LEN

    # Budget remaining (approximate)
    head_budget_remaining = HEAD_MAX_LEN - (input_tokens // 2) if input_tokens else None
    body_budget_remaining = MAX_LEN - HEAD_MAX_LEN - (input_tokens // 2) if input_tokens else None

    return {
        "state_tokens": state_tokens,
        "input_tokens": input_tokens,
        "n_questions": n_questions,
        "max_options": max_options,
        "truncated": truncated,
        "head_budget_remaining": head_budget_remaining,
        "body_budget_remaining": body_budget_remaining,
    }


def collect_run_provenance(endpoint: str) -> dict[str, Any]:
    """Collect full runtime/model provenance for this run."""
    prov = {}

    # Service info from /health
    try:
        h = health(endpoint)
        prov["service_version"] = h.get("version", "unknown")
        prov["provider"] = h.get("providers", ["unknown"])[0] if h.get("providers") else "unknown"
    except Exception as e:
        prov["service_version"] = "unknown"
        prov["provider"] = f"error: {e}"

    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        prov["service_git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        prov["service_git_commit"] = "unknown"

    # ONNX model hash
    try:
        model_path = _MODEL_DIR / "laya.onnx"
        if model_path.exists():
            prov["onnx_model_hash"] = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
        else:
            prov["onnx_model_hash"] = "unknown"
    except Exception:
        prov["onnx_model_hash"] = "unknown"

    # ONNX config hash
    try:
        config_path = _MODEL_DIR / "laya_config.json"
        if config_path.exists():
            prov["onnx_config_hash"] = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
        else:
            prov["onnx_config_hash"] = "unknown"
    except Exception:
        prov["onnx_config_hash"] = "unknown"

    # Service config hash
    try:
        svc_config_path = Path("config/laya_config.json")
        if svc_config_path.exists():
            prov["service_config_hash"] = hashlib.sha256(svc_config_path.read_bytes()).hexdigest()[:16]
        else:
            prov["service_config_hash"] = "unknown"
    except Exception:
        prov["service_config_hash"] = "unknown"

    # Python version
    prov["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Package version
    try:
        from laya_server import __version__
        prov["package_version"] = __version__
    except Exception:
        prov["package_version"] = "unknown"

    # Run date
    prov["run_date"] = datetime.now(timezone.utc).isoformat()

    return prov


def run_cases(
    cases_path: Path,
    endpoint: str | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """Run all cases against the live service. Returns list of raw records."""
    endpoint = resolve_endpoint(endpoint)
    cases = load_cases(cases_path)
    tokenizer = get_tokenizer()

    # Collect run-level provenance once
    run_prov = collect_run_provenance(endpoint)
    run_id = run_prov["run_date"]

    records = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id} ({case.family}, {case.split})")

        # Build request
        request = build_request(case)

        # Inject questions for provenance computation
        request["_request_questions"] = request["questions"]

        # Execute with timing
        t0 = time.time()
        try:
            response = system_one(endpoint, request["state"], request["questions"])
            timing_ms = (time.time() - t0) * 1000
            error = None
        except Exception as e:
            timing_ms = (time.time() - t0) * 1000
            response = None
            error = str(e)

        # Compute provenance
        if response:
            prov = compute_provenance(case, response, tokenizer)
        else:
            prov = {"error": error}

        # Build record
        record = {
            "case_id": case.id,
            "split": case.split,
            "family": case.family,
            "tier": case.tier,
            "difficulty": case.difficulty,
            "request": {"state": request["state"], "questions": request["questions"]},
            "response": response,
            "timing_ms": timing_ms,
            "provenance": prov,
            "run_id": run_id,
            "run_provenance": run_prov,
        }
        records.append(record)

    # Write output
    if output_path is None:
        output_path = cases_path.parent.parent / "results" / f"{run_id[:10]}-raw.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(records)} records to {output_path}")
    return records


def main():
    """CLI entry point: python -m benchmarks.harness.runner <cases.jsonl> [--endpoint URL]"""
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmark cases against Laya service")
    parser.add_argument("cases", type=Path, help="Path to cases.jsonl")
    parser.add_argument("--endpoint", type=str, default=None, help="Service endpoint URL")
    parser.add_argument("--output", type=Path, default=None, help="Output path for raw.jsonl")
    args = parser.parse_args()

    run_cases(args.cases, args.endpoint, args.output)


if __name__ == "__main__":
    main()
