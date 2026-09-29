#!/usr/bin/env python3
"""Score characterisation — ordered semantic projection onto a supplied scale.

Tests the intuition: score ≈ state representation → compatibility with
ordered level descriptions → distribution across levels → weighted mean.

Design: fix the level ladder; vary the state from mildest to most severe.
If the expected-position scalar moves monotonically with severity, score
is behaving as an ordered semantic placement (not a flat number guess).

Two ladders:
  1. Clean ladder — semantically distinct, evenly spaced anchors
  2. Mushy ladder — near-synonym anchors (tests whether scale quality matters)

Run:  .venv/bin/python campaign/score-probe.py
(disposable experiment script — not a durable test corpus)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Ladder 1: clean, distinct, evenly-meaningful anchors (index 0 = lowest).
CLEAN_LEVELS = [
    "no impact",
    "minor inconvenience",
    "degraded service",
    "major outage",
    "complete outage",
]

# Ladder 2: mushy, near-synonym anchors (quality of the scale is poor).
MUSHY_LEVELS = [
    "bad",
    "okay",
    "decent",
    "pretty good",
    "good",
    "really good",
    "excellent",
]

INSTRUCTIONS = "How severe is the impact of this ticket on the customer?"

# States ordered mildest → most severe (5 steps).
STATES = [
    ("s1_mild",
     "Customer reports a minor typo in the invoice PDF; "
     "asks for a corrected copy by email."),
    ("s2_low",
     "Customer says the dashboard loads slowly (~5s) on peak hours; "
     "functionality is intact, just laggy."),
    ("s3_medium",
     "Nightly backup has been failing for 2 days; "
     "customer cannot retrieve yesterday's data until it's fixed."),
    ("s4_high",
     "All customers are locked out of the platform since the morning deploy; "
     "no data accessible, support queue is full."),
    ("s5_critical",
     "Complete data loss on production database after a failed migration; "
     "all customer records gone, no backup available, business cannot operate."),
]


def run_score(state: str, instructions: str, levels: list[str]) -> tuple[float, dict]:
    """Call the CLI score verb; return (score_value, probabilities_dict)."""
    questions = {"q": {
        "type": "score",
        "instructions": instructions,
        "criteria": levels,
    }}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        sp = tmpdir / "state.json"
        sp.write_text(json.dumps(state))
        qp = tmpdir / "questions.json"
        qp.write_text(json.dumps(questions))
        try:
            out = subprocess.run(
                [sys.executable, "-m", "laya_server", "batch",
                 "--state", str(sp), "--questions", str(qp), "--json"],
                capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("timed out")
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    result = json.loads(out.stdout)
    ans = result["answers"]["q"]
    return ans.get("score", float("nan")), ans.get("probabilities", {})


def main() -> int:
    for ladder_name, levels in [("CLEAN", CLEAN_LEVELS), ("MUSHY", MUSHY_LEVELS)]:
        print(f"ladder: {ladder_name} ({len(levels)} levels)")
        print(f"  levels: {levels}\n")
        print(f"  {'state':<12} {'score':<8} {'expected_level':<16} probabilities")
        print(f"  {'-'*78}")
        for label, state in STATES:
            score_val, probs = run_score(state, INSTRUCTIONS, levels)
            # Expected level index (0-based) as a more interpretable number.
            max_idx = len(levels) - 1
            exp_level = score_val * max_idx if max_idx > 0 else 0.0
            # Compact probability display.
            prob_str = " ".join(f"{v:.2f}" for v in probs.values())
            print(f"  {label:<12} {score_val:<8.4f} {exp_level:<16.2f} [{prob_str}]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
