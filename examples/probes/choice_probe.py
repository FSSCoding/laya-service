#!/usr/bin/env python3
"""Choice characterisation — relative semantic competition.

Tests the intuition: choice ≈ argmax over semantic compatibility among
supplied candidates. The winning probability is RELATIVE, not absolute:
it should shift when you change the distractors, even if the winning
option and the state stay fixed.

Design: fix state + correct option; vary distractor quality.

  A. strong distractors  — genuinely plausible alternatives
  B. weak distractors    — obviously wrong / unrelated options
  C. many distractors    — 10+ options, most irrelevant
  D. near-synonym        — distractor that overlaps the winner semantically

If the winning prob drops a lot with strong distractors vs rises with
weak ones, choice is strongly relational. If it barely moves, it's more
absolute.

Run:  .venv/bin/python examples/probes/choice_probe.py
(disposable experiment script — not a durable test corpus)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

STATE = (
    "Ticket #4821: Customer says the nightly backup has been failing for "
    "three nights after the storage upgrade. Error shows insufficient space "
    "on /backup. Customer is on enterprise plan and requesting a refund "
    "if not resolved by Friday."
)

INSTRUCTIONS = "Which team should handle this ticket?"

# Correct option held fixed across all conditions.
WINNER = ("technical_support", "Faults, outages, configuration, storage and backup failures")

CONDITIONS = {
    "A_strong_distractors": [
        WINNER,
        ("billing", "Refunds, invoices, subscriptions, payment disputes"),
        ("infrastructure", "Datacenter, network, and hardware provisioning issues"),
        ("compliance", "Data retention, privacy, and regulatory audit requests"),
    ],
    "B_weak_distractors": [
        WINNER,
        ("marketing", "Brand, social media, and advertising campaigns"),
        ("hr", "Employee onboarding, offboarding, and benefits questions"),
    ],
    "C_many_distractors": [
        WINNER,
        ("billing", "Refunds, invoices, subscriptions, payment disputes"),
        ("marketing", "Brand, social media, and advertising campaigns"),
        ("hr", "Employee onboarding, offboarding, and benefits questions"),
        ("legal", "Contract disputes, IP, and terms-of-service interpretation"),
        ("sales", "Pre-sales inquiries, quotes, and demo scheduling"),
        ("support_l1", "Password resets, account lockouts, and login issues"),
        ("data_science", "Model training, feature engineering, and ML ops"),
        ("devops", "CI/CD pipelines, deployment automation, and infra-as-code"),
        ("product", "Feature requests, roadmap feedback, and UX reports"),
    ],
    "D_near_synonym": [
        WINNER,
        ("storage_admin", "Disk capacity, volume management, and backup scheduling"),
    ],
}


def main() -> int:
    print(f"state: {STATE[:60]}...\n")
    print(f"{'condition':<22} {'winner_prob':<12} {'margin':<10} winner")
    print("-" * 72)

    for name, options in CONDITIONS.items():
        # Build criteria dict: {label: description}
        criteria = {label: desc for label, desc in options}
        questions = {"q": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": criteria,
        }}
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            sp = tmpdir / "state.json"
            sp.write_text(json.dumps(STATE))
            qp = tmpdir / "questions.json"
            qp.write_text(json.dumps(questions))
            try:
                out = subprocess.run(
                    [sys.executable, "-m", "laya_server", "batch",
                     "--state", str(sp), "--questions", str(qp), "--json"],
                    capture_output=True, text=True, timeout=60,
                )
            except subprocess.TimeoutExpired:
                print(f"[{name}] timed out", file=sys.stderr)
                return 1

        if out.returncode != 0:
            print(f"[{name}] {out.stderr}", file=sys.stderr)
            return out.returncode

        result = json.loads(out.stdout)
        ans = result["answers"]["q"]
        probs = ans.get("probabilities", {})
        winner_prob = probs.get("technical_support", float("nan"))
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        n_opts = len(options)
        print(f"{name:<22} {winner_prob:<12.4f} {margin:<10.4f} "
              f"({n_opts} options)")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
