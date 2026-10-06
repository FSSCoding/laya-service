#!/usr/bin/env python3
"""Noul characterisation — semantic compatibility, not objective truth.

Tests the intuition: noul ≈ "how compatible is this proposition with the
representation of this state?" (NOT "is this proposition objectively true?")

Probes per state, ordered by expected compatibility strength:
  1. lexical identity        (restates the state's own noun)
  2. synonym                 (same referent, different word)
  3. category relation       (hyponym → hypernym entailment)
  4. world-knowledge attr    (true attribute requiring a knowledge chain)
  5. false attribute         (objectively false attribute, negative control)

Two states to check the pattern generalises:
  - "a basketball"  (object)
  - "an email"      (document — closer to noul's designed judgment domain)

Run:  .venv/bin/python examples/probes/noul_probe.py
(disposable experiment script — not a durable test corpus)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# (state, [(qid, proposition)])
STATES = [
    (
        "a basketball",
        [
            ("n1", "The object is a basketball."),          # lexical
            ("n2", "The object is a sphere."),               # synonym-ish (ball→sphere)
            ("n3", "The object is a ball."),                 # category (hypernym)
            ("n4", "The object is round."),                  # world-knowledge attr
            ("n5", "The object is square."),                 # false attr (control)
        ],
    ),
    (
        "an email",
        [
            ("n1", "The document is an email."),            # lexical
            ("n2", "The document is a message."),           # synonym
            ("n3", "The document is a communication."),     # category (hypernym)
            ("n4", "The document was written by a person."),# world-knowledge attr
            ("n5", "The document is a novel."),             # false attr (control)
        ],
    ),
]


def main() -> int:
    for state, probes in STATES:
        questions = {
            qid: {"type": "noul", "instructions": prop} for qid, prop in probes
        }
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
                print(f"[{state}] timed out", file=sys.stderr)
                return 1

        if out.returncode != 0:
            print(f"[{state}] {out.stderr}", file=sys.stderr)
            return out.returncode

        result = json.loads(out.stdout)

        labels = ["lexical", "synonym", "category", "attr", "false-ctrl"]
        print(f"state: {state!r}")
        print(f"  {'#':<3} {'P(true)':<8} {'bar':<20} {'kind':<10} proposition")
        print(f"  {'-'*68}")
        for i, (qid, prop) in enumerate(probes):
            p = result["answers"][qid].get("noul", float("nan"))
            bar = "█" * int(p * 40)
            print(f"  {i+1:<3} {p:<8.4f} {bar:<20} {labels[i]:<10} {prop}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
