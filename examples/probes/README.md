# Characterisation Probes

One-off experiment scripts used to characterise how each Laya primitive behaves. They are **not**
part of the test suite and **not** a durable benchmark corpus — they exist to form intuition about
the model's behaviour during development (Phase D, 2026-09-29).

Each probe is a small standalone script that talks to a **live** `laya-service` over HTTP.

| Script | What it probes |
|---|---|
| `choice_probe.py` | Whether `choice` is *relational* (winning probability shifts with distractor quality) vs *absolute* (argmax only). |
| `score_probe.py` | How `score` levels behave across a clean ladder vs a near-synonym ("mushy") ladder. |
| `noul_probe.py` | The negation / factual-mismatch pattern observed in the hardening run, on two different states. |

## Requirements

- A running `laya-service` (the probes call the live HTTP API; defaults to `http://127.0.0.1:11448`).
- The service venv: `.venv/bin/python`.

## Run

```bash
# from the repository root, with the service running:
.venv/bin/python examples/probes/choice_probe.py
.venv/bin/python examples/probes/score_probe.py
.venv/bin/python examples/probes/noul_probe.py
```

Each prints its findings to stdout. Output is exploratory — interpret alongside
`benchmarks/HARDENING-REPORT.md`, which is the canonical hardening result.

## Notes

- These are intentionally disposable. If a probe becomes load-bearing, promote it into the
  `benchmarks/harness/` suite with proper case/gold structure instead of keeping it here.
- The findings these probes informed are recorded in `docs/MODEL-DOCTRINE.md` (safe-usage rules per
  primitive) and `benchmarks/HARDENING-REPORT.md`.
