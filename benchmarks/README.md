# Laya Benchmark & Hardening Harness

**Phase 3.5 (Service Hardening & Operating Envelope) complete 2026-09-29.** Safe operating envelope established. 8 bugs/issues identified. See [HARDENING-REPORT.md](HARDENING-REPORT.md) for findings.

**Next: Use it.** Run actual inferencing. Form intuition through interaction.

Spec: [SPEC-HARDENING.md](SPEC-HARDENING.md). Report: [HARDENING-REPORT.md](HARDENING-REPORT.md).

**Deferred research:** The large-scale semantic evaluation methodology (300-case corpus, inversion pairs, consistency perturbation suite, OOD taxonomy, domain-specific adoption slices, 27B reference comparison) is preserved in [FUTURE-METHODS.md](FUTURE-METHODS.md). It is **not** the current mission and should not be built against without explicit direction.

The service layer is proven (fast, stable, parity-checked). This harness proves the **model layer** and the **service operating envelope**.

---

## Quick start

```bash
# from the repository root, with the venv activated:
# 1. Run the smoke corpus against the live service
python -m benchmarks.harness.runner benchmarks/v1/cases.jsonl

# 2. Compute metrics
python -m benchmarks.harness.metrics results/<date>-raw.jsonl benchmarks/v1/cases.jsonl

# 3. Generate report
.venv/bin/python -m benchmarks.harness.report results/<date>-metrics.json

# 4. (Optional) Run systems benchmark
.venv/bin/python -m benchmarks.harness.systems
```

All commands use the service venv (`.venv/bin/python`). The harness talks to the live HTTP API at `http://127.0.0.1:11448` (override with `--endpoint` or `LAYA_ENDPOINT`).

---

## Directory structure

```
benchmarks/
  v1/                        # frozen smoke corpus (33 cases, regression baseline)
    cases.jsonl              # case corpus (schema below)
    manifest.json            # version, counts, split assignment
  harness/                   # Python modules (quality metrics, runner, report)
    schemas.py               # case schema (pydantic), loader
    runner.py                # HTTP runner + provenance capture
    metrics.py               # per-family model-quality metrics
    systems.py               # systems benchmark (latency, concurrency, cache, VRAM)
    reference.py             # 27B reference runner (deferred — skeleton only)
    report.py                # markdown + JSON report generator
  hardening/                 # Phase 3.5 test scripts (done 2026-09-29)
    _common.py               # shared HTTP/timing/stats utilities
    primitive_contracts.py   # §2: choice/score/noul contract tests
    batching.py              # §3: coalescing window, saturation, heterogeneous
    token_limits.py          # §4: state-length sweep, combined budget, limits
    systems_ops.py           # §5: latency, cold/warm, cache, VRAM, concurrency
    validation_audit.py      # §7: validation probes + envelope assembly
  results/                   # run outputs (gitignored)
  SPEC-HARDENING.md          # mission spec
  HARDENING-REPORT.md        # findings: safe operating envelope, bugs, recommendations
  FUTURE-METHODS.md          # deferred semantic evaluation methodology
  README.md                  # this file
```

---

## Case schema

Each case in `cases.jsonl` is one JSON object per line:

```json
{
  "id": "noul-factual-0001",
  "family": "noul/factual",
  "tier": "deterministic",
  "difficulty": "easy",
  "split": "dev",
  "state": {"subject": "...", "body": "..."},
  "formulation": {
    "type": "noul",
    "questions": {"q1": {"type": "noul", "instructions": "..."}}
  },
  "gold": {"q1": {"answer": true}},
  "variant_of": null,
  "tags": ["billing"],
  "provenance": "hand-authored 2026-09-29",
  "notes": "clear factual billing question"
}
```

**Fields:**
- `id`: unique case identifier
- `family`: metric family (e.g. `choice/multiclass`, `noul/symmetry`)
- `tier`: gold-label tier — `deterministic` (hard gold), `objective` (human-labelled), `subjective` (judgement-based). **Tiers are never pooled into one global accuracy number.**
- `difficulty`: `easy` / `medium` / `hard`
- `split`: `dev` or `lock` — thresholds and pass bars are selected on dev, evaluated on lock
- `state`: the input state (JSON object)
- `formulation`: the question(s) to ask (mirrors `/system_one` request shape)
- `gold`: ground truth per question (shape depends on type)
- `variant_of`: links adversarial/paraphrase variants to their base case
- `tags`: free-form tags for filtering
- `provenance`: who/when/how the case was authored
- `notes`: free-form notes

**Formulation types:**
- `choice`, `score`, `noul`: direct mapping to `/system_one` question types
- `multi_select`: composed — one `noul` per candidate
- `rank`: composed — one `score` per item against a shared rubric

---

## Dev / lock split

The corpus is internally split so that operating choices (thresholds, pass bars, envelope definitions) are made on **development cases** and final reported numbers are computed on **locked cases**. This prevents tuning to the benchmark.

- Split ratio: **~60% dev / ~40% lock** (stratified by family)
- The split is a **corpus property** (fixed in `cases.jsonl`), not a per-run choice
- `variant_of` groups never straddle the boundary
- The report shows dev and lock results side by side

---

## Metrics

Computed per family, per tier, per split. Never one global number.

**choice:** top-1 accuracy, P(gold), confidence, ECE, option-count sensitivity curve
**noul:** AUC, Brier, ECE, accuracy at thresholds, phrasing symmetry
**score:** MAE, exact hit, Spearman ρ, Kendall τ
**multi_select:** precision/recall/F1 at threshold sweep
**rank:** Spearman/Kendall vs gold ordering, pairwise accuracy
**consistency:** decision-flip rate under perturbation
**inversion:** agreement between formulations (choice vs noul)
**routing head:** AUC of `escalate_probability` vs `confidence` for predicting incorrectness

See `harness/metrics.py` for implementation.

---

## Systems benchmark

Measures what the service costs:
- Warm single-call latency by question-shape (50 reps)
- Cold reload latency (5 reps)
- Concurrency sweep: 1/4/16/64 concurrent (≥200 requests per level)
- Cache hit vs miss latency (50 hit, 20 miss)
- Latency vs option count (2 → 25+ options)
- VRAM snapshot via `nvidia-smi`

See `harness/systems.py` for implementation.

---

## Reference comparison (27B)

The 27B model is a **capability/cost comparator**, never an adjudicator. Both Laya and the 27B are evaluated against the same human gold. If Laya disagrees with the 27B and Laya matches the human gold, Laya wins that case.

The reference runner (`harness/reference.py`) is a skeleton — adapt `_call_reference()` to the actual 27B endpoint API.

---

## Adoption-gate policy

1. **General benchmark first** — run the v1 corpus, produce the envelope map (green/yellow/red per family).
2. **Before any production adoption for a specific decision** (e.g. Email Manager "legal obligation", OCR review filtering): author a domain slice (20–50 cases) for *that exact decision*, run it, record pass/fail against a pre-agreed bar. **No slice → no adoption.**
3. **Every production surprise** → capture, label, append to next corpus version (v2), re-run.

---

## Build phases

| Phase | Content | Status |
|---|---|---|
| **A** | Harness skeleton + smoke corpus (33 cases) + end-to-end run + correctness pass | ✅ Done |
| **B** | Service hardening: primitive contracts, coalescing, token limits, concurrency, error handling | ✅ Done (2026-09-29) |
| **C** | Safe operating envelope report + ROADMAP update | ✅ Done (2026-09-29) |
| **D** | Use it. Run actual inferencing. Form intuition through interaction. | ⬜ **Next** |
| **E** | Domain-specific adoption slices (when real use cases emerge) | ⬜ Future |
| **F** | Deferred semantic methods (see FUTURE-METHODS.md) | ⬜ Deferred |

---

## Open decisions

- **D2 — reference model/endpoint:** which local 27B endpoint is canonical? (Deferred — not needed for hardening.)
- **D3 — math deps:** hand-rolled zero-dependency math, validated against known fixtures. (Settled.)
- **D4 — results in git?** Raw outputs gitignored; frozen corpus + manifest + specs + baseline metrics committed. (Settled.)
- **D5 — pass bars:** deferred until after hardening; the safe operating envelope table (§7 of SPEC-HARDENING.md) replaces generic pass bars.

---

## Specs

- [SPEC-HARDENING.md](SPEC-HARDENING.md) — current mission: service hardening & operating envelope.
- [FUTURE-METHODS.md](FUTURE-METHODS.md) — deferred: large-scale semantic evaluation methodology.
