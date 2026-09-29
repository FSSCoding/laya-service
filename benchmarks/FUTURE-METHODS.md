# Laya Evaluation — Future Methods (deferred research)

> **STATUS: DEFERRED.** This document preserves the proposed large-scale semantic evaluation
> methodology (300-case corpus, inversion pairs, consistency perturbation suite, OOD taxonomy,
> domain-specific adoption slices, 27B reference comparison). It is **not** the current mission.
>
> The current mission is **Service Hardening & Operating Envelope** — see `SPEC-HARDENING.md`.
>
> These methods are preserved so they can be resumed quickly when a real use case requires
> deeper semantic characterisation. Do not build against this spec without explicit direction.

---

# Laya Evaluation Harness — Scope & Spec (v1.0, 2026-09-29)

**Status:** Phase A complete (harness built, smoke corpus authored, end-to-end pipeline proven). Phases B–E deferred.
**Goal:** characterise the Laya 421M model — where it is trustworthy, where it is a weak signal,
where it must never decide anything — *before* it is wired into Email Manager, OCR, telemetry, or
any operational path. The service layer is proven; this proves the **model layer**.

**Non-goals:** no platform, no service changes, no new primitives, no adapters. The harness is a
benchmark/research asset that lives in the repo and talks to the existing HTTP API.

**v0.2 changes vs v0.1** (review feedback incorporated):
1. Dev/lock split inside the corpus so thresholds and bars are not tuned on the cases they report on (§3.1, §4.1).
2. Full runtime/model provenance captured at every run, not just service version (§8.1).
3. Token accounting on the rendered sequence (state + questions + syntax), not raw state length (§3.2, §6.1).
4. Percentile statistics backed by adequate sample counts per measurement class (§6.2).
5. 27B explicitly a reference comparator, never an adjudicator over human gold (§7.1).

---

## 1. Architecture: two halves + one experiment

```
benchmarks/
  v1/                        # frozen, versioned corpus (never edited in place — new version dir)
    cases.jsonl              # the case corpus (schema §3); each case has split ∈ {dev, lock}
    manifest.json            # version, date, counts per family/tier/difficulty/split, provenance notes
  harness/                   # Python, uses laya_server.client / primitives — no GPU-side imports
    schemas.py               # case schema (pydantic), load + validate cases.jsonl
    runner.py                # runs cases against the live service, records raw responses + timing + provenance (§8.1)
    metrics.py               # per-type model-quality metrics (§5); dev-split selection, lock-split evaluation
    systems.py               # systems benchmark: latency/tokens/cache/reload/VRAM (§6)
    reference.py             # 27B reference runner (forced structured output, comparator only) (§7)
    report.py                # aggregate + per-family markdown/JSON report (§8); --diff regression mode
  results/                   # run outputs: results/<date>-<tag>/{raw.jsonl, metrics.json, report.md, provenance.json}
  README.md                  # how to run, how to freeze a baseline, adoption-gate policy
```

- Harness runs **through HTTP** (`http://127.0.0.1:11448`, `LAYA_ENDPOINT` override) so it
  measures what a real consumer measures. Systems benchmark additionally does one in-process
  pass (like current `bench.py`) to isolate HTTP overhead.
- Existing `bench.py` stays as-is (quick CPU-vs-GPU sanity). `systems.py` supersedes it for
  formal runs; it may import the same fixtures.
- Harness deps: stdlib + what the package already ships (httpx, pydantic, click). Add
  `scipy`/`numpy` **only** to a `[bench]` extra (Spearman/Kendall/AUC/ECE math; could be
  hand-rolled if we want zero new deps — decision D3).

## 2. Determinism note (shapes the whole design)

Laya inference is **deterministic** — temperature-scaled softmax, no sampling. Repeated identical
calls return identical results within FP noise (CPU-vs-GPU ~1e-4). Therefore:

- Run-to-run "stability" tests are cheap and mostly verify determinism itself (assert max abs
  delta across N repeats ≈ 0, flag anything > 1e-3 as a harness/service bug).
- The meaningful stability axis is **perturbation consistency** (§5.6): does the decision change
  when wording/order/irrelevant content changes? That is where a 421M model will show its limits.
- The **27B reference is stochastic** → per the standing rule, reference metrics need ≥3 runs,
  reported as mean ± spread. Never a single before/after.

## 3. Case schema (cases.jsonl, one JSON object per line)

```json
{
  "id": "choice-dept-0001",
  "family": "choice/multiclass",
  "tier": "objective",
  "difficulty": "easy",
  "split": "lock",
  "state": {"subject": "...", "body": "..."},
  "formulation": {
    "type": "choice",
    "questions": {
      "q1": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}
    }
  },
  "gold": {
    "q1": {"answer": "a", "acceptable": ["b"], "score": null, "labels": null}
  },
  "variant_of": null,
  "tags": ["email-manager", "routing"],
  "provenance": "hand-authored 2026-09-30 by agent, human-reviewed",
  "notes": "overlapping categories on purpose — tests choice vs multi_select formulation"
}
```

- `tier` ∈ `deterministic` (hard gold, e.g. exact string match), `objective` (human-labelled,
  reasonably objective), `subjective` (inherently judgement-based). **Tiers are never pooled into
  one global accuracy number** — every aggregate is per-tier and per-family first.
- `split` ∈ `dev` | `lock` (see §3.1). A case belongs to exactly one split; the split is part of
  the frozen corpus, not a per-run choice.
- `gold` shape depends on type: choice → `answer` + `acceptable` alternatives; noul →
  `answer: true/false`; score → ordinal `score` (0..K-1) ± `tolerance`; multi_select → `labels`
  set; rank → gold ordering (list of item ids).
- `variant_of`: links adversarial/paraphrase variants to their base case (drives consistency
  metrics — §5.6).
- Frozen at v1. New production surprises get labelled and appended in **v2**, never by editing v1.

### 3.1 Dev / lock split

The corpus is internally split so that operating choices (thresholds, pass bars, envelope
definitions) are made on **development cases** and final reported numbers are computed on
**locked cases**. This prevents tuning to the benchmark.

- Split ratio: **70% dev / 30% lock**, stratified by family (each family contributes
  proportionally to both splits). Exact case-level assignment is fixed in the frozen corpus —
  it is a corpus property, not a runner option.
- **Dev cases** may be used to: choose multi-select operating thresholds, select score tolerance,
  define green/yellow/red pass bars, sanity-check metric implementations, explore behaviour.
- **Lock cases** are used only to compute the final reported numbers. No metric implementation,
  threshold choice, or bar definition is allowed to read lock-case outcomes during development.
- The report shows dev and lock results side by side. If dev and lock diverge materially
  (e.g. multi-select F1 differs >10 pp between splits), the report flags it as a
  generalisation warning.
- Consistency-suite and inversion-pair cases are **always assigned to the same split as their
  base case** (via `variant_of`), so perturbation/inversion comparisons never straddle the
  dev/lock boundary.

### 3.2 Rendered-sequence token accounting

The model's `max_len: 512` budget applies to the **rendered sequence**: state tokens +
instruction tokens + criteria/option tokens + sequence-syntax tokens (markers, delimiters)
that `sequence.py` inserts. Raw state length alone does not predict truncation.

For every case the runner records (in `raw.jsonl` alongside the response):

| Field | Meaning |
|---|---|
| `state_tokens` | Token count of the state field alone (tokenizer) |
| `input_tokens` | Total rendered sequence tokens (from `/system_one` response `usage.input_tokens`) |
| `n_questions` | Number of questions in the request |
| `max_options` | Max option/level count across all questions |
| `truncated` | Boolean — true if `input_tokens >= max_len` or the worker logged a truncation |
| `head_budget_remaining` | `head_max_len` − tokens consumed by the head portion (if derivable) |
| `body_budget_remaining` | `max_len` − `head_max_len` − tokens consumed by body portion (if derivable) |

Truncation detection: the harness can also independently render the sequence using
`laya_server.sequence` (same code path as the worker) and compare token counts to detect
whether the sequence would exceed `max_len`. This gives a **predicted truncation** field even
for cases that might fail.

The state-shape probe family includes deliberate cases at and past the 512-token rendered
budget to document the degradation behaviour, with the `truncated` field confirming which
portion was cut.

## 4. Corpus plan (v1 target: ~300 cases across families)

Authored deliberately, not synthetically mass-generated. Every case carries provenance, a
human-review flag, and a `split` assignment. Distribution plan:

| Family group | Families | ~Cases |
|---|---|---|
| **noul** | clear factual binary; ambiguous binary; adversarial/misleading wording; positive/negative phrasing **pairs** (same semantics, inverted wording — `variant_of` linked) | 70 |
| **choice** | 2-option; 3–5 option; 6–10; 11–20; **25+ (past the practical ceiling — expected to collapse; we document where)**; overlapping categories (choice-is-wrong-formulation cases) | 70 |
| **score** | 3-level; 4-level urgency; 7-level; 10-level (max); subtle-boundary pairs (adjacent states that should score adjacently) | 45 |
| **multi_select** | clean disjoint labels; genuinely overlapping labels; none-apply cases; all-apply cases | 30 |
| **rank** | 3–5 items shared rubric; 8–10 items; near-tie sets (should be unstable — measure it) | 25 |
| **state-shape probes** | structured JSON state; long text (near/at the **512-token rendered budget** — include deliberately-over cases to document truncation); very short state; noisy state with irrelevant info; missing-information case | 25 |
| **consistency suite** | perturbations of ~10 base cases: option reorder, option-id rename (descriptions preserved), irrelevant text added, evidence moved early/late, instruction paraphrase, JSON key reorder, duplicated irrelevant fields | 10 bases × ~4 variants ≈ 35 (counted in the families above via `variant_of`) |
| **inversion pairs** | same semantic problem as (a) one multiclass choice, (b) N binary nouls, (c) hierarchical choice, (d) score/rubric where sensible | 10 problems × 2–4 formulations ≈ 25 |
| **out-of-distribution** | nonsense states, wrong-domain questions, empty-ish states — expected-to-fail cases; gold = "should not be confident" | 10 |

Domain flavour biased toward Brett's actual intended uses (email triage/actionability, OCR
review filtering, ticket routing, urgency) so the map is relevant to the adoption decisions,
not generic.

**Gold policy:** `deterministic` and `objective` tiers get a single human-approved gold.
`subjective` tier gets gold + a documented "reasonable range"; metrics on subjective cases
report distributional agreement, not binary accuracy.

### 4.1 Split assignment

After all ~300 cases are authored and gold-labelled, the dev/lock split is assigned:

1. Within each family, shuffle cases (seeded RNG, seed recorded in `manifest.json`).
2. Assign 70% to `dev`, 30% to `lock`.
3. For `variant_of` groups (consistency, inversion): all members take the split of the base
   case (whichever has the highest difficulty / is most representative).
4. Record the assignment in `manifest.json` and stamp `split` on each case.
5. Once the corpus is frozen, the split is immutable.

The split is applied **before any run** — there is no "first run is dev, second is lock"
temporal split. Both splits are always run together; the discipline is in what you *do*
with each set of results.

## 5. Model-quality metrics (metrics.py)

Computed per family, per tier, per split (dev/lock shown side by side), and as per-family
rollups. Never one global number.

> **Threshold-selection discipline:** any parameter that is chosen by looking at results
> (multi-select threshold, score tolerance, pass bars) must be selected on **dev cases only**,
> then evaluated on **lock cases**. The report states which split was used for each
> selection and shows both splits' results.

### 5.1 choice
- top-1 accuracy (gold ∈ `answer`, with `acceptable` counted separately as "soft hit")
- top-2 accuracy; P(gold) distribution (mean/median/p10)
- confidence (normalized entropy) vs correctness correlation
- **option-count sensitivity curve**: accuracy binned by number of options (2, 3–5, 6–10, 11–20, 25+)
- calibration: ECE + reliability histogram of P(chosen) vs hit rate

### 5.2 noul
- accuracy at thresholds {0.3, 0.4, 0.5, 0.6, 0.7}; ROC/AUC; precision/recall/F1 at each
- Brier score; calibration curve (binned P(true) vs empirical true rate)
- **phrasing symmetry**: for positive/negative pairs, flip-consistency rate + |P − (1−P′)| delta
- operating threshold (if one is selected): chosen on dev split, reported on both splits

### 5.3 score
- MAE / exact-hit against gold ordinal (with tolerance for subjective tier)
- Spearman + Kendall τ of predicted vs gold ordinal **within each family** (needs ≥5 ordinal levels across cases in the family)
- monotonicity: for subtle-boundary pairs, does the ordering hold?
- full-distribution sanity: probability mass on levels adjacent to gold vs far

### 5.4 multi_select
- precision / recall / F1 over selected labels at threshold sweep {0.3…0.8}
- recommended operating threshold per family: **selected on dev split only** (best F1),
  then evaluated on lock split. Report shows the dev-selected threshold's lock performance.
- if no threshold is selected (i.e. reporting raw probabilities), label the table "pre-threshold".

### 5.5 rank
- Spearman / Kendall τ vs gold ordering; pairwise ordering accuracy
- near-tie stability: how often do near-tied items flip under rubric rewording
- verdict on the honest question: **does independent shared-rubric scoring produce a meaningful total ordering at all?**

### 5.6 consistency (adversarial suite)
- per perturbation kind: decision-flip rate vs base case, and probability-shift magnitude
- ranked "fragility table": which perturbations matter most (expected: option reorder and id-rename on choice; evidence position on long states)
- consistency metrics use all cases in the family regardless of split (the perturbation
  comparison is intra-family, not a threshold-selection exercise), but the *aggregate*
  fragility rate is reported per split.

### 5.7 inversion pairs
- per problem: agreement between formulations (choice argmax vs per-class noul argmax vs hierarchical result)
- probability-structure comparison: does the noul-vector formulation give better-separated probabilities than the softmax over many options?
- output: an evidence-based answer to "is query inversion an engineering technique or just interesting?"

### 5.8 routing-head experiment (its own section in every report)
For every labelled case record: correct/incorrect, confidence, act_probability, escalate_probability. Then:
- AUC of `escalate_probability` for predicting **incorrectness** (the only question that matters)
- AUC of entropy `confidence` for predicting incorrectness — head-to-head
- point-biserial correlations; calibration of escalate on failures
- **Decision rule written into the report:** if escalate-AUC ≈ 0.5 or worse than plain
  confidence, it is declared operationally dead (kept experimental, never consumed).

## 6. Systems benchmark (systems.py)

Separate from quality; measures what the service costs. Sample counts are class-dependent —
percentiles require adequate N (see §6.2).

### 6.1 Rendered-sequence token accounting in systems runs

Every systems-benchmark request records the same token fields as the quality runner
(§3.2): `state_tokens`, `input_tokens`, `n_questions`, `max_options`, `truncated`. This lets
the latency-vs-state-length and latency-vs-option-count curves be plotted against **rendered
tokens**, not raw state length. The option-count scaling sweep (2 → 10 → 20 → 25+ options)
records `input_tokens` at each step to show exactly how head budget is consumed.

### 6.2 Sample counts per measurement class

| Measurement | Reps | Statistics reported | Why |
|---|---|---|---|
| Warm single-call latency (per question-shape) | **50** | min, median, p95, p99, mean | p95/p99 need ≥50 for a meaningful tail |
| Cold reload latency | **5** | min, median, max | Expensive (~1 s each); 5 reps enough for range |
| Cache hit vs miss latency | **50** hit, **20** miss | min, median, p95, mean | Hit is cheap; miss is more variable |
| Concurrency sweep (per concurrency level) | **≥200 requests per level** (e.g. 200 at 1-conc, 200 at 4-conc, …) | per-request latency: min, median, p95, p99; throughput: requests/s | Need a distribution behind the percentiles |
| HTTP vs in-process delta | **50** | min, median, p95, mean | Isolates FastAPI+batcher overhead |
| Batcher window effect | **50** single-request, **50** burst-request | p50 for each condition | 5 ms window is small; need precise median |
| Model reload time | **5** | min, median, max | Same as cold reload |
| VRAM snapshot | 1 per state change | absolute value | Not a distribution |

All systems measurements record `nvidia-smi` concurrent-tenant GPU util at measurement time
to flag runs affected by other tenants.

### 6.3 Systems measurement list

- warm single-call latency by question-shape (1q/3q/10q/24q; choice/score/noul mixes)
- cold latency: after `/control/unload`, first request (reload path, ~0.7 s expected) and second (warm)
- end-to-end HTTP vs in-process delta (isolation of FastAPI+batcher overhead)
- input_tokens vs rendered sequence length (document where the 512-token ceiling bites; plot against `input_tokens`, not raw state length)
- latency vs option count (head-token scaling: 2 → 10 → 20 → 25+ options; record `input_tokens` at each step)
- concurrency sweep: 1 / 4 / 16 / 64 concurrent requests — coalescing behaviour, per-request latency, throughput
- cache: hit vs miss latency (repeat identical payload), cache-key sensitivity (1-char state change → miss)
- model reload time (unload → next-request-ready), VRAM before/after via `nvidia-smi` snapshot
- batcher window: does the 5 ms window add measurable latency at low concurrency? (single-request p50 with/without traffic)

Output: `systems.json` + a table in the report. This becomes the regression baseline for
service-side changes (ORT upgrade, sequence.py edits, config changes).

## 7. Reference comparison (reference.py)

### 7.1 Role: reference comparator, never adjudicator

The 27B model is a **capability/cost comparator**, not a ground-truth source. Both Laya and
the 27B are evaluated against the same human gold. If Laya disagrees with the 27B and Laya
matches the human gold, Laya wins that case. If both disagree with gold, both lose. The
comparison table shows Laya-vs-gold and 27B-vs-gold side by side; it does not show
Laya-vs-27B agreement as a success metric.

This is stated explicitly in the report header and in `benchmarks/README.md`.

- Same cases, same gold, run through the **main local 27B model** with forced structured output
  (JSON-schema-constrained answer per question type). Endpoint/model supplied via config
  (`--reference-endpoint`, `--reference-model`) — no ports or model names hardcoded; ask Brett
  which endpoint is canonical at build time.
- ≥3 runs per case, mean ± spread (stochastic — standing rule).
- Budget note: ~300 cases × 3 runs × 2 formulations ≈ 1800 calls; run reference **once per
  corpus version**, cache results under `results/reference-v1/`, not per harness run.
- Comparison table per family: `Laya acc / ms` vs `27B acc / ms` (both vs gold) — the "where
  does Laya belong in the stack" evidence.
- Optional rule/keyword baseline for email-triage families where a deterministic baseline is
  plausible (cheap to add; flagged per family as `baseline_applicable`).

## 8. Report (report.py)

`report.md` (human) + `metrics.json` (machine, for future diffs) + `provenance.json`:

### 8.1 Provenance (every run, including systems runs)

`provenance.json` records exactly what produced the results:

| Field | Source |
|---|---|
| `corpus_version` | `manifest.json` version string |
| `corpus_hash` | SHA-256 of `cases.jsonl` |
| `service_git_commit` | `git rev-parse HEAD` in the service repo |
| `service_version` | `/health` → `version` |
| `onnx_model_hash` | SHA-256 of `laya.onnx` (or `laya.onnx.data` if separable) — recorded once per model bundle, not per run |
| `onnx_config_hash` | SHA-256 of `onnx/laya_config.json` (temperature buckets, max_len, etc.) |
| `tokenizer_dir_hash` | SHA-256 of tokenizer directory contents (sorted file list + sizes) |
| `onnxruntime_version` | `onnxruntime.__version__` from the service venv |
| `provider_versions` | CUDA / cuDNN / driver versions (`nvidia-smi`, `ldconfig -p | grep cuda`) |
| `gpu_model` | `nvidia-smi --query-gpu=name --format=csv` |
| `gpu_index` | `CUDA_VISIBLE_DEVICES` value |
| `service_config_hash` | SHA-256 of `config/laya_config.json` |
| `python_version` | `sys.version` |
| `package_version` | `laya_server.__version__` |
| `run_date` | ISO 8601 UTC |
| `endpoint` | `LAYA_ENDPOINT` or default |

If a field cannot be determined, it is set to `"unknown"` with a note — the report is never
silent about missing provenance.

### 8.2 Report sections

1. Header: corpus version + hash, service version + git commit, provider (must show CUDA —
   refuse to record a CPU-fallback run as a baseline), date, reps, provenance summary
2. Per-family quality tables (tier-separated, dev/lock side by side) — the primary content
3. Option-count sensitivity curve; score-level-count curve; rendered-token-length curve
4. Consistency fragility table
5. Inversion-pair findings
6. Routing-head verdict (act/escalate vs confidence AUCs, keep/kill recommendation)
7. Systems table
8. Reference comparison table (Laya vs gold, 27B vs gold, side by side — never Laya-vs-27B
   as the success metric)
9. **Envelope summary**: green (trust), yellow (weak signal only), red (never decide) per
   family — **pass bars selected on dev split, evaluated on lock split**; the report states
   which cases informed each bar and shows the lock-split performance
10. Dev/lock divergence flag: any metric where dev and lock differ materially (configurable
   threshold, default 10 pp for accuracy-type metrics)

Regression mode: `report.py --diff results/<baseline>` → per-family deltas (accuracy ±pp,
calibration drift, latency change, routing-signal change, provenance diff). Any service/model
change re-runs the frozen v1 corpus and diffs against the frozen baseline. The provenance diff
answers "what changed?" before the metric diff answers "what regressed?".

## 9. Adoption-gate policy (written into benchmarks/README.md)

- General benchmark (v1 corpus) first — produces the envelope map.
- **Before any production adoption for a specific decision** (e.g. Email Manager "legal
  obligation", OCR review filtering): author a domain slice (20–50 cases) for *that exact
  decision*, run it, record pass/fail against a pre-agreed bar. No slice → no adoption.
- Every production surprise → capture, label, append to next corpus version, re-run.

## 10. Build phases

| Phase | Content | Rough size |
|---|---|---|
| **A** | Harness skeleton: schemas.py (with split + token fields), runner.py (with provenance capture), report plumbing; smoke corpus of ~30 hand-authored cases (dev+lock); end-to-end run + report against live service | small |
| **B** | Full v1 corpus authoring (~300 cases) per §4 — the bulk of the work; human review pass on gold labels; dev/lock split assignment (§4.1) | large, iterative |
| **C** | systems.py full sweep + baseline freeze | small |
| **D** | reference.py 27B comparison run (needs endpoint decision) | medium |
| **E** | metrics.py completion (all §5 math, dev/lock discipline), diff mode, adoption-gate README | medium |

Phases A, C, E are pure engineering. Phase B is authoring/judgement work — likely done in
agent+human review passes, family group by family group. Phase D depends on one decision (which
endpoint/model is the reference).

## 11. Open decisions (for Brett — noted, not blocking the spec)

- **D1 — corpus authoring split:** who authors/reviews gold labels? Proposal: agent drafts all
  families with provenance tags; Brett reviews `objective`/`subjective` tier golds in batches.
- **D2 — reference model/endpoint:** which local 27B endpoint is canonical for §7.
- **D3 — math deps:** add `scipy`/`numpy` as a `[bench]` extra, or hand-roll Spearman/Kendall/
  AUC/ECE (≈100 lines, zero deps). Lean-repo convention suggests hand-rolled; scipy suggests trust.
- **D4 — results in git?** Proposal: `results/` gitignored; only the *baseline* `metrics.json`
  per corpus version committed (small, diffable).
- **D5 — pass bars:** numeric thresholds for green/yellow/red per family are selected on the
  **dev split** after the first real run produces the distribution, then frozen as the v1 bar
  and evaluated against the **lock split**. The frozen bars are recorded in `manifest.json`
  with the date and the dev-split evidence that justified them.

## 12. Known constraints baked into the design

- `max_len: 512` tokens for the **rendered sequence** (state + questions + syntax);
  `head_max_len: 192` — the corpus has explicit probe cases at and past these limits
  (documented as expected-degradation zones, not bugs). Token accounting is on the rendered
  sequence, not raw state length (§3.2).
- `score` hard limits: 2–10 levels. `choice` design ceiling ~255 but practically < ~20
  (temperature bucket for 11+ options is 0.1006 — very sharp; the option-count sweep exists to
  find the real collapse point empirically).
- Parity invariant untouched: the harness never modifies sequence rendering; it only measures.
- Card 1 is shared — systems numbers record concurrent GPU tenants where visible, and the
  sample-count rules in §6.2 absorb noise.
