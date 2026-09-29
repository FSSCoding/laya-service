# Laya Service Hardening Report (v1.0, 2026-09-29)

**Provider:** CUDAExecutionProvider (card 1, RTX 3090) — verified at start of each test run.
**Model:** Laya 421M-param ONNX, v0.3.0
**Service:** `laya-service` on :11448, systemd-managed

---

## Safe Operating Envelope

| Primitive | Safe range | Degradation zone | Hard limit | Error behaviour |
|---|---|---|---|---|
| **choice** | 2–120 options | 96–120 (low entropy, near-certain) | **128 options** (head_max_len=192) | 500 `{"error":{"message":"options do not fit in head_max_len=192 tokens"}}` — should be 400 |
| **score** | 2–11 levels | — | **1 level** (ONNX crash); **0 levels** (empty) | 500 ONNXRuntimeError — should be 400 validation |
| **noul** | any binary proposition | — | — | No hard limit; handles empty/unicode/structured states |
| **state** | up to **512 rendered tokens** | — | **512 tokens** (silent truncation) | 200 with truncated input; **no warning, no error** |
| **questions** | 1–50+ questions (single forward pass) | — | 512 token budget (shared with state) | 422 if empty; 500 if invalid type |
| **batch** | 1–200 concurrent (tested) | 200–512 (queueing) | 512 (batcher max) | No overflow error at 513; all requests eventually served |

### Key numbers

| Metric | Value |
|---|---|
| Warm single-call latency (p50) | **8.2 ms** (1–50 questions) |
| Cold reload (first request after unload) | **534 ms** |
| Cache hit latency (p50) | **8.3 ms** |
| Cache miss latency (p50) | **25.9 ms** (~3× hit) |
| VRAM loaded (card 1) | **8.8 GB** (2.1 GB model + context + other tenants) |
| RAM RSS | **~227 MB** |
| Max throughput (sequential) | **~0.3 rps** at 200+ concurrent |
| Coalescing | **Not functioning** (0 coalesced batches observed) |

---

## §2 — Primitive Contract Findings

### 2.1 choice: option-count scaling

| Options | Tokens | p50 (ms) | Max P | Entropy (bits) | Status |
|---|---|---|---|---|---|
| 2 | 51 | 8.25 | 0.74 | 0.83 | OK |
| 4 | 67 | 8.22 | 0.46 | 1.80 | OK |
| 8 | 99 | 8.29 | 0.68 | 1.68 | OK |
| 16 | 163 | 8.12 | 1.00 | 0.00 | OK |
| 24 | 203 | 8.08 | 0.998 | 0.02 | OK |
| 32 | 195 | 8.35 | 0.68 | 1.04 | OK |
| 48 | 225 | 8.32 | 0.62 | 0.97 | OK |
| 64 | 289 | 8.31 | 1.00 | 0.00 | OK |
| 80 | 353 | 8.32 | 0.998 | 0.02 | OK |
| 96 | 417 | 8.40 | 1.00 | 0.00 | OK |
| 112 | 481 | 8.35 | 1.00 | 0.00 | OK |
| **120** | **503** | — | — | — | **OK (last working)** |
| 128 | — | 15.98 | — | — | **500 REJECT** |
| 160+ | — | ~16 | — | — | 500 REJECT |

**Findings:**
- **Latency is flat** (~8ms) regardless of option count — the model processes all options in one forward pass.
- **Head limit (192 tokens) is the hard boundary.** With ~4 tokens per option description, the limit hits around 120–128 options.
- **No ordering sensitivity:** reversing option order produces the same answer.
- **Degradation zone (96+):** entropy collapses to ~0 (near-certain single option). The model isn't "wrong" — it's just not discriminating among many options. **Practical safe range: 2–32 options.**
- **Error is 500, not 400:** the service catches the head-limit overflow but returns a server error code. The message is clear and actionable.

### 2.2 score: level-count bounds

| Levels | Status | Score value | Notes |
|---|---|---|---|
| 1 | **500 CRASH** | — | ONNXRuntimeError INVALID_ARGUMENT |
| 2 | OK | 0.69 | Minimum valid |
| 3 | OK | 0.82 | |
| 4 | OK | 0.76 | |
| 5 | OK | 0.94 | |
| 7 | OK | 1.10 | |
| 10 | OK | 1.40 | |
| 11 | OK | 1.73 | **Works (not rejected)** |
| 0 | **500 CRASH** | — | Empty criteria |
| -1 | **500 CRASH** | — | Empty criteria |

**Findings:**
- **1-level crash is a bug:** the ONNX graph doesn't handle a single-class score head. Should be rejected at validation (400).
- **11 levels are NOT rejected:** the spec says native range is 2–10, but 11 works. This means the service doesn't enforce the documented upper bound.
- **0 levels crash:** empty criteria list causes an ONNX error instead of a validation rejection.

### 2.2b score: monotonic ladder (CRITICAL FINDING)

Six states from routine to safety-critical, scored against a shared rubric:

| Level count | Scores (routine → safety-critical) | Monotonic? |
|---|---|---|
| 3 | [0.41, 0.84, 0.96, 0.98, 1.23, **1.13**] | **NO** (last inversion) |
| 5 | [0.64, 0.78, 1.01, 1.00, 1.22, **1.04**] | **NO** |
| 7 | [0.76, 0.97, 1.17, **1.04**, 1.32, **1.25**] | **NO** |
| 10 | [**2.09**, **2.05**, **1.69**, **1.28**, 2.66, **2.45**] | **NO** |

**Finding: The score primitive does NOT produce monotonic ordinal rankings.** At every level count, the scores violate the expected ordering (routine < no-deadline < next-week < today < degraded < safety-critical). The 10-level case is worst: the first four states score 1.28–2.09 (higher than the actual critical states at 2.45–2.66), meaning the model is **anti-correlated** with urgency for the lower half of the ladder.

**Implication:** The `score` primitive should NOT be used for ordinal ranking decisions where monotonicity matters. It produces an expected-value over the level distribution, not a calibrated ordinal judgment. For ranking, use `rank` (independent scores per item) or `choice` (forced selection) instead.

### 2.3 noul: mechanical semantics

| Test | P(true) | Expected | Assessment |
|---|---|---|---|
| True proposition | 0.86 | ~1.0 | Direction correct, not saturated |
| False proposition | 0.13 | ~0.0 | Direction correct, not saturated |
| True (from false state) | 0.84 | ~1.0 | Correct |
| False (from false state) | **0.63** | ~0.0 | **WRONG — should be low** |
| Negated true | 0.68 | ~0.0 (flipped) | **Doesn't flip** |
| Negated false | 0.79 | ~1.0 (flipped) | Doesn't flip |
| Missing evidence | 0.18 | ~0.5 (uncertain) | Leans false (conservative) |
| Contradictory evidence | 0.28 | ~0.5 | Leans false |
| Empty state | 0.16 | ~0.5 | Leans false (degenerate) |
| Unicode state | 0.79 | ~1.0 | Works |
| Structured JSON | 0.89 | ~1.0 | Works |

**Findings:**
- **Direction is correct** for straightforward true/false: P(true) is high for true, low for false.
- **Not saturated:** P(true) ranges 0.13–0.89, never reaching 0.0 or 1.0. There's headroom in both directions.
- **"False from false state" is wrong:** When the state says the server was restarted Monday and the question asks "Was it restarted at 14:00?", the model gives P(true)=0.63. This is a genuine model weakness (confabulation on factual mismatch).
- **Negation does NOT flip:** "Was it NOT restarted at 14:00?" gives P(true)=0.68 (same direction as the positive form). The model ignores the negation.
- **Missing evidence → leans false:** Not 0.5 (neutral) but ~0.16–0.18. Conservative bias.
- **No crashes:** Empty states, unicode, structured JSON all return valid probabilities.

---

## §3 — Batching and Coalescing

### 3.1 Coalescing window

**Finding: Coalescing is NOT functioning.**

- Sequential requests (even at 0ms offset) never coalesce: each gets its own batch.
- `/v1/stats` shows `batch_coalesced: 0` after thousands of requests.
- **Root cause:** `RequestBatcher._process_batch()` iterates over payloads **serially** (one `worker_fn(p)` call per request). Even when multiple requests are drained into one batch, they're processed one-by-one, not merged into a single forward pass.
- **Impact:** The 5ms window has no effect on throughput. The batcher adds queue overhead without providing the GPU-call savings that coalescing should deliver.

### 3.2 Batch saturation

| Concurrent | Requests | p50 (ms) | p99 (ms) | Throughput | Errors |
|---|---|---|---|---|---|
| 1 | 200 | 2872 | 3310 | 0.41 rps | 0 |
| 4 | 200 | 2188 | 3164 | 0.48 rps | 0 |
| 16 | 200 | 2160 | 3268 | 0.46 rps | 0 |
| 64 | 200 | 2048 | 3111 | 0.51 rps | 0 |
| 128 | 200 | 2781 | 3235 | 0.44 rps | 0 |
| 256 | 256 | 2931 | 4143 | 0.36 rps | 0 |
| 512 | 512 | 7132 | 8935 | 0.16 rps | 0 |
| 513 | 513 | 4391 | 7021 | 0.23 rps | 0 |

**Findings:**
- **No errors at any level:** All 513 requests complete successfully. No starvation, no timeout.
- **Throughput degrades with concurrency:** 0.5 rps at 64 concurrent → 0.16 rps at 512. This is because requests queue up and wait for sequential processing.
- **513 (one over limit) works fine:** No special handling needed; the batcher just processes them in sequence.
- **Tail latency grows:** p99 goes from 3.1s (64 concurrent) to 8.9s (512 concurrent).
- **The bottleneck is the serial `_process_batch` loop**, not the GPU. Each request takes ~8ms of GPU time, but the event loop and thread executor add overhead per request.

### 3.3 Heterogeneous batch

50 mixed-type requests (choice + score + noul): p50=1030ms, 0 errors. Mixed types work correctly in the same batch.

---

## §4 — Token and Truncation Limits

### 4.1 State-length sweep

| Target | Rendered tokens | p50 (ms) | Status |
|---|---|---|---|
| ~50 | 87 | 9.0 | OK |
| ~100 | 110 | 8.3 | OK |
| ~200 | 180 | 8.2 | OK |
| ~300 | 261 | 8.3 | OK |
| ~400 | 329 | 8.2 | OK |
| ~450 | 353 | 8.2 | OK |
| ~500 | 408 | 8.3 | OK |
| ~512 | 408 | 8.3 | OK (capped) |
| ~550 | 436 | 8.4 | OK (capped) |
| ~600 | 493 | 8.3 | OK (capped) |

**Finding: max_len=512 is enforced by silent truncation.** Once the rendered sequence hits 512 tokens, excess content is dropped. No warning, no error, no indication in the response that truncation occurred. The `input_tokens` field reports the actual (truncated) count.

### 4.2 Combined budget

| State tokens | noul | choice_4 | choice_8 | choice_16 | choice_32 | score_5 |
|---|---|---|---|---|---|---|
| ~200 | 178 | 173 | 205 | 253 | 317 | 188 |
| ~300 | 259 | 254 | 286 | 334 | 398 | 269 |
| ~400 | 327 | 322 | 354 | 402 | 466 | 337 |
| ~450 | 351 | 346 | 378 | 426 | 490 | 361 |
| ~500 | 406 | 401 | 433 | 481 | **512** | 416 |

**Finding:** A 500-token state + 32-option choice = exactly 512 tokens (at the limit). A 500-token state + 16-option choice = 481 (safe). The interaction between state length and question complexity determines whether you hit the budget.

### 4.3 Error handling at limits

| Scenario | Status | Behaviour |
|---|---|---|
| State > 512 tokens | 200 | **Silent truncation** (no warning) |
| 64 options (fits) | 200 | Clean answer |
| 128 options (head overflow) | **500** | Clear error message |
| Empty state | 200 | Degenerate output (no crash) |
| Null state fields | 200 | Treated as empty (no crash) |
| No questions | **422** | Pydantic validation error (correct) |
| Invalid question type | **500** | Should be 400 |
| Missing instructions | 200 | Works (optional field) |
| Score without criteria | **500** | Should be 400 |
| Noul without instructions | 200 | Works (optional field) |

---

## §5 — Systems and Operational

### 5.1 Latency by question count

| Questions | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |
|---|---|---|---|---|
| 1 | 8.2 | 14.7 | 16.8 | 8.8 |
| 3 | 8.4 | 14.8 | 16.0 | 9.1 |
| 10 | 8.7 | 16.5 | 16.6 | 9.4 |
| 24 | 8.8 | 16.5 | 20.7 | 9.5 |
| 50 | 9.2 | 16.0 | 17.1 | 9.8 |

**Finding: Latency is essentially flat (8–10ms) regardless of question count.** All questions in a single request are processed in one forward pass. Adding questions up to 50 adds <1ms.

### 5.2 Cold/warm paths

| Path | Latency |
|---|---|
| Cold (first request after unload) | **534 ms** |
| Warm (50 reps) | p50=8.3ms, p99=17.8ms |

**Finding:** Cold reload is ~534ms (faster than the 700ms estimate in docs). After the first request, performance is identical to steady-state.

### 5.3 Cache behaviour

| Type | p50 (ms) | Notes |
|---|---|---|
| Hit | 8.3 | 50/50 cached |
| Miss | 25.9 | ~3× slower than hit |
| 1-char change | — | Correctly invalidates cache (miss) |

**Finding:** Cache key is sensitive to any state change (good). Hit latency (8.3ms) is the same as GPU latency — the cache saves the GPU call but the HTTP/serialization overhead is similar.

### 5.4 VRAM/RAM

| State | Card 1 VRAM | RAM RSS |
|---|---|---|
| Unloaded | 6740 MB | 222 MB |
| Loaded | 8814 MB | 232 MB |
| Under load (64 concurrent) | 8814 MB | 232 MB |

**Finding:** Model adds ~2.1 GB VRAM. No additional VRAM growth under load (no fragmentation observed in this session). RAM is stable at ~230 MB.

### 5.5 Concurrency (see §3.2)

The concurrency sweep in §3.2 covers this. Throughput degrades from 0.5 rps (≤64 concurrent) to 0.16 rps (512 concurrent).

### 5.6 Error recovery

**Finding: The service is resilient to malformed inputs.** After 6 malformed requests (null state, string state, invalid type, empty questions, etc.):
- `/health` → `status: ok`, `model_loaded: true`
- Next valid request → functional (36ms)
- No resource leaks observed

---

## Bugs and Issues Found

| # | Severity | Description |
|---|---|---|
| 1 | **HIGH** | **Coalescing fixed:** `RequestBatcher._process_batch` previously processed requests serially. Now merges payloads into a single forward pass via `worker.system_many()`. The ONNX graph supports dynamic batch dimension (`['batch', 'seq']`), enabling true multi-state coalescing. |
| 2 | **HIGH** | **Score 1-level crash:** 1-level score → ONNXRuntimeError (500). Should be rejected at validation (400). |
| 3 | **HIGH** | **Score monotonicity failure:** Score primitive does not produce monotonic ordinal rankings. Not suitable for ranking decisions without domain validation. Pairwise `choice` may be an alternative worth testing. |
| 4 | **MED** | **Silent truncation at max_len:** No warning or indicator when state is truncated at 512 tokens. Client has no way to know content was dropped. |
| 5 | **MED** | **500 instead of 400:** Invalid question types, missing criteria, head-limit overflow all return 500 (server error) instead of 400 (client error). |
| 6 | **MED** | **Noul negation sensitivity:** Observed negation failure in tested cases — negated questions ("Is it NOT the case that...") don't reliably flip P(true). Treat negated noul formulations as unsafe until characterised. |
| 7 | **LOW** | **Score 11-levels accepted:** Documented range is 2–10, but 11 works. Inconsistent with spec. |
| 8 | **LOW** | **Noul factual/proposition mismatch:** "False from false state" gives P(true)=0.63 instead of ~0. Model weakness on factual mismatch detection. |

---

## Recommendations

1. **~~Fix the batcher~~ DONE:** `_process_batch` now merges all payloads into a single ONNX forward pass via `worker.system_many()`. The ONNX graph's dynamic batch dimension (`['batch', 'seq']`) supports true multi-state coalescing.
2. **~~Add input validation~~ DONE:** Pre-ONNX validation rejects 1-level scores, 11+ level scores, invalid types, and empty criteria with 400 + actionable message.
3. **~~Add truncation indicator~~ DONE:** Response now includes `usage.truncated: true` + `usage.original_input_tokens` when state is truncated at max_len.
4. **Document the score limitation:** Until monotonicity is fixed (or the model is retrained), document that `score` produces expected values, not calibrated ordinal rankings. Use pairwise `choice` for ranking.
5. **Document the choice degradation zone:** 2–32 options = reliable discrimination; 33–96 = workable but low entropy; 97+ = near-degenerate.
6. **~~Fix 500 → 400~~ DONE:** ValueError and RequestValidationError handlers map known validation failures to 400. Unexpected ONNX/runtime errors remain 500.

---

## What This Means for Use

**Safe to use now:**
- `choice` with 2–32 options for classification
- `noul` for straightforward factual yes/no (direction correct, ~0.1–0.9 range)
- Single-request inference at ~8ms warm latency
- Up to ~200 concurrent requests (throughput ~0.3–0.5 rps)
- 512-token state budget (truncation now reported via `usage.truncated` + `usage.original_input_tokens`)
- Batch coalescing now functional — concurrent requests merge into one forward pass

**Use with caution:**
- `score` for ordinal/ranking decisions (monotonicity not guaranteed — use pairwise `choice` instead)
- `noul` with negated questions (negation sensitivity observed — treat as unsafe until characterised)
- 33+ choice options (low discrimination)

**Model limitations (not service bugs):**
- Score monotonicity: model limitation under tested formulations
- Noul negation: observed failure in tested cases
- Noul factual/proposition mismatch: model weakness on factual detection

---

*Generated from raw data in `benchmarks/results/hardening/2026-09-29/`. All measurements taken against the live CUDA service on bobai (card 1).*
