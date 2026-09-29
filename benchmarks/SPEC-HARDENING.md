# Laya Service Hardening & Operating Envelope (v1.0, 2026-09-29)

**Status:** current mission. Supersedes the semantic-corpus campaign (see `FUTURE-METHODS.md`).
**Goal:** characterise and harden the **actual native service and its three primitives** — `choice`, `score`, `noul` — plus heterogeneous batching and request coalescing. Establish the mechanical operating envelope: input limits, scaling behaviour, performance curves, failure modes, validation behaviour, concurrency characteristics.

**Exit criteria:** The native primitives have known input limits, scaling behaviour, performance curves, failure modes, validation behaviour and concurrency characteristics; a small semantic smoke baseline exists; future evaluation methodologies are documented and deferred.

---

## 1. What this is NOT

This is **not** a large-scale semantic evaluation campaign. The 300-case corpus, inversion pairs, consistency perturbation suite, OOD taxonomy, and domain-specific adoption slices are preserved in `FUTURE-METHODS.md` as deferred research. They are not the current mission.

The current mission is: **punish the three primitives and the service until we understand their operating limits, failure behaviour, latency, batching, truncation, validation and error handling. Then use the thing.**

---

## 2. Native primitive contract testing

The model surface is only `choice`, `score`, and `noul`. Everything else is composition. These three deserve almost all the attention.

### 2.1 choice — option-count scaling

The theoretical design ceiling is much higher than the practical one. The real limitation comes from the **head token budget** (`head_max_len=192`). As option count grows, each option gets less representation budget.

**Sweep:** 2, 4, 8, 16, 24, 32, 48, 64 options — continue until the service rejects the request or behaviour clearly collapses.

**Record per step:**
- `input_tokens` (rendered sequence)
- head tokens if derivable
- latency (warm, 50 reps → min/median/p95/p99)
- probability distribution (entropy, max P, spread)
- confidence
- whether options were truncated
- whether option ordering changes the answer (run same case with options reversed)
- whether longer option descriptions change behaviour
- whether the request is rejected cleanly when the representation can no longer fit

**Output:** a scaling curve answering "how many choice options can I safely give this thing before I should use hierarchy or another strategy?"

### 2.2 score — level-count bounds

Native range is 2–10 levels.

**Boundary tests:**
- 1 level → should reject cleanly (what error?)
- 2 levels → valid (minimum)
- 3, 4, 5, 7, 10 → valid
- 11 levels → should reject cleanly (what error?)
- 0 levels, negative, non-integer → validation behaviour

**Quality sweep (controlled ladder):** For a single semantic problem (e.g. urgency), construct an ordered sequence of states where only the relevant property changes:
- routine FYI
- request with no deadline
- deadline next week
- deadline today
- service degraded
- production unavailable / safety-critical

Run the same rubric across the ladder at 3, 5, 7, and 10 levels. Measure whether `score(A) < score(B) < score(C) ...` — monotonicity. This tells us about actual ordinal resolution much more cleanly than MAE over unrelated examples.

### 2.3 noul — mechanical semantics

Not semantic intelligence testing. Mechanical contract testing:
- true proposition → P(true) near 1
- false proposition → P(true) near 0
- missing evidence → what does it return?
- contradictory evidence → what does it return?
- negated wording ("Is it NOT the case that...") → does it flip correctly?
- very long state (near 512 tokens) → truncation behaviour
- empty state → validation or degenerate output?
- structured JSON state → does it parse correctly?
- Unicode, weird punctuation → robustness
- huge input (past max_len) → clean rejection?
- invalid request shapes → actionable error messages?

---

## 3. Batching and coalescing

The service batcher uses a **5 ms coalescing window** and can merge up to **512 queued requests** into a forward pass. This is the core operational characteristic that needs hammering.

### 3.1 Coalescing window behaviour

**Questions to answer empirically:**
- If request A arrives at t=0 and request B arrives at t=4 ms, do they enter the same GPU pass?
- What about one at t=6 ms?
- When does the first batch dispatch?
- While that batch is executing, what happens to newly arriving requests? Do they queue for the next coalescing window?
- Does a huge first batch increase tail latency for the second?

**Method:** instrument with precise timestamps. Send pairs/triples of requests at controlled offsets (0 ms, 2 ms, 4 ms, 6 ms, 8 ms, 10 ms). Record whether they were coalesced (same batch) or sequential (different batches). Measure per-request latency in each scenario.

### 3.2 Batch saturation

**Sweep:** 1, 4, 16, 64, 128, 256, 512 concurrent requests. Then 513 (one over the limit).

**Record per level:**
- throughput (requests/second)
- per-request latency distribution (min/median/p95/p99)
- whether the 512 limit is hit cleanly or causes errors
- what happens at 513 — does it split into 512+1? Is there starvation? Does latency jump?
- does the process stay responsive?
- queueing behaviour between adjacent batches

### 3.3 Heterogeneous batch

Mixed-type questions (choice + score + noul) in one batch. Does the batcher handle mixed types correctly? Does one slow question type block others? Record per-type latency within a heterogeneous batch.

---

## 4. Token and truncation limits

`max_len=512` for the rendered sequence (state + questions + syntax). `head_max_len=192`.

### 4.1 State-length sweep

Controlled states at increasing token counts: 50, 100, 200, 300, 400, 450, 500, 512, 550, 600 tokens.

**Record per step:**
- `input_tokens` (rendered)
- whether truncation occurred
- which portion was truncated (state? questions? both?)
- latency
- whether the answer changes qualitatively at the truncation boundary
- whether the service returns a clear warning/error when truncation occurs

### 4.2 Combined budget

A 450-token state with a 20-option choice question may already exceed the budget. A 450-token state with a noul may fit. Test the **interaction** between state length and question complexity.

### 4.3 Error handling at limits

What happens when:
- state alone exceeds max_len?
- state + questions exceed max_len?
- head alone exceeds head_max_len?
- the request is technically valid but empirically unsafe (e.g. 64 options that fit but produce pathological degradation)?

Does the service reject cleanly with an actionable error? Does it silently truncate? Does it produce garbage?

---

## 5. Systems and operational behaviour

### 5.1 Latency distributions

Warm single-call latency by question-shape (1q, 3q, 10q, 24q, 50q). 50 reps → min/median/p95/p99/mean.

### 5.2 Cold/warm paths

- After `/control/unload`, first request (cold reload, ~0.7 s expected) — 5 reps → min/median/max
- Second request (warm) — 50 reps
- After idle auto-unload (300 s) — same cold/warm pattern

### 5.3 Cache behaviour

- Hit vs miss latency (50 hit, 20 miss)
- Cache-key sensitivity: 1-char state change → miss?
- Cache eviction under memory pressure?
- Cache persistence across restarts (SQLite DB)?

### 5.4 VRAM/RAM behaviour

- VRAM loaded vs unloaded (nvidia-smi snapshot)
- RAM RSS under load
- Does VRAM fragment over time? (weekly restart timer exists for this reason)
- Recovery after OOM or CUDA error?

### 5.5 Concurrency and throughput

Sweep: 1, 4, 16, 64, 128, 256, 512 concurrent. ≥200 requests per level. Record throughput, latency distribution, tail latency.

### 5.6 Error recovery

- What happens after a malformed request? Does the service stay healthy?
- What happens after a CUDA error? Does it fall back to CPU? Does it recover?
- What happens after `/control/unload` mid-batch?
- Does the service leak resources under sustained error conditions?

---

## 6. Small semantic probe set

Retain **30–60 carefully chosen semantic probes** — enough to establish rough capability shape and detect glaring regressions. Not 300 hand-authored cases.

These are the existing smoke corpus (33 cases) plus targeted additions for the controlled ladders (§2.2) and boundary probes (§4). The smoke corpus is already authored and frozen; it serves as the regression baseline.

**Do not spend hours manufacturing hundreds of synthetic labelled cases.** The eventual real Email Manager traffic will be vastly more valuable than fifty invented "billing vs support" examples.

---

## 7. Validation and error-handling audit

For every limit discovered, verify that the client/CLI/API either:
- prevents invalid requests (validation at the boundary), or
- returns clear actionable errors (not silent truncation or garbage output).

If tooling permits a technically valid but empirically unsafe usage — for example an option count that fits but produces pathological degradation — **document the measured safe operating region**.

Output: a **safe operating envelope** table:

| Primitive | Safe range | Degradation zone | Hard limit | Error behaviour |
|---|---|---|---|---|
| choice | 2–? options | ?–? options | ? options | ? |
| score | 2–10 levels | — | 11 levels | ? |
| noul | any binary | — | — | ? |
| state | ?–? tokens | ?–512 tokens | 512 tokens | ? |
| batch | ?–512 requests | — | 512 requests | ? |

---

## 8. Deliverables

1. **`benchmarks/hardening/`** — test scripts for each section above (primitive contracts, coalescing, token limits, systems, error handling).
2. **`benchmarks/results/hardening-<date>/`** — raw outputs, scaling curves, latency distributions.
3. **`benchmarks/HARDENING-REPORT.md`** — the safe operating envelope table, scaling curves, failure modes, error-handling audit.
4. **Updated `ROADMAP.md`** — Phase 3.5 = Service Hardening & Operating Envelope, with exit criteria.
5. **Updated `benchmarks/README.md`** — reflects the hardening mission, not the semantic corpus campaign.
6. **`FUTURE-METHODS.md`** — preserved, deferred, clearly marked.

---

## 9. What happens after

Once the hardening pass is complete and the safe operating envelope is documented:

**Stop. Use it.**

Run actual inferencing. See a request. See the raw probabilities. Try stupid inputs. Try something meaningful. Put 30 questions in one batch. Send 200 concurrent requests. Make the thing earn confidence through interaction rather than through another week of agent-authored paperwork.

When a real use case emerges (Email Manager actionability, OCR filtering, telemetry enrichment), **then** author a 20–30 case domain slice from actual representative data and evaluate that specific decision. That's much stronger than trying to simulate every future domain before using the thing once.

The adoption-gate policy remains: **no operational Laya decision is adopted solely from general benchmark performance. Each production decision requires a representative domain-specific evaluation slice with pre-agreed acceptance criteria.**

---

## 10. Preserved future methods (deferred)

The following are documented in `FUTURE-METHODS.md` and preserved for later:

- Query inversion (choice vs noul reformulation)
- Hierarchical classification
- Routing/escalation experiments
- Perturbation consistency testing
- Large-scale semantic corpus expansion (~300 cases)
- Domain-specific adoption slice methodology
- 27B reference comparison

These are **research backlog / future evaluation methods**, not current build requirements. They can be resumed quickly when a real use case requires them.
