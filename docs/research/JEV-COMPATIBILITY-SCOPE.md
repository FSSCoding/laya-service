# Jev `system_one` Compatibility (Research)

**Status:** Research / scoping only. No adapter code, no API changes from this document.
**Date:** 2026-10-06.
**Authoritative reference:** the upstream JS type definitions
(`@receptron/laya`, file `src/types.ts`), which state: *"They follow TypeSafe Jev's `system_one`
API, which Laya reproduces."* That file is the source of truth for the *base* shape.

> "Jev" and "TypeSafe" are trademarks of their respective owners. Nothing in this service or its
> docs implies partnership, endorsement, or affiliation. This doc describes wire-shape compatibility
> only.

## Purpose

Laya's `/system_one` endpoint is deliberately shaped to match TypeSafe Jev's `system_one` API so that
clients already speaking Jev can point at Laya with minimal change. This document maps the exact
field-by-field correspondence, records where Laya **deliberately diverges** (extensions), and sizes
the adapter layer a Jev client would need. It does **not** implement any of it.

## Base `system_one` contract (from `src/types.ts`)

**Request** — a `state` plus a `questions` map. Each question is one of three types:

| Type | `instructions` | `criteria` |
|---|---|---|
| `choice` | `string \| object` | `Record<string, string\|null>` (option → description) **or** `string[]` (plain option list) |
| `score` | `string \| object` | `string[]` (ordered levels, index 0 = lowest) |
| `noul` | `string \| object` | optional `{ true?: string; false?: string }` |

**Response** — `{ model, answers, usage }`:

| Answer type | Fields |
|---|---|
| `choice` | `type`, `choice: string`, `probabilities: Record<string,number>`, `confidence: number`, `rl_agent` |
| `score` | `type`, `score: number`, `legend: Record<string,string>`, `probabilities`, `confidence`, `rl_agent` |
| `noul` | `type`, `noul: number` (P(true)), `rl_agent` |

| Envelope | `model: string`, `answers: {[qid]: Answer}`, `usage: { input_tokens, output_tokens }` |

`rl_agent` in the **base** Jev shape is `{ act_probability: number }`.

## Field-by-field match (Laya → base Jev)

Laya's Python worker (`worker.py`) emits exactly the base fields. Verified against the live code:

| Base field | Laya emits? | Notes |
|---|---|---|
| `model` | ✅ `"laya"` | Jev would use its own model id; harmless string difference. |
| `answers.<qid>.type` | ✅ `choice` / `score` / `noul` | Identical discriminator values. |
| choice `.choice` | ✅ argmax option name | Matches. |
| choice `.probabilities` | ✅ `Record<option, p>` rounded 4dp | Matches. |
| choice `.confidence` | ✅ `1 − normalized_entropy` | Same definition as the Jev comment. |
| score `.score` | ✅ expected level `Σ i·p[i]` | Matches "expected level (0..levels-1)". |
| score `.legend` | ✅ `Record<str(i), criterion>` | Matches. |
| score `.probabilities` | ✅ `Record<str(i), p>` | Matches (keys are stringified indices). |
| noul `.noul` | ✅ `p[1]` (P(true)) | Matches. |
| `usage.input_tokens` | ✅ actual (post-truncation) rendered tokens | Matches. |
| `usage.output_tokens` | ✅ always `0` | See divergence D3. |
| `rl_agent.act_probability` | ✅ present | Matches the base field. |

**Bottom line:** for a Jev client that only reads the base fields, Laya is a drop-in for the
choice/score/noul semantics. The request side also accepts the same question shapes (including
`criteria` as either a map or a plain list, and `instructions` as string-or-object).

## Deliberate divergences (Laya extensions beyond base Jev)

These are **additive** — a strict base-Jev client that ignores them sees nothing different. A
forward-looking client can read them.

| # | Field | Base Jev | Laya | Effect on a strict base client |
|---|---|---|---|---|
| **D1** | `rl_agent.escalate_probability` | absent | present (0.0–1.0) | Ignored (extra key). Safe. |
| **D2** | `rl_agent.experimental` | absent | present, always `true` | Ignored (extra key). Safe. |
| **D3** | `usage.output_tokens` | (semantic: generated tokens) | always `0` | A client asserting `output_tokens > 0` on a text model would be surprised; for a decision model (no text generation) `0` is correct. Document this. |
| **D4** | `usage.truncated` / `usage.original_input_tokens` | absent | present **only when** the state exceeded `max_len` and was truncated | Ignored when not truncated. A client should treat their absence as "no truncation". |
| **D5** | HTTP transport | Jev spec is transport-agnostic (RPC-shaped) | FastAPI JSON over HTTP on a fixed port | A Jev RPC client needs a thin HTTP binding (see adapter below). |
| **D6** | `model` id | vendor-specific string | `"laya"` | Purely informational; clients must not hard-match it. |

None of D1–D6 breaks a client that only consumes the base fields. The extensions are strictly
additive and are flagged so a client author can decide whether to surface `escalate_probability`
(an experimental routing signal) or the truncation indicators.

## What cannot be verified without the official Jev spec

The upstream file says Laya *reproduces* the Jev `system_one` API, and it fully specifies the TS
types above. But the **TypeSafe Jev runtime spec** (e.g. `docs.typesafe.ai`) is not available in this
repo, so the following remain **unverified** and must be confirmed before claiming full wire
compatibility:

1. **Exact request envelope.** `src/types.ts` defines the *question* shapes but not the outer
   request wrapper (e.g. whether Jev wraps `{state, questions}` in an envelope, names the port,
   or uses a different top-level key). Laya uses `{state, questions}` directly.
2. **`instructions: object` semantics.** Both allow `string | object`, but the *object* form's
   schema (what keys, what nesting) is defined by Jev, not by this repo. Laya currently treats
   `instructions` as free-form and passes it through; if Jev expects a specific object shape,
   a client sending that shape will work, but Laya does not validate it.
3. **Error contract.** Jev's error format (shape, status codes) is not specified here. Laya
   returns FastAPI/Pydantic errors (422 for validation, 500 for unexpected, 400 for known
   validation failures). A strict Jev client expecting a different error envelope would need
   mapping.
4. **Transport/auth.** Whether Jev assumes a particular RPC framework, transport security, or
   authentication is unknown. Laya is bare HTTP with no auth (loopback-oriented by design).

These four items are the honest gap: the *data* shape matches the published `types.ts`, but the
*protocol* (envelope, transport, errors, auth) has not been cross-checked against the official Jev
runtime because that spec is out of scope for this repo.

## Minimal adapter layer a Jev client would need

To point an existing Jev `system_one` client at Laya without changing the client:

1. **Transport shim (D5).** Replace the Jev RPC/SDK call with a plain `POST <laya>/system_one`
   sending the same `{state, questions}` JSON and reading the same `{model, answers, usage}`
   JSON. ~One function.
2. **Model-id tolerance (D6).** Do not assert on the `model` string.
3. **Output-tokens expectation (D3).** Remove any assertion that `output_tokens > 0`; expect `0`.
4. *(Optional)* Read the extensions (D1, D2, D4) if the feature is wanted; otherwise ignore.
5. *(Only if needed after the unverified-gap check)* Map error shapes and confirm the request
   envelope against the official Jev spec.

Estimated size: a single module (≈50–150 lines) plus tests. This is **not** built as part of the
public release; it is scoped here so the effort is known.

## Recommendation

- **Data compatibility: high.** Laya intentionally reproduces the base Jev `system_one` data shape
  and adds only additive, safe extensions.
- **Protocol compatibility: unverified** pending the official Jev runtime spec (items 1–4 above).
- **Action:** publish the doc, keep the extensions additive, and **do not implement the adapter**
  until a real Jev-client integration request exists. At that point, fill in the four unverified
  items against the official spec, then build the ~1-module transport shim.
- **Trademark guard:** all references to "Jev"/"TypeSafe" stay descriptive ("compatible with the
  Jev `system_one` shape") and never imply partnership or endorsement.
