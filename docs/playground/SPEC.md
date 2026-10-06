# Laya Playground — learning instrument specification and build plan

**Status:** Proposed build; not implemented. 2026-09-29.
**Audience:** Brett, experimenting manually with real Laya responses.
**Purpose:** Type a state and a question, press **Run**, see what Laya actually returns, then change the inputs and try to break it. This is not a production client or a new model interface.

> **Important: separate HTTP requests are NOT coalesced into one inference pass today.** The 5 ms request-batcher window does not provide GPU-call savings: `RequestBatcher._process_batch()` processes queued request payloads serially. Do not present multiple browser submissions as a single GPU batch or imply that raising HTTP concurrency improves throughput. **Several questions inside ONE `/system_one` request** *are* processed together in one model forward pass. The playground's **Batch** tab means this latter, working feature only. See `benchmarks/HARDENING-REPORT.md` §3 and `src/laya_server/batcher.py`.

## 1. Boundaries and launch

- A small **local-only Python HTTP server** serves ordinary HTML/CSS/vanilla JavaScript and proxies to the existing Laya service. No FastAPI instance, build tool, npm, SPA framework, model loading, database, or changes to `laya_server`.
- Bind the playground to `127.0.0.1` only, on a configurable port **chosen before implementation**; do not assume an unused port. The upstream is fixed to `http://127.0.0.1:11448` unless the user explicitly requests remote operation. Browser requests go to the playground's own origin (`/api/query`, `/api/health`), so no direct cross-origin request to :11448 and no Laya CORS change.
- Package the playground separately from the service application code, with a small server script and static files; decide the exact source directory when implementing. The only new runtime dependency, if used, is `httpx`, already in `pyproject.toml`. Python standard-library HTTP serving and request parsing are sufficient.
- Run from the repo with its `.venv/bin/python` and print the loopback URL. Do not automatically launch a browser or manage the systemd unit. Existing human CLI: `.venv/bin/laya`; existing agent MCP tools: `laya_system_one`, `laya_health`, `laya_stats`, `laya_unload` (not needed to use this page).
- Avoid adding deployment, authentication, accounts, persistence, telemetry, OpenAI compatibility, or service controls. Local-only is a safety boundary, **not** a claim that there is authentication. Do not expose the playground on `0.0.0.0` or proxy arbitrary user-supplied URLs.

## 2. Learning flow and layout

A polished, responsive, single-page **experiment workbench** with a clear left-to-right flow on desktop (prompt → Run → visual explanation), stacked on small screens. The primary interaction is manual: edit input, click Run, inspect the result; *nothing executes when an example, mode, or break-it button is clicked*. Keep typed inputs in place while exploring a result. No animated loading effect that suggests model output before the request completes.

1. Intro: “Laya is a decision model, not a chatbot. Give it a state and ask a typed question; it returns probabilities, not generated prose.” Include a compact three-primitive visual key and a link to the existing CLI command.
2. Connection strip: service reachable/unreachable; loaded/unloaded; active providers if reported by `/health`. An unloaded model is normal and may reload on the next query. Warn visibly when CPU is primary; show “provider not yet verified” rather than claiming CUDA while unloaded. Health refresh is explicit or at page load, not a rapid polling loop.
3. Tabs: **Choice**, **Score**, **Noul**, **Batch**, **Raw query**. Each has a short “what question does this answer?” description; shared explanations are available inline, not a detached manual.
4. Results: top-line answer + chart/gauge first, explanatory notes second, collapsed **Raw Response** last. All numbers remain visible as text alongside colour/shape. On failure, show HTTP status and the upstream message instead of drawing a bogus chart; keep the user’s input for editing.
5. Bottom: **Try to break it** lab and **Explore more** disclosure cards. These teach the currently measured limits rather than silently preventing experiments.

### State editor (all guided tabs)

Switch between **Plain text** (sent as a JSON string) and **JSON** (parsed locally and sent as that JSON value). Label this explicitly: Laya sees the supplied state as input content, not as an instruction to generate text. JSON parse errors stay local with a readable inline message. Do not invisibly wrap plain text in invented `subject`/`body` fields. State is shared among guided tabs; editing it never starts inference.

### Choice

- Controls: question/instructions, 2+ editable rows of **option key** and optional **description**, add/remove row. Explain that a key identifies the option in the answer; a description adds context but spends tokens. Preserve row order; reject duplicate keys locally rather than overwriting one.
- Request: `{"state": ..., "questions": {"q": {"type": "choice", "instructions": ..., "criteria": {"key": "description", ...}}}}`.
- Result: winning key; ordered horizontal probability bars for **all** options, numeric P for each, concentration/confidence value with tooltip “not a correctness guarantee.” Show the sum/rounding caveat only if helpful (server rounds to four decimals).
- Adjacent limit note: practical initial experiments 2–32 options; more options and longer descriptions consume the **192-token head budget**, and tested 128 short options returned HTTP 500. There is no universal safe option count independent of text lengths. See `benchmarks/HARDENING-REPORT.md` §2.1.

### Score

- Controls: question/instructions and reorderable ordered level rows, lowest first; normal guided mode suggests **2–10 levels**. Make index **0 = lowest** conspicuous. Do not equate score with a 0–1 probability or a calibrated urgency scale.
- Request criteria is an array of level descriptions, not an object.
- Result: each level’s probability as a labelled bar, with the server’s `legend` mapped to indices; an expected-score marker on a **0 through N−1** axis; the exact `score` and `confidence` alongside. Explain that the marker is a probability-weighted mean, possibly between labels, **not a selected level**.
- Warning: hardening probes found non-monotonic ordinal results; do not recommend score as a reliable priority ranking. A one-level rubric caused a 500 crash, and 11 levels were accepted despite the documented 2–10 range. Keep 11-level experimentation possible in the break-it lab, with a warning, rather than presenting 11 as supported. See `benchmarks/HARDENING-REPORT.md` §2.2.

### Noul

- Controls: state and a proposition/question phrased so “true” has a clear meaning; optionally expand **true description** and **false description** to learn how criteria affect the answer. Use the service’s `instructions` field for the proposition.
- Result: large **P(true)** value and two-part true/false bar; show **P(false) = 1 − P(true)** as a derived complement, not an independently returned field. Include the actual numeric values and labels. Do not assign a “yes/no” verdict at an invented threshold.
- Warning: the hardening probes found negation blindness and factual mismatches; a high probability is not proof that a statement is true. See `benchmarks/HARDENING-REPORT.md` §2.3.

### Batch (many questions, one state)

- A single shared state with a small, editable list of questions. Each question has a distinct ID, type selector, instructions, and the appropriate option/level/proposition controls. Start with one choice, one score, one noul as an **editable example**; changing/removing any is allowed. Use one POST with `questions` mapping all IDs to specs.
- Show a separate answer card of the corresponding type for every returned ID; shared request timing and token count belong above the cards. This is **one HTTP request / one model forward pass**, not a demonstration of cross-request coalescing. Explain that `usage.input_tokens` aggregates rendered tokens across question sequences, not a single 512-token pool for an entire multi-question request.
- Add a prominent “Batch ≠ simultaneous HTTP requests” disclosure linking to the warning at the top of this spec and `benchmarks/HARDENING-REPORT.md` §3.

### Raw query (advanced)

- Full editable `/system_one` JSON body, with a documented starter template and a **Run raw query** button. Do not rewrite the payload into guided fields or auto-submit while typing. Display the same charts when the returned `answers` types are recognisable, plus collapsed complete raw request/response and the actual HTTP status. Allow server-rejected payloads to pass through for exploration; only syntactically invalid JSON is blocked locally.
- Keep raw and guided input independent so pasting arbitrary JSON cannot silently alter a guided example.

## 3. Metadata and interpretation

Every submitted request shows **browser-observed round-trip latency**, `usage.input_tokens` and `usage.output_tokens` when present, and the response’s `cached` boolean. Label timing honestly: browser-to-playground-to-service wall time includes local HTTP, tokenization/queueing and any reload; it is **not GPU kernel time**. Show `cached: true/false` as supplied, not inferred from speed.

Cold/warm: snapshot `/api/health` before submission (or refresh just before sending) to label **“model was unloaded before request; reload likely”** when applicable; otherwise say **“model loaded before request”**. Do not infer a cold path from latency alone, promise a cache hit while unloaded, or label the model’s provider from an empty unloaded-health response. After a query, refresh health to report the observed provider; if health cannot be fetched, mark it unknown. This status is advisory, not per-request execution tracing.

Each answer’s `rl_agent.act_probability` and `escalate_probability` appear in a subdued, expandable **Experimental routing signal** panel, separate from choice/score/noul probabilities and confidence. Label both **experimental / currently untrusted for operational routing**; no “act now” or “escalate now” recommendation. Do not silently substitute missing fields with 0.

The ~512-token limit applies **per rendered question sequence**, combining question type, instructions, option/level head and state; the head has its own ~192-token budget. `usage.input_tokens` is measured **after** any truncation and does **not** prove the original state fitted. The page cannot reliably calculate the model’s exact tokenizer count without the tokenizer: no fake remaining-token meter, no automatic “truncated” flag. Explain silent truncation clearly beside long inputs and provide a break-it example. See `src/laya_server/sequence.py` and `benchmarks/HARDENING-REPORT.md` §4.

## 4. Examples and deliberate failure lab

All examples are small, realistic, editable and **populate only**. Examples: support triage choice (billing vs account vs technical), urgency score with a user-written rubric, a factual noul proposition, and a three-question batch about the same support ticket. The initial page has no request history or stored state; nothing is sent until Brett presses Run.

A conspicuous **Try to break it** area has one-click “load this input” buttons, with expected behaviour and a link to the matching report section:

| Probe | Populate | Teaching point |
|---|---|---|
| Empty state | Plain-text empty state + noul | Currently accepted; output can be degenerate. |
| Huge state | Long state with a relevant fact near the end | Silent truncation may remove decisive facts; cannot infer exact cutoff from character count. |
| Lots of options | Choice with ~128 short options | Observed head overflow/HTTP 500, not a clean client error; description length changes the boundary. |
| 10-level score | Ordered ten-level rubric | Upper end of documented range; inspect entire distribution. |
| 11-level score | Eleven-level rubric | Documented invalid, but observed accepted; demonstration is **not** endorsement. |
| 1-level score | Raw query with one score level | Known HTTP 500; deliberately bypasses guided safeguards. |
| Negated proposition | Noul with “NOT” and an otherwise equivalent positive example | Observe measured negation weakness; run each yourself. |
| Bad question type | Raw query with unknown `type` | Upstream may return 500 rather than expected 4xx. |

No load button runs a probe. Destructive service actions (unload/restart/stop), stress floods and concurrency testing are deliberately absent.

## 5. Explore more: teach the next question

Inline disclosure cards open within the current page, without navigation or network fetches. One short paragraph, a concrete “try this next” prompt, and a real repository document reference per card:

- **Multi-select:** composition of one noul question per candidate; compare all probabilities, do not imply a native fourth primitive. `src/laya_server/primitives.py` and `README.md` §Python primitives.
- **Ranking:** a shared rubric across separate score questions in one request; note the measured score monotonicity failure, so rankings need independent evaluation. `src/laya_server/primitives.py` and `benchmarks/HARDENING-REPORT.md` §2.2b.
- **Inversion:** ask a similar decision as choice and as multiple binary noul questions; differences are observations, not an accuracy verdict. `benchmarks/FUTURE-METHODS.md`.
- **Many questions vs many requests:** one request with many questions uses one pass; multiple HTTP requests are currently serial at the worker. `src/laya_server/worker.py`, `src/laya_server/batcher.py`, `benchmarks/HARDENING-REPORT.md` §3.
- **Token budgets and truncation:** rendered head/state limits and no trustworthy client-side token counter. `src/laya_server/sequence.py`, `benchmarks/HARDENING-REPORT.md` §4.
- **What has actually been tested:** safe operating envelope, bugs, and deferred adoption evaluation. `benchmarks/HARDENING-REPORT.md`, `benchmarks/README.md`, `ROADMAP.md`.

Use the real file `benchmarks/SPEC-HARDENING.md` when citing the hardening spec; **`benchmarks/SPEC-1.0.md` does not exist in the inspected project**. Prior docs sometimes describe coalescing as functional; measured current behavior is in the hardening report, not those older claims.

## 6. Visual language and media

Follow the supplied Fox Software Solutions guide rather than Bootstrap or a generic admin console: dark warm gradient (`#0a0805` → `#100d09`), orange `#FF6B35`/`#E55A2A` accents, warm off-white text, subtle panel borders, restrained rounded cards, and an optional light theme using the supplied light tokens. Playfair Display/Source Sans 3 when available, with the guide’s system-font fallbacks. Do not assume this repo has the site's logo or `Note_this.ttf`: use text branding, no broken asset references. External fonts must be optional; the interface works offline.

Create four small **original text-free abstract vector illustrations**, locally served or inline SVG: branching paths (choice), gradient ladder (score), balanced halves (noul), converging streams (batch). Keep them decorative and lightweight; no downloaded assets, third-party image upload, or decorative media obscuring controls. Charts themselves are semantic HTML/CSS/SVG and encode values in text, bar length, and labels—not colour alone. Respect reduced-motion preferences; ensure keyboard-operated controls, visible focus, programmatic labels, readable contrast, and mobile layout. Use the supplied style guide as a visual reference, not a reason to copy the website's full header/footer or run a canvas animation.

## 7. Server and browser contract

- `GET /` and a small allowlist of static assets serve only the playground; no arbitrary filesystem paths or repository file serving.
- `GET /api/health` forwards the service's `/health`, reporting upstream HTTP status and JSON safely.
- `POST /api/query` accepts the **same JSON request body** the Laya endpoint expects, forwards it to `/system_one`, and preserves upstream status and JSON response, including error bodies. Keep a modest body-size limit and explicit timeout; surface unreachable/timeout errors in a consistent local gateway envelope without pretending they came from Laya.
- No proxy endpoint for `/control/unload`, no configurable upstream URL from browser input, no unbounded request queue and no auto-retries that duplicate manual experiments. Never interpolate user-provided text as HTML: create DOM nodes and use `textContent` for labels, messages and raw JSON. Restrict the server to loopback and set a restrictive browser policy compatible with local styles/scripts.
- Backend returns raw service data, not a parallel invented model contract; the frontend only assembles request JSON and renders the response. If the upstream adds fields later, retain them in Raw Response even if the visual card only knows the present types.

## 8. Build sequence and verification gate

1. **Confirm placement/port at implementation time.** Add a small local playground directory and a short start command in its own README. Do not edit the systemd unit, installed MCP, model bundle, or live service. Preserve existing untracked `benchmarks/results/hardening/` and `.qwen/`.
2. **Build the tiny proxy/static server**, with loopback-only bind, fixed upstream, safe static allowlist, upstream status passthrough, timeout/error mapping and health route.
3. **Implement the teaching UI and state editor**, then guided choice/score/noul editors and faithful rendering of each response. Add batch and raw query without hiding raw output.
4. **Add the manual lab, examples, explore panels and original mini-illustrations**, then polish responsive FSS styling and accessibility. No example automatically sends a request.
5. **Add permanent tests** using existing pytest: server routes/static traversal/loopback policy, forwarding exact payloads and upstream errors with mocked HTTP, field-to-payload mapping, example “populate only” interactions where practical, renderer fixtures for the three answers plus mixed batch, missing fields, zero values and error states. Avoid adding a test framework just for frontend code; a small browser-independent JS test approach or a permitted manual browser acceptance pass can cover interactions.
6. **Verify** project pytest/ruff for touched Python, exercise the local server against a mocked upstream, then (with explicit permission to use a browser) manually confirm layout, keyboard use and all four primitives against the real service. Check `/health` provider before attributing a result to GPU; if unloaded, verify after the request. Do not turn benchmark expectations into deterministic assertions about model semantics.

### Acceptance checks

- A user can start the page with one documented local command, enter an original state and one question, press Run, see the complete labelled probability structure without opening JSON, and intentionally open the raw response.
- Choice bars match all `probabilities`; score bars follow the server's `legend` and show expected score on 0..N−1; noul displays returned P(true) and explicitly derived P(false); mixed batch maps each ID to the appropriate visual card.
- Latency is identified as wall time; input token count is reported as returned; `cached` is displayed exactly; unloaded/CPU/unknown provider states are not falsely shown as GPU; routing heads remain clearly experimental.
- Every example/break-it button populates without running. Malformed raw inputs show either local JSON parse feedback or the actual upstream status/message. No clipboard, history, background requests, or persistence without a later request.
- The warning **“Separate HTTP requests are not coalesced”** is visible both near the introduction and in the batch explanation; no chart or copy implies otherwise.
- No dependency installation, production service mutation, remote deployment, or browser launch is necessary just to read this spec. The eight issues in `benchmarks/HARDENING-REPORT.md` remain deferred.

## Source of truth consulted

`src/laya_server/server.py`, `worker.py`, `sequence.py`, `batcher.py`, `client.py`, `cli.py`, `primitives.py`, `pyproject.toml`, `config/laya_config.json`, `benchmarks/HARDENING-REPORT.md`, `benchmarks/SPEC-HARDENING.md`, `benchmarks/README.md`, `README.md`, `SERVICE.md`, `ROADMAP.md`, and the Fox Software Solutions style guide supplied with the request. Where older docs conflict with current source or measured hardening results, this plan follows the latter.
