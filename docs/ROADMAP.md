# Roadmap — Laya GPU Inference Service

Working direction (not a fixed plan). This file records where things stand and the intended next
steps, so the CLI/primitives work can continue without re-deriving context.

---

## Current state (2026-09-30)

The **service** is built, verified, and packaged for public release:
- GPU (CUDA) ONNX inference over HTTP on **:11448** (port configurable), ~20× faster than CPU,
  decision-identical to the CPU reference.
- Request batching with **real cross-request coalescing** (multiple concurrent requests merged
  into one forward pass), two-tier cache, idle-unload (VRAM released after 5 min idle or on
  demand), lazy reload on next request. `laya-server` console command + a systemd **template**
  for lifecycle.
- `laya` CLI — sole owner of systemd lifecycle; typed primitives layer (v0.3.0):
  `laya_choice`/`laya_score`/`laya_noul`/`laya_batch` + composed helpers (`classify`,
  `multi_select`, `rank`), CLI verbs, experimental routing signal.
- **Benchmark harness** (Phase A–C complete): smoke corpus (33 cases), quality metrics with
  validity guards, systems benchmark, provenance capture. A canonical measured snapshot lives in
  `benchmarks/baseline/`.
- **Service hardening** (Phase 3.5 complete): safe operating envelope established. Bugs found
  were fixed in a follow-up pass (pre-ONNX validation → 400, truncation detection, real
  coalescing). Report: `benchmarks/HARDENING-REPORT.md`.
- **Public-release preparation (2026-09-30):** docs restructured under `docs/`, config + systemd
  generalised (no machine-specific defaults), Apache-2.0 LICENSE + NOTICE added, package metadata
  filled, code-quality fixes, FSS branding.

What is **not** done yet: Windows/macOS support (researched, not ported), Jev API compatibility
(scoped, not implemented), remote deployment / ops hardening (Phase 4).

---

## Phase 1 — finish the service (DONE)

- [x] **Install the systemd unit + timer** — shipped as a template
      (`systemd/laya-service.service.template`) with explicit configuration points; operators
      substitute their machine values (see [INSTALL.md](INSTALL.md)).
- [x] Decide **resident vs on-demand as the default**: `idle.unload_after_s=300` is the default
      so VRAM is released when idle; set `0`/disabled for a permanently-resident high-use period.
- [x] **Repo** — ready for publication (see `PUBLIC-RELEASE-READINESS.md` once complete).
- [ ] Optionally add a **lockfile** for reproducibility (deps are in `pyproject.toml`).

## Phase 2 — CLI client (DONE)

Built as part of the `laya_server` package (`src/laya_server/cli.py`, entry point
`laya = "laya_server.cli:cli"`). Subcommands: `serve`, `query`, `choice`, `score`, `noul`,
`batch`, `health`, `stats`, `unload`, `version`, `start`, `stop`, `restart`, `status`.

- [x] **Inference**: `laya query` / `laya choice` / `laya score` / `laya noul` / `laya batch`
      with `--json` for scripting.
- [x] **Health/stats**: `laya health` / `laya stats` (thin wrappers over `/health`, `/v1/stats`).
- [x] **Lifecycle**: `laya start` / `laya stop` / `laya restart` (system unit, no `--user`;
      requires root) and `laya status` (combined systemd + health; `--json` for a frozen schema).
- [x] **Unload**: `laya unload` (thin wrapper over `/control/unload`).
- [x] **Endpoint config**: `--endpoint` flag → `LAYA_ENDPOINT` env → `http://127.0.0.1:11448`.

Remaining / not yet:
- [ ] Optionally a **`laya bench`** subcommand (currently benchmarking is via the standalone
      `bench.py`, not a CLI verb).

## Phase 3 — interface primitives (DONE)

Built as `src/laya_server/primitives.py`, importable from the package. Native wrappers, batch
primitive, composed helpers, experimental routing signal, and CLI verbs.

- [x] **Routing signal** — every answer carries `rl_agent` with `act_probability`,
      `escalate_probability`, and `experimental: True`.
- [x] **Native wrappers** — `laya_choice`, `laya_score`, `laya_noul`: thin typed wrappers over
      `/system_one` returning frozen dataclasses.
- [x] **Batch primitive** — `laya_batch`: one forward pass for many mixed-type questions.
- [x] **Composed helpers** — `classify`, `multi_select`, `rank` (independent score per item
      against a shared rubric, sorted desc).
- [x] **CLI verbs** — `laya choice`, `laya score`, `laya noul`, `laya batch` with `--json`.
- [x] **Tests** — primitives, CLI verbs, routing signal covered across parity / contract /
      integration suites.

Remaining / not yet:
- [ ] **Composed CLI verbs** for classify, multi_select, rank — Python-only for now.
- [ ] **Node.js client** — a thin HTTP client in the Node ecosystem would let both share the
      same GPU service.

## Phase 3.5 — Service Hardening & Operating Envelope (DONE)

Characterise and harden the three native primitives (`choice`, `score`, `noul`), plus batching,
coalescing, token limits, concurrency, and error handling. Spec: `benchmarks/SPEC-HARDENING.md`.

- [x] **Primitive contract testing** — option-count scaling (2→256, hard limit 128),
      score-level bounds (2–10 valid), noul mechanical semantics.
- [x] **Batching & coalescing** — real cross-request coalescing proven: multiple concurrent
      payloads merged into one forward pass via the worker's `system_many` (dynamic ONNX batch
      dimension). Batch saturation (1→512→513) with no errors.
- [x] **Token & truncation limits** — state-length sweep; truncation now detected and reported
      (`usage.truncated` + `original_input_tokens`).
- [x] **Systems & operational** — latency flat across 1–50 questions, cold start ~0.5 s,
      cache hit ~8 ms / miss ~26 ms.
- [x] **Validation & error-handling audit** — pre-ONNX validation now returns 400 (not 500) for
      client errors. Safe operating envelope table in `benchmarks/baseline/`.
- [x] **Small semantic probe set** — smoke corpus (33 cases) retained as regression baseline.
- [x] **Hardening report** — `benchmarks/HARDENING-REPORT.md` with envelope, scaling curves,
      bugs found, recommendations.

**Key findings:** score is non-monotonic under tested formulations (unsuitable for ordinal
ranking); silent truncation at 512 tokens is now detected; noul negation is an observed failure
mode under tested cases.

**Exit criteria met:** the native primitives have known input limits, scaling behaviour,
performance curves, failure modes, validation behaviour and concurrency characteristics; a small
semantic smoke baseline exists; future evaluation methodologies are documented and deferred.

**Deferred research** (preserved in `benchmarks/FUTURE-METHODS.md`, not current build):
large-scale semantic corpus (~300 cases), inversion pairs, consistency perturbation suite,
OOD taxonomy, domain-specific adoption slices, 27B reference comparison.

## Phase 4 — deployment / multi-machine

- [ ] **Move to a dedicated inference server**: the CLI's `LAYA_ENDPOINT` makes this a config
      change, not a code change. Document the remote setup (port, firewall, TLS if exposed
      beyond the local network).
- [ ] **GPU pinning / co-location**: pin via `CUDA_VISIBLE_DEVICES` (use an index the driver
      exposes). If the service shares a box, decide whether to add a VRAM lock so other tenants
      know the card is busy.
- [ ] **Ops**: alerting on `/health` (provider fell back to CPU?), log rotation, disk for the
      cache DB.

## Phase 5 — public release (IN PROGRESS, 2026-09-30)

- [x] Docs restructured under `docs/` (INSTALL, ARCHITECTURE, MODEL-DOCTRINE, ROADMAP,
      playground/, research/).
- [x] Config + systemd generalised (no machine-specific defaults); `model_dir` now required.
- [x] Apache-2.0 LICENSE + NOTICE; package metadata filled.
- [x] Code-quality fixes (provenance attribution, no fabricated VRAM figures, lint cleanup).
- [x] FSS branding (restrained logo + attribution) in the playground.
- [ ] Research docs: Windows/macOS feasibility + Jev compatibility scope (no implementation).
- [ ] Final verification: lint, tests, service + playground smoke, clean-install simulation.
- [ ] `PUBLIC-RELEASE-READINESS.md` with the final verified state.

---

## Open decisions (for the operator)

1. **Resident vs on-demand default** — keep the 300 s idle-unload (current), or run permanent
   for a high-use period? Trivial config flip (`idle.unload_after_s`).
2. ~~**CLI form**~~ — **resolved**: the `laya` CLI ships as part of the `laya_server` package.
3. ~~**Primitives**~~ — **resolved**: `laya_server.primitives` module + CLI verbs.
4. **Remote** — keep the service local, or move it to a dedicated inference server? (The CLI
   already supports a remote endpoint via `LAYA_ENDPOINT`.)

## Design constraints (keep in mind)

- **Parity is the invariant.** Any change to sequence building / temperature / rounding must keep
  `test_parity.py` green (decision-identical, within 1e-3). GPU-vs-CPU FP32 non-determinism is
  expected and fine; a real logic change is not.
- **The model bundle is owned elsewhere**, read-only here. Don't duplicate weights into this repo.
- **Don't hardcode the port or card** beyond config; the operator owns port allocation and may
  relocate the service.
- **Lean & reliable** is the bar: small dep surface, one process, obvious failure modes, cheap
  when idle.
- **Adoption gate.** No operational Laya decision is adopted solely from general benchmark
  performance. Each production decision requires a representative domain-specific evaluation
  slice with pre-agreed acceptance criteria.
