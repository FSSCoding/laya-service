# Roadmap — Laya GPU Inference Service

Working direction (not a fixed plan). The user (Brett) will pick up this project and drive it from
here; this file records where things stand and the intended next steps, so the CLI/primitives work
can continue without re-deriving context.

---

## Current state (2026-09-29)

The **service** is built, verified, deployed, and pushed:
- GPU (CUDA) ONNX inference over HTTP on **:11448**, ~20× faster than CPU, decision-identical to
  the reference.
- Request batching, two-tier cache, idle-unload (VRAM released after 5 min idle or on demand),
  with lazy reload on next request. `laya-server` console command + systemd unit for lifecycle.
- **Installed as a system systemd unit on bobai** (`/etc/systemd/system/laya-service.service` +
  weekly restart timer; drop-in override re-pins `CUDA_VISIBLE_DEVICES=1`). Journal confirms
  `systemd[1]` owns the process; `/health` shows `CUDAExecutionProvider` first.
- `laya` CLI in the venv (`.venv/bin/laya`) — sole owner of systemd lifecycle; JS MCP server at
  `~/.qwen/mcp-servers/laya.js`.
- **Repo pushed** to Gitea: `https://gitea.bobai.com.au/foxadmin/laya-service.git`.
- **Typed primitives** (v0.3.0): `laya_choice`/`laya_score`/`laya_noul`/`laya_batch` +
  composed helpers (`classify`, `multi_select`, `rank`), CLI verbs, experimental routing signal.
- **Benchmark harness** (Phase A complete): smoke corpus (33 cases), quality metrics with
  validity guards, systems benchmark, provenance capture, end-to-end pipeline proven.
  Correctness pass done (noul false-label bug fixed, metric applicability guards added,
  routing-head validity gating, confidence-AUC independence fix).
- **Service hardening** (Phase 3.5 complete 2026-09-29): safe operating envelope established.
  8 bugs/issues identified (coalescing non-functional, score non-monotonic, silent truncation,
  500-vs-400 validation errors, noul negation blindness). Report: `benchmarks/HARDENING-REPORT.md`.

What is **not** done yet: remote deployment / ops hardening (Phase 4), and the batcher
fix (serial `_process_batch` should merge into one forward pass).

---

## Phase 1 — finish the service (DONE except lockfile + resident/on-demand decision)

- [x] **Install the systemd unit + timer** — installed on bobai 2026-09-28: system unit
      `/etc/systemd/system/laya-service.service` + `laya-service-restart.timer`, drop-in override
      `laya-service.service.d/override.conf` re-pinning `CUDA_VISIBLE_DEVICES=1`. Journal confirms
      `systemd[1]` owns the process; `/health` shows `CUDAExecutionProvider` first.
- [ ] Decide **resident vs on-demand as the default**: keep `idle.unload_after_s=300` (current) so
      VRAM is released when idle, or set it permanent for a high-use period. Trivial config flip.
- [x] **Repo**: pushed to Gitea `https://gitea.bobai.com.au/foxadmin/laya-service.git` (`master`,
      clean tree, 2026-09-28).
- [ ] Optionally add a **lockfile** for reproducibility (deps are in `pyproject.toml`).

## Phase 2 — CLI client (DONE — the `laya` CLI ships in the package)

Built as part of the `laya_server` package (`src/laya_server/cli.py`, entry point
`laya = "laya_server.cli:cli"`), installed into the service venv at `.venv/bin/laya`. Subcommands:
`serve`, `query`, `health`, `stats`, `unload`, `version`, `start`, `stop`, `restart`, `status`.

- [x] **Inference**: `laya query --state <json|file> --questions <json|file> [--json]` →
      human-formatted answers or raw JSON for scripting.
- [x] **Health/stats**: `laya health` / `laya stats` (thin wrappers over `/health`, `/v1/stats`).
- [x] **Lifecycle**: `laya start` / `laya stop` / `laya restart` (system unit, no `--user`;
      requires root) and `laya status` (combined systemd + health; `--json` for a frozen schema).
- [x] **Unload**: `laya unload` (thin wrapper over `/control/unload`).
- [x] **Endpoint config**: `--endpoint` flag → `LAYA_ENDPOINT` env → `http://127.0.0.1:11448`, so
      the CLI works locally or against a remote server.
- [x] **Installed** on bobai in the service venv (editable install tracks the source tree).

Remaining / not yet:
- [ ] **Cross-machine install path** beyond bobai: a `~/.local/bin` wrapper (Chatterbox-style) or a
      pip-installable distribution. Decide mechanics before the service moves off bobai.
- [ ] Optionally a **`laya bench`** subcommand (currently benchmarking is via the standalone
      `bench.py`, not a CLI verb).

## Phase 3 — interface primitives (DONE — typed primitives layer in `laya_server`)

Built as `src/laya_server/primitives.py`, importable from the package. Native wrappers,
batch primitive, composed helpers, experimental routing signal, and CLI verbs.

- [x] **Routing signal** — every answer now carries `rl_agent.routing_signal` with
      `act_probability`, `escalate_probability`, and `experimental: True`. The escalate
      head (`act_probs[1]`) is exposed under an unmistakable experimental flag.
- [x] **Native wrappers** — `laya_choice`, `laya_score`, `laya_noul`: thin typed wrappers
      over `/system_one` that return frozen dataclasses (`ChoiceResult`, `ScoreResult`,
      `NoulResult`).
- [x] **Batch primitive** — `laya_batch`: one forward pass for many mixed-type questions.
      Returns `BatchResult` with per-qid dataclass answers.
- [x] **Composed helpers** — `classify` (sugar over choice), `multi_select` (batched noul
      per candidate, returns full probability dict), `rank` (independent score per item
      against a shared caller-supplied rubric, sorted desc by expected value).
- [x] **CLI verbs** — `laya choice`, `laya score`, `laya noul`, `laya batch` with
      `--json` flag for structured output.
- [x] **Tests** — 22 new tests (primitives, CLI for new verbs, routing signal in parity /
      contract / integration). Full suite: 70 tests, all green.
- [x] **Decision/query inversion** — documented and testable via the primitives (pose the
      same problem as choice vs. multi_select and compare). No helper shipped yet.

Remaining / not yet:
- [ ] **Composed CLI verbs** for classify, multi_select, rank — Python-only for now; add
      CLI verbs only if they prove genuinely convenient during use.
- [ ] **Node.js client** — the lib at `../laya` is Node; a thin HTTP client there would let
      both ecosystems share the same GPU service.

## Phase 3.5 — Service Hardening & Operating Envelope (DONE 2026-09-29)

Characterise and harden the three native primitives (`choice`, `score`, `noul`), plus batching,
coalescing, token limits, concurrency, and error handling. Establish the mechanical operating
envelope before operational use. Spec: `benchmarks/SPEC-HARDENING.md`.

- [x] **Primitive contract testing** — option-count scaling (2→256, hard limit 128),
      score-level bounds (1→crash, 2–11 work), noul mechanical semantics, controlled
      monotonic ladders (score non-monotonic at all level counts).
- [x] **Batching & coalescing** — 5 ms window (not functioning — serial processing),
      batch saturation (1→512→513, no errors, 0.3–0.5 rps), heterogeneous batches OK.
- [x] **Token & truncation limits** — state-length sweep (silent truncation at 512 tokens),
      combined budget (500-token state + 32-choice = 512), error handling at limits.
- [x] **Systems & operational** — latency flat at 8ms (1–50q), cold start 534ms,
      cache hit 8.3ms / miss 25.9ms, VRAM 8.8GB, concurrency 0.3 rps, error recovery clean.
- [x] **Validation & error-handling audit** — 500s where 400s expected, silent truncation,
      score 1-level crash. Safe operating envelope table produced.
- [x] **Small semantic probe set** — smoke corpus (33 cases) retained as regression baseline.
- [x] **Hardening report** — `benchmarks/HARDENING-REPORT.md` with envelope, scaling curves,
      8 bugs/issues found, recommendations.

**Key findings:** Coalescing not functioning (serial batch processing); score non-monotonic
(unsuitable for ordinal ranking); silent truncation at 512 tokens; 500 instead of 400 for
validation errors; noul negation blindness.

**Exit criteria met:** The native primitives have known input limits, scaling behaviour,
performance curves, failure modes, validation behaviour and concurrency characteristics;
a small semantic smoke baseline exists; future evaluation methodologies are documented and
deferred.

**Next: Use it.** Run actual inferencing. Form intuition through interaction.

**Deferred research** (preserved in `benchmarks/FUTURE-METHODS.md`, not current build):
large-scale semantic corpus (~300 cases), inversion pairs, consistency perturbation suite,
OOD taxonomy, domain-specific adoption slices, 27B reference comparison.

## Phase 4 — deployment / multi-machine

- [ ] **Move to a dedicated inference server** (the user may do this): the CLI's `LAYA_ENDPOINT`
      makes this a config change, not a code change. Document the remote setup (port, firewall on
      the 10.23.88.0/24 LAN, TLS if exposed beyond the LAN).
- [ ] **GPU pinning / co-location**: currently pinned to card 1 via `CUDA_VISIBLE_DEVICES` (card 0
      hosts other tenants). If the service moves or shares a box, re-decide which card and whether
      to add the Chatterbox-style VRAM lock (so ComfyUI knows the card is busy).
- [ ] **Ops**: alerting on `/health` (provider fell back to CPU?), log rotation, disk for the cache DB.

### Incident record — transient CUDA-driver wedge (resolved 2026-09-28)

During deployment on 2026-09-28 the service briefly ran on **CPU fallback** because the machine's
userland CUDA driver was wedged (no process could create a context, not just Laya). It **cleared
on its own the same day** — no action was required from this repo. Keeping the evidence here so
the same symptom is recognized fast next time.

Symptoms (when it happened):
- `nvidia-smi` (kernel module 580.159.03) reported **two** GPUs (`0000:01:00.0`, `0000:03:00.0`),
  but the process libcuda view showed a phantom device matching neither; root in a fresh shell saw
  **0** devices.
- `journalctl -k` showed repeated `NVRM: nvAssertFailedNoLog ... NV_ERR_NO_MEMORY ... vaspace_api.c`
  / `pool_alloc.c` assertion failures.
- ORT threw `EP Error ... CUDA failure 100: no CUDA-capable device ; GPU=-1`; the worker fell back
  to `CPUExecutionProvider` (`/health` → `providers` empty / CPU-only).

Diagnosis + fix: an OS/CUDA-driver-layer fault (userland runtime out of sync with the kernel
module), **outside this repo**. Confirmed resolved when `libcuda` again reported 2 devices and
Laya loaded on the GPU (verified 2026-09-28: `/health` → `CUDAExecutionProvider` first,
`input_tokens` 267). If it recurs, the heavy-handed fix is a driver reload
(`rmmod nvidia* && modprobe nvidia` or reboot) — but that evicts every other GPU tenant on the box,
so coordinate a maintenance window first.

---

## Open decisions (for the user)

1. **Resident vs on-demand default** — keep the 300 s idle-unload (current), or run permanent for
   a high-use period? Trivial config flip (`idle.unload_after_s`).
2. ~~**CLI form**~~ — **resolved**: the `laya` CLI ships as part of the `laya_server` package
   (editable install into the service venv). Cross-machine install path is still open (see Phase 2).
3. ~~**Primitives**~~ — **resolved**: `laya_server.primitives` module (native 3 + batch +
   composed helpers) and CLI verbs (`laya choice/score/noul/batch`). See Phase 3.
4. **Remote** — move off bobai now or keep it here until there's another machine? (CLI already
   supports a remote endpoint via `LAYA_ENDPOINT`.)
5. ~~**Repo**~~ — **resolved**: pushed to `https://gitea.bobai.com.au/foxadmin/laya-service.git`.

## Design constraints (keep in mind)

- **Parity is the invariant.** Any change to sequence building / temperature / rounding must keep
  `test_parity.py` green (decision-identical, within 1e-3). GPU-vs-CPU FP32 non-determinism is
  expected and fine; a real logic change is not.
- **The model bundle is owned by `../laya`**, read-only here. Don't duplicate weights into this repo.
- **Don't hardcode the port or card** beyond config; the user owns port allocation and may relocate
  the service.
- **Lean & reliable** is the bar (per the embed-server template): small dep surface, one process,
  obvious failure modes, cheap when idle.
- **Adoption gate.** No operational Laya decision is adopted solely from general benchmark
  performance. Each production decision requires a representative domain-specific evaluation
  slice with pre-agreed acceptance criteria. General hardening first; domain slice before
  adoption; real-world monitoring afterwards.
