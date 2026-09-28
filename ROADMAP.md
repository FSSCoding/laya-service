# Roadmap — Laya GPU Inference Service

Working direction (not a fixed plan). The user (Brett) will pick up this project and drive it from
here; this file records where things stand and the intended next steps, so the CLI/primitives work
can continue without re-deriving context.

---

## Current state (2026-09-28)

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
- Docs: README, LINK, SERVICE, INSTALL, CHANGELOG, ROADMAP. Parity + bench tests in-tree.

What is **not** done yet: the typed interface primitives (Phase 3) and remote deployment / ops
hardening (Phase 4).

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

## Phase 3 — interface primitives (so primitives aren't rebuilt every time)

"Build in a few easier-to-interface-with variations of how this tool can be used." Intended:

- [ ] **Typed client functions** exposing each question type as a first-class primitive, e.g.:
  - `laya_choice(state, instructions, options: dict|list) -> {choice, probabilities, confidence}`
  - `laya_score(state, instructions, levels: list) -> {score, legend, probabilities}`
  - `laya_noul(state, statement) -> {noul}`  (P(true) of a yes/no statement)
  These wrap `/system_one` so callers don't hand-build the request shape each time.
- [ ] **Batch primitive**: one call that takes many questions and returns all answers (the
  efficiency win — grouping questions into one forward pass).
- [ ] Optionally a **small library** (Python) bundling these, importable from other FSS projects —
  the "template" the user described. Keep it thin; the GPU work stays server-side.
- [ ] Consider a **Node.js client** only if a Node project needs it (the lib at `../laya` is Node;
  a thin HTTP client there would let both ecosystems share the same GPU service).

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
3. **Primitives** — pure CLI subcommands, or also an importable Python library? (and a Node client?)
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
