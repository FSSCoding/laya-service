# Roadmap — Laya GPU Inference Service

Working direction (not a fixed plan). The user (Brett) will pick up this project and drive it from
here; this file records where things stand and the intended next steps, so the CLI/primitives work
can continue without re-deriving context.

---

## Current state (2026-09-28)

The **service** is built, verified, and documented:
- GPU (CUDA) ONNX inference over HTTP on **:11448**, ~20× faster than CPU, decision-identical to the
  reference.
- Request batching, two-tier cache, idle-unload (VRAM released after 5 min idle or on demand),
  and a `laya-daemon` start/stop/status/unload CLI.
- Docs: README, LINK, SERVICE, CHANGELOG, ROADMAP. Parity + bench tests in-tree.

What is **not** done yet: systemd install (needs sudo), the CLI *client*, interface primitives, and
a remote repo.

---

## Phase 1 — finish the service (immediate)

- [ ] **Install the systemd unit + timer** (user's sudo; exact commands in LINK.md). Verify under
      `systemctl` that CUDA loads (the unit carries the CUDA-13 `LD_LIBRARY_PATH`).
- [ ] Decide **resident vs on-demand as the default**: keep `idle.unload_after_s=300` (current) so
      VRAM is released when idle, or set it permanent for a high-use period. Trivial config flip.
- [ ] **Repo**: init git (done locally — see below), review with user, then push to Gitea
      (`https://gitea.bobai.com.au/<owner>/laya-service`). Confirm the remote + owner.
- [ ] Optionally add a **`.gitignore`**-excluded `requirements.txt`/lock for reproducibility
      (currently deps are in the venv: `onnxruntime-gpu==1.30.0`, `tokenizers`, `fastapi`, `uvicorn`,
      `numpy`).

## Phase 2 — CLI client (the "streamline implementation" goal)

The user wants a **CLI that ties the tool together** — not just for humans, but as the *template*
other projects implement against (installable on any machine on the network). Intended shape:

- [ ] A `laya` CLI (e.g. a Python package installed into a venv, or a thin wrapper like
      `fss-chatterbox`) that wraps the HTTP endpoints:
  - `laya ask --state <json|file> --questions <json|file>` → formatted answers
  - `laya system-one` (raw JSON in/out, for scripting)
  - `laya status` / `laya unload` (thin wrappers over `/health`, `/control/unload`)
  - `laya bench` (runs the benchmark client-side)
- [ ] Config: `LAYA_ENDPOINT` (default `http://127.0.0.1:11448`) so the CLI works locally or against
      a remote server (the user may move the service to another box).
- [ ] **Install path** on other machines: a wrapper script + venv (mirror the `fss-chatterbox`
      `~/.local/bin` wrapper pattern) or a pip-installable package. Decide install mechanics.

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
- [ ] **GPU pinning / co-location**: currently pinned to card 0 via `CUDA_VISIBLE_DEVICES`. If the
      service moves or shares a box, re-decide which card and whether to add the Chatterbox-style
      VRAM lock (so ComfyUI knows the card is busy).
- [ ] **Ops**: alerting on `/health` (provider fell back to CPU?), log rotation, disk for the cache DB.

---

## Open decisions (for the user)

1. **Resident vs on-demand default** — keep the 300 s idle-unload, or run permanent for now?
2. **CLI form** — Python package + `~/.local/bin` wrapper (Chatterbox-style) vs pip-installable?
3. **Primitives** — pure CLI subcommands, or also an importable Python library? (and a Node client?)
4. **Remote** — move off bobai now or keep it here until there's another machine? (CLI already
   supports a remote endpoint.)
5. **Repo** — Gitea owner/repo name for the push.

## Design constraints (keep in mind)

- **Parity is the invariant.** Any change to sequence building / temperature / rounding must keep
  `test_parity.py` green (decision-identical, within 1e-3). GPU-vs-CPU FP32 non-determinism is
  expected and fine; a real logic change is not.
- **The model bundle is owned by `../laya`**, read-only here. Don't duplicate weights into this repo.
- **Don't hardcode the port or card** beyond config; the user owns port allocation and may relocate
  the service.
- **Lean & reliable** is the bar (per the embed-server template): small dep surface, one process,
  obvious failure modes, cheap when idle.
