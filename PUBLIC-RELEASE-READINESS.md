# Laya Service — Public Release Readiness

**Date:** 2026-10-06 (final pre-upload verification re-run 2026-10-09)
**Status:** Ready as a clean public-release candidate. No push, no GitHub repo, no release tag performed (hard boundary held).
**Scope:** Release-preparation pass on an already-complete service. Behaviour preserved; no redesign, no new features, no porting, no Docker/auth, no Jev adapter implementation.

---

## Acceptance gate (stranger test)

**Verdict: PASS.** A stranger on supported Linux can, from this checkout alone:
clone → understand what Laya is (and isn't) → obtain the model from Hugging Face and configure
`model_dir` → install into a venv → start the service → verify CUDA/CPU via `/health` providers →
run all three primitives (`choice`/`score`/`noul`) via HTTP, the CLI, and the Python client → open the
playground → and read the documented limitations — **without any Brett-specific paths, machines, IPs,
or GPU-card assumptions.**

Verified against a real model and a live CUDA service on 2026-10-06 (results below).

---

## What changed (commits)

| Commit | Scope |
|---|---|
| `1edbfcd` | Restructure for public release: `docs/` tree, `examples/probes/`, canonical benchmark baseline, `.gitignore` hygiene; removed runtime/dev-only files from git. |
| `aacee5e` | Generalise config + systemd: `model_dir` required (no default), ship `config/laya_config.example.json`, generic `systemd/laya-service.service.template`; tests/bench read env-overridable model dir. |
| `23ba42d` | Add Apache-2.0 `LICENSE` + `NOTICE`; code-quality fixes (provenance docstrings, dead `/MasterFOLDER` paths removed, fabricated stats removed, duplicate probe removed); fill `pyproject` metadata. |
| `4e42d50` | Playground: FSS fox logo (compressed 138 KB→13 KB) + footer credit; `--upstream` flag with loopback guard; accurate font note. |
| `6060f3b` | Stranger-facing docs rewrite + full private-reference sanitisation across README, INSTALL, ARCHITECTURE, ROADMAP, CHANGELOG, benchmarks, playground. Corrected a stale ROADMAP claim (coalescing does work). |
| `1945f4c` | Two research/scoping docs (Windows/macOS feasibility, Jev compatibility scope), `examples/probes/README.md`, lightweight `CONTRIBUTING.md`, fixed stale probe run-paths. |

Total: 6 commits, ~69 files touched. Functional behaviour unchanged (parity bar holds).

---

## Sanitised list (private references removed)

Swept the whole tree for: `/MASTERFOLDER`, `/home/bob`, `bobai`, `10.23.88.*`, `foxadmin`,
`sudo-gui`, Ollama CUDA path, MCP server path, private Gitea URL. After the pass, **no hit remains in
any tracked file** (the two scratch docs were deleted at the end; `QWEN.md` is gitignored and
untracked — see Unresolved items).

Removed/generalised, by category:

- **Paths:** all `/MASTERFOLDER/...` and `/MasterFOLDER/...` → repo-relative or configurable
  (`model_dir`, `$LAYA_HOME`, `$MODEL_DIR`).
- **Model dir default:** `_config.py` no longer defaults `model_dir` to a machine path; it is
  **required** and validated (fails clearly if absent/nonexistent). Example config ships with a
  `CHANGE_ME` placeholder.
- **Machine/host:** `bobai`, `10.23.88.5`, `10.23.88.0/24` → "the deployment host" / removed.
- **Person/user:** `Brett`, `bob` → "the operator" / `__SERVICE_USER__` placeholder.
- **GPU card pin:** hard-coded card `1` → generic `<gpu-card>` / `__GPU_CARD__` placeholder (docs note
  to pick a free card).
- **CUDA lib path:** Ollama-specific `/usr/local/lib/ollama/cuda_v13` → generic
  `ldconfig -p | grep cublasLt` guidance with the Ollama path demoted to one worked example.
- **Privilege tooling:** `sudo-gui` → `sudo` (with a note that systemd install needs root).
- **MCP server path:** `~/.qwen/mcp-servers/laya.js` → removed (not part of the shipped service).
- **Private remote:** `gitea.bobai.com.au/foxadmin/laya-service.git` → removed; `pyproject` now points
  at the (future) public GitHub URL.
- **Dead provenance docstrings:** `worker.py`/`sequence.py`/`batcher.py`/`result_cache.py` no longer cite
  `/MasterFOLDER` source paths; `sequence.py` cites `@receptron/laya (MIT)` instead.
- **Runtime/dev files untracked:** `config/laya_config.json` (machine config), `systemd/laya-service.service`
  (installed bobai unit), `benchmarks/results/`, `PLAYGROUND-INCIDENT-2026-09-29.md`, `.pb-context`, `.qwen/`,
  `.ruff_cache/`.

---

## Licensing (conclusion)

- **Service code:** Apache-2.0. Copyright 2026 Fox Software Solutions. (`LICENSE`)
- **Model weights:** Apache-2.0 (Convai Innovations, Hugging Face `convaiinnovations/laya`). The
  weights are **not redistributed** with this repo — `NOTICE` carries the attribution and the
  download pointer. `model_dir` points at a separately-obtained bundle.
- **Upstream `@receptron/laya`:** MIT © 2026 Receptron. `sequence.py` is a Python port of
  `sequence.ts`; attribution recorded in the module docstring and `NOTICE`.
- **Dependencies:** all permissive (FastAPI, onnxruntime-gpu, Pydantic, httpx, Click, etc.). No copyleft
  contamination.
- **Trademark caveat:** "Jev"/"TypeSafe" are third-party trademarks. Docs describe Laya as compatible
  with the Jev `system_one` *shape* and never imply partnership/endorsement. `NOTICE` restates this.

**No ambiguous licensing found.** (The one stop-trigger condition — ambiguous license — did not arise.)

---

## Platform verified

**Linux x86_64 with an NVIDIA GPU and CUDA 13 libraries** is the supported and verified platform.

- **CUDA active:** `/health` and `/v1/stats` report `providers: ["CUDAExecutionProvider","CPUExecutionProvider"]`
  (CUDA first) on the live service.
- **All four HTTP endpoints** return correct shapes: `/health`, `/system_one` (choice/score/noul/batch),
  `/v1/stats`, `/control/unload`.
- **Routing envelope:** every answer carries `rl_agent: {act_probability, escalate_probability, experimental:true}`.
- **CLI:** 14 subcommands; `health` and `choice` exercised live.
- **Playground:** static page serves, `/api/health` + `/api/query` proxy to the live upstream, FSS logo +
  footer credit render, loopback-only.
- **Parity:** test suite 94 passed / 10 skipped (skips = GPU-only parity tests). Decision invariant within
  the `1e-3` numeric bar; no exact-decimal assertions.

**Windows / macOS:** out of scope for this pass. See `docs/research/WINDOWS-MACOS-FEASIBILITY.md` —
macOS = do-not-port (no CUDA EP); Windows = feasible-in-principle but not built (needs a Win service +
`PATH`/CUDA-13 rewrite + its own CI leg).

---

## Verification results (recorded 2026-10-06)

| Check | Result |
|---|---|
| `ruff check src tests playground bench.py` | **All checks passed** (exit 0) |
| `pytest tests/` (CUDA env) | **94 passed, 10 skipped** (10 = GPU parity, skipped on CPU fallback) |
| `/health` | `status:degraded`→`ok` after warm; `providers: [CUDA, CPU]`; `model_loaded:true` |
| `/system_one` choice | `choice:network_outage`, valid probabilities + `confidence` + `rl_agent` |
| `/system_one` score | `score:0.8246`, valid `legend` + `probabilities` + `confidence` |
| `/system_one` noul | `noul:0.7796` (P(true)) |
| `/v1/stats` | requests, latency p50/p99, cache, `batch_coalesced`, providers all present |
| CLI `laya health` / `laya choice` | correct output incl. routing line |
| Clean-install (example config) | raises clear `ValidationError: model_dir does not exist: CHANGE_ME...` |
| Clean-install (real `model_dir`) | loads cleanly, neutral defaults (host 127.0.0.1, port 11448, visible_devices 0) |
| Playground | static + health proxy + query proxy + logo (200, 13 KB) + footer credit |
| Repo-wide secret/private-ref sweep | only hit: `QWEN.md` (now gitignored/untracked; not in public tree) |

---

## Limitations & model character (preserved philosophy)

Laya is a **bounded semantic decision tool**, not a chatbot or oracle. The documentation preserves
this doctrine:

- It returns **calibrated probabilities over typed questions**, never generated text (`output_tokens` is always 0).
- `score` is an **expected value over the level distribution**, not a guaranteed monotonic ordinal — use
  pairwise `choice` for ranking (documented in `docs/MODEL-DOCTRINE.md` + `benchmarks/HARDENING-REPORT.md`).
- `noul` direction is correct but **not saturated** (≈0.1–0.9) and is **negation-sensitive** (treat negated
  formulations as unsafe until characterised).
- `choice` is reliable at **2–32 options**; 33+ degrades (low entropy); ~120–128 options hits the
  `head_max_len` hard limit.
- State is **silently truncated** at `max_len` (512 tokens) — now surfaced via `usage.truncated` +
  `usage.original_input_tokens`.

---

## Excluded / not-included

- **Not redistributed:** model weights (Apache-2.0, Convai) — obtained separately, referenced by `model_dir`.
- **Not committed:** `config/laya_config.json` (machine config; example shipped instead), installed
  systemd unit (generic template shipped), `benchmarks/results/` (gitignored; canonical baseline in
  `benchmarks/baseline/`), dev/runtime scratch (`.pb-context`, `.qwen/`, `.ruff_cache/`).
- **Removed (dev-only memory):** `PLAYGROUND-INCIDENT-2026-09-29.md` (incident log, not public material).
- **Not implemented (research-only):** Windows/macOS port, Jev `system_one` client adapter.

---

## Research / scoping docs added (no implementation)

- **`docs/research/WINDOWS-MACOS-FEASIBILITY.md`** — portability analysis. Linux supported; macOS
  do-not-port; Windows feasible-in-principle, not yet. No code/CI changes.
- **`docs/research/JEV-COMPATIBILITY-SCOPE.md`** — field-by-field map vs the base Jev `system_one` shape
  (from `@receptron/laya` `types.ts`). Data-compatible; 6 additive divergences; 4 protocol gaps flagged
  as unverified pending the official spec; adapter scoped (~1 module), **not built**.

---

## Unresolved items (need your action)

1. **`QWEN.md` is gitignored, not sanitised.** It is the agent operating context for this repo on
   this machine (live deploy details: host, GPU pin, CUDA lib path, systemd state, private paths). It
   is kept local and **excluded from the public tree via `.gitignore`** — a stranger installing the
   service doesn't need it (that story lives in `docs/INSTALL.md`). It stays useful to agents working
   here. It was `git rm --cached`'d so it no longer ships with a clone.
2. **`[project.urls] repository`** points at
   `https://github.com/foxsoftwaresolutions/laya-service` — verify this matches the org/repo you will
   create. No repo was created and nothing was pushed.
3. **Future results convention:** `benchmarks/results/` is gitignored; confirm the team will keep future
   runs out of git (canonical baseline already committed under `benchmarks/baseline/`).

---

## Hard boundary (held)

- **No GitHub repository created.**
- **No push / no publish.**
- **No release tag.**
- Stop-trigger conditions (ambiguous licensing, secrets in history requiring rewrite, sanitisation
  altering behaviour) — **none triggered.** Git history contains no secrets (audited during recon);
  sanitisation changed no functional behaviour (parity bar holds).

---

## Final pre-upload verification (2026-10-09)

Re-ran the full gate on the verified release candidate. Results (unchanged from the 2026-10-06
run except the two doc-consistency fixes below):

- **Lint:** `ruff check src tests playground bench.py` → all checks passed.
- **Tests:** `pytest tests/` → **94 passed, 10 skipped** (skips = GPU-only parity).
- **Live (real model, CUDA EP):** `/health` ok + `CUDAExecutionProvider` first; choice/score/noul and a
  mixed choice+score+noul batch all returned correct shapes with the `rl_agent` routing envelope;
  `/v1/stats` normal. Playground: static title, logo (200, 13 KB), `/api/health` + `/api/query` proxy,
  all against a live upstream.
- **Clean-install:** example config (`CHANGE_ME`) raises a clear `ValidationError`; real `model_dir` loads
  with neutral defaults.
- **Secrets/private-refs:** none in tracked HEAD (only the sanitisation-list mentions in this file).
  Full history has **no credentials**; the only private-host reference in history is the descriptive
  "repo pushed to Gitea" note in old docs (the private mirror URL, not a credential).
- **`onnxruntime-gpu`:** confirmed required (`worker.py` does `import onnxruntime as ort`; declared as a
  runtime dep) — kept.
- **Scanner dispositions:** `CHANGE_ME` placeholder in the example config and the readiness-doc
  references kept intentionally; the Google Fonts `wght@600` basic-auth flag treated as a false positive
  (URL unchanged); `brett@foxsoftwaresolutions.com.au` retained in `pyproject.toml` (no packaging/privacy
  reason to change it).

**Two doc-consistency fixes made in this pass** (no behaviour change):

1. `README.md` — removed the "calibrated probabilities" overclaim (3 places) → "confidence /
   directional confidence probabilities", consistent with the doctrine and the hardening data
   (outputs are directional, not saturated/calibrated).
2. `benchmarks/HARDENING-REPORT.md` — reconciled internal contradiction: the safe-envelope table and
   Key-numbers now state coalescing **works** and truncation is **reported** (current shipped behaviour);
   §3.1/§4.1 pre-fix findings are preserved as dated historical records; a post-report-fixes note added.
   This resolves the stale "coalescing NOT functioning" / "silent truncation" claims that contradicted the
   report's own Bug/Recommendations table.

**Cross-document consistency (verified):** README/INSTALL/ARCHITECTURE/ROADMAP/MODEL-DOCTRINE/
HARDENING-REPORT/CHANGELOG agree on coalescing (works), truncation (now reported), supported platform
(Linux + NVIDIA GPU + CUDA 13), routing-signal status (experimental), model acquisition
(`huggingface.co/convaiinnovations/laya`), config path (`config/laya_config.json` + example), and the
three supported primitives. No stale exact test-count claims; Windows/macOS and Jev compatibility are
presented as research-only (not implemented); model weights are consistently stated as **not
redistributed** (obtained separately). `sequence.py` MIT provenance retained. Future public URL in
`pyproject.toml` is exactly `https://github.com/foxsoftwaresolutions/laya-service` (repo not yet created —
reported only, not created).
