# Laya GPU Inference Service

**Stable, GPU-accelerated Laya inference over HTTP** — a 421M-param System-1 decision model
(onnxruntime-gpu) exposed as a lean FastAPI service on **port 11448**, pinned to one RTX 3090.

Laya does not generate text: it takes a *state* (ticket, email, JSON object) and typed questions,
and returns calibrated probabilities for every answer in a single forward pass. This service wraps
that in a production topology (resident model, request coalescing, caching, systemd liveness) so it
can be addressed over HTTP by other projects or a CLI.

---

## Why this exists

The library at `/MASTERFOLDER/Services/laya` (`@receptron/laya`) runs Laya from Node.js via
`onnxruntime-node` — but **Node.js has no official CUDA build** (verified 2026-09-28; see the
research doc at `/MASTERFOLDER/Services/laya/docs/LAYA-GPU-SERVICE.md`). The clean way to run Laya
on a GPU is to move inference into an officially-supported CUDA runtime (`onnxruntime-gpu`) and
serve it over HTTP — the same topology as `/MASTERFOLDER/Services/embed-server` (which we used as
the template). This project **is** that service.

**Result: ~20× faster than CPU** (297 ms → 15 ms warm single call on the 3090), decision-identical
to the reference, deterministic, and cache-friendly.

---

## Quick start

Install the package (all runtime dependencies are included — this is a service package):

```bash
cd /MASTERFOLDER/Services/laya-service
.venv/bin/python -m pip install -e ".[dev]"
```

```bash
# Parity check (CUDA):
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m pytest tests/ -v

# Benchmark (CPU vs GPU, single + batched):
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python bench.py

# Run the server (foreground):
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/laya-server
# → http://0.0.0.0:11448
```

The `LD_LIBRARY_PATH` is **required for CUDA** — without it the CUDA EP silently falls back to CPU
(`libcublasLt.so.13` not found). See [LINK.md](LINK.md) §"CUDA 13 library path".

---

## CLI

The `laya` CLI is the primary interface for humans and scripts, and the sole owner of systemd
lifecycle:

```bash
laya query --state '{"subject":"Refund"}' \
          --questions '{"urgency":{"type":"score","instructions":"How urgent?","criteria":["low","med","high"]}}'
laya health
laya stats
laya status --json      # stable schema for automation
laya start              # systemctl start laya-service   (system unit; root)
laya stop
laya restart
laya unload             # free ~2.4 GB VRAM; lazily reloads on next query
```

Subcommands: `serve`, `query`, `health`, `stats`, `unload`, `version`, `start`, `stop`,
`restart`, `status`. `start`/`stop`/`restart` manage the **system** unit (no `--user`) and
require root (route through `sudo-gui`); `status` is readable by any user.

Endpoint resolution: `--endpoint` flag → `LAYA_ENDPOINT` env → `http://127.0.0.1:11448`.

Exit codes: `laya health` → 0=ok, 1=degraded, 2=unreachable. `laya status` → 0=healthy,
1=degraded (model unloaded), 2=unreachable/unknown, 3=stopped.

---

## MCP server

A JS MCP server (`~/.qwen/mcp-servers/laya.js`) exposes the HTTP API as agent tools (direct HTTP,
no CLI spawning). Tools: `laya_system_one`, `laya_health`, `laya_stats`, `laya_unload`.
Does NOT manage service lifecycle — use the `laya` CLI for start/stop/restart.

---

## HTTP API

Base URL: `http://<host>:11448`

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/system_one` | Run Laya: `{state, questions}` → calibrated answers |
| `GET` | `/health` | Liveness + active providers + uptime |
| `GET` | `/v1/stats` | Request/latency/cache/provider stats |

### `POST /system_one`

Request:
```json
{
  "state": { "subject": "Refund not received", "body": "..." },
  "questions": {
    "department": { "type": "choice", "instructions": "Which team should handle this?",
                    "criteria": { "billing": "payments", "support": "help" } },
    "urgency":    { "type": "score", "instructions": "How urgent?",
                    "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"] },
    "churn_risk": { "type": "noul", "instructions": "Likely to cancel?" }
  }
}
```

Response (mirrors Jev's `systemOne` / the TS library exactly):
```json
{
  "model": "laya",
  "answers": {
    "department": { "type": "choice", "choice": "billing",
                    "probabilities": { "billing": 0.9417, "support": 0.0309 },
                    "confidence": 0.7609, "rl_agent": { "act_probability": 1.0 } },
    "urgency":    { "type": "score", "score": 1.3879, "legend": { "0": "not urgent", ... },
                    "probabilities": { "0": 0.1755, ... }, "confidence": 0.1865,
                    "rl_agent": { "act_probability": 1.0 } },
    "churn_risk": { "type": "noul", "noul": 0.0984, "rl_agent": { "act_probability": 1.0 } }
  },
  "usage": { "input_tokens": 267, "output_tokens": 0 }
}
```

Response headers: `X-Request-Id`, `X-Latency-Ms`. Errors use an OpenAI-style envelope:
`{"error": {"message": "...", "type": "...", "param": null, "code": null}}`.

**Question types:** `choice` (pick one), `score` (expected level on an ordered rubric), `noul`
(P(true) for a yes/no statement). Full type/shape reference in [SERVICE.md](SERVICE.md).

---

## Resource footprint (measured 2026-09-28, RTX 3090, GPU card 1)

| Resource | Value | Notes |
|----------|-------|-------|
| **VRAM (loaded, idle)** | **~2.3 GB** | Model resident while loaded; auto-unloads after 300 s idle (default), lazily reloads ~0.7 s on next request. |
| **System RAM (RSS)** | **~736 MB** | Python worker process (tokenizer + ONNX session host). |
| **GPU util (idle)** | ~0–2% (any higher util on the card is *other* tenants) | Laya itself is idle when not inferring. |
| **Single warm call** | **~15 ms** | 3-question call, warm. |
| **Batched (16 concurrent)** | ~15 ms/request (one pass) | Coalesced into a single forward pass. |

**Idle behaviour:** the model is pre-warmed at startup and held while busy. With the default
config (`idle.unload_after_s=300`) it **auto-unloads after 300 s idle** and **lazily reloads
(~0.7 s) on the next request**, so a left-running service doesn't hoard 2.3 GB VRAM. Set
`unload_after_s=0` / `idle.enabled=false` to keep it permanently resident (avoids the ~0.7 s
per-request cold-start after an unload). While loaded and idle the cost is ~2.3 GB VRAM +
~736 MB RAM + ~0% GPU util. `/health` shows `model_loaded` and `status: "degraded"` while
unloaded — that's normal after idle, not a fault.

---

## Documentation set

| File | What it is |
|------|-----------|
| [README.md](README.md) | This file — entry point, quick start, API, resources. |
| [LINK.md](LINK.md) | Deploy paths, port, env, systemd, CUDA-13 gotcha, ops. The canonical how-it-runs. |
| [SERVICE.md](SERVICE.md) | Architecture, components, HTTP contract, data flow, tuning knobs. |
| [CHANGELOG.md](CHANGELOG.md) | Dated change log. |
| [ROADMAP.md](ROADMAP.md) | Planned work (CLI, client libs, primitives, deploy). |
| [../laya/docs/LAYA-GPU-SERVICE.md](../laya/docs/LAYA-GPU-SERVICE.md) | The original research + design rationale (in the library repo). |

---

## Status (2026-09-28)

- ✅ Built & verified: parity (decision-identical), CUDA active, ~20× speedup, batching, caching.
- ✅ Docs written.
- ✅ **Installed as a system systemd unit** (`/etc/systemd/system/laya-service.service` + timer;
      active on bobai). Drop-in override at `laya-service.service.d/override.conf` re-pins
      `CUDA_VISIBLE_DEVICES=1`.
- ✅ `laya` CLI in venv (`.venv/bin/laya`), sole owner of systemd lifecycle.
- ✅ MCP server (`~/.qwen/mcp-servers/laya.js`, registered in `~/.qwen/settings.json`).
- ✅ Repo pushed: `https://gitea.bobai.com.au/foxadmin/laya-service.git` (`master @ b59eced`,
      clean tree).
- ⬜ Interface primitives (typed `laya_choice`/`laya_score`/`laya_noul` wrappers, batch
      primitive) — see ROADMAP.md Phase 3.
- ⬜ Remote deployment / ops (alerting, log rotation, cache-DB disk) — ROADMAP.md Phase 4.
