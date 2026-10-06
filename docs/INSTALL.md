# INSTALL — Laya GPU Inference Service

Copy-paste-runnable install + deploy + verify instructions for a **fresh machine** (or a clean
re-install). Every path and value below is concrete and verified as of **2026-09-28**. There are
no placeholders. If a step fails, the troubleshooting section maps the real error to its cause.

> **On bobai the service is already installed and active** (as of 2026-09-28): system unit at
> `/etc/systemd/system/laya-service.service` + timer, with a drop-in override at
> `laya-service.service.d/override.conf`. This doc is the reference for a fresh deploy; on bobai
> you normally only re-run the verify steps (Steps 3–4) after a change.

This doc covers **deploying the systemd service** (production). For the dev
foreground loop, the test suite, and the CUDA library gotcha in detail, see
`QWEN.md`. For the HTTP API contract and architecture, see `SERVICE.md`.

---

## What you're installing

| Component | Value (verified 2026-09-28) |
|-----------|------------------------------|
| Repo / service root | `/MASTERFOLDER/Services/laya-service/` |
| Python package (editable install) | `/MASTERFOLDER/Services/laya-service/src/laya_server/` |
| Virtual environment | `/MASTERFOLDER/Services/laya-service/.venv` |
| Server console command | `/MASTERFOLDER/Services/laya-service/.venv/bin/laya-server` |
| CLI console command | `/MASTERFOLDER/Services/laya-service/.venv/bin/laya` |
| Config file (read at startup) | `/MASTERFOLDER/Services/laya-service/config/laya_config.json` |
| Model bundle (read-only, not in this repo) | `/MASTERFOLDER/Services/laya/onnx/` |
| Result cache DB (created at runtime) | `~/.laya-service/result_cache.db` |
| systemd unit (source) | `/MASTERFOLDER/Services/laya-service/systemd/laya-service.service` |
| systemd timer (source) | `/MASTERFOLDER/Services/laya-service/systemd/laya-service-restart.timer` |
| HTTP port | `11448` |
| GPU card | `1` (pinned; card 0 is saturated by vLLM as of 2026-09-28) |
| CUDA 13 library stack | `/usr/local/lib/ollama/cuda_v13` |

---

## Prerequisites (verify before installing)

Run these first. Each one failing means a later step will fail for a reason this
table makes obvious.

```bash
# 1. Repo is present and has the source tree + venv
ls -l /MASTERFOLDER/Services/laya-service/src/laya_server/server.py \
      /MASTERFOLDER/Services/laya-service/config/laya_config.json \
      /MASTERFOLDER/Services/laya-service/systemd/laya-service.service \
      /MASTERFOLDER/Services/laya-service/systemd/laya-service-restart.timer \
      /MASTERFOLDER/Services/laya-service/.venv/bin/laya-server

# 2. Model bundle exists (the service reads it read-only; it lives in the
#    separate `laya` repo, not this one)
ls -l /MASTERFOLDER/Services/laya/onnx/laya.onnx \
      /MASTERFOLDER/Services/laya/onnx/laya_config.json \
      /MASTERFOLDER/Services/laya/onnx/tokenizer/tokenizer.json

# 3. CUDA 13 stack exists (onnxruntime-gpu 1.30 links against it)
ls -d /usr/local/lib/ollama/cuda_v13
ls /usr/local/lib/ollama/cuda_v13/libcublasLt.so.13

# 4. Port 11448 is free (a foreground instance from a previous run will block
#    the unit from binding)
ss -tlnp | grep 11448 || echo "port 11448 free"

# 5. Confirm which GPU card is free (the unit pins card 1; if vLLM moved to
#    card 1 instead, you'll need to flip the pin — see Troubleshooting)
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

> Why the model bundle isn't in this repo: `/MASTERFOLDER/Services/laya/onnx/`
> is owned by the `laya` library repo and is downloaded there. This service only
> reads it. If you move the service to another machine, the bundle must already
> exist at the `model_dir` path in `config/laya_config.json` or be downloaded first.

---

## Step 1 — Install the package into the service venv

Editable install so the running service tracks the source tree. Run the exact
command; do not install globally.

```bash
cd /MASTERFOLDER/Services/laya-service
.venv/bin/python -m pip install -e .
```

Confirm the entry points exist afterward:

```bash
.venv/bin/laya-server --help >/dev/null && echo "laya-server OK"
.venv/bin/laya version
```

---

## Step 2 — Install and enable the systemd unit + restart timer

The unit is a **system** unit (not a `--user` unit). It needs root, so route
through `sudo-gui` (never plain `sudo` — the daemon has no password prompt).
The unit sets all required environment (CUDA-13 `LD_LIBRARY_PATH`,
`CUDA_VISIBLE_DEVICES=1`, thread caps), so nothing else is needed at start time.

```bash
sudo-gui cp /MASTERFOLDER/Services/laya-service/systemd/laya-service.service /etc/systemd/system/laya-service.service
sudo-gui cp /MASTERFOLDER/Services/laya-service/systemd/laya-service-restart.timer /etc/systemd/system/laya-service-restart.timer
sudo-gui systemctl daemon-reload
sudo-gui systemctl enable --now laya-service.service
sudo-gui systemctl enable --now laya-service-restart.timer
```

> Why a weekly restart timer: it reclaims fragmented CUDA memory (mirrors
> embed-server). The service itself is stable; the timer is a memory-hygiene
> measure, not a liveness mechanism.

> Why a single uvicorn worker: the unit does not set `workers=` (defaults to 1).
> Each worker loads its own ~2.3 GB ONNX session; multiple workers would
> duplicate the VRAM footprint. Intentional for a single-card service.

---

## Step 3 — Verify it's up and actually on CUDA

```bash
# Liveness + provider. `providers` MUST list "CUDAExecutionProvider" first.
# If it only shows "CPUExecutionProvider", the CUDA EP silently fell back to
# CPU — jump to Troubleshooting.
sudo-gui systemctl status laya-service.service
curl -s http://127.0.0.1:11448/health

# Stats: request counts, latency p50/p99, cache hits/misses, providers.
curl -s http://127.0.0.1:11448/v1/stats
```

---

## Step 4 — Prove the inference path end-to-end

A real `/system_one` round-trip. This is the same payload the parity test suite
uses (`input_tokens` should come back as `267`).

```bash
curl -s -X POST http://127.0.0.1:11448/system_one \
  -H 'Content-Type: application/json' \
  -d '{"state":{"subject":"Refund not received","body":"I cancelled my subscription two weeks ago and I still have not received my refund. This is the third time I am writing. If this is not resolved I will dispute the charge with my bank."},"questions":{"department":{"type":"choice","instructions":"Which team should handle this ticket?","criteria":{"billing":"payments, refunds, invoices","support":"product help and bugs","sales":"new purchases and upgrades"}},"urgency":{"type":"score","instructions":"How urgent is this ticket?","criteria":["not urgent","somewhat urgent","urgent","critical"]},"churn_risk":{"type":"noul","instructions":"Is the customer likely to cancel or dispute?"}}}'
```

Expected shape (values vary by model version; this sample is illustrative as of
2026-09-28): `department` → `billing`, `urgency` → a 0–3 score, `churn_risk` → a
`noul` probability, `usage.input_tokens: 267`.

### Optional — prove via the MCP tool

If the Qwen Code `laya` MCP server is registered (it lives at
`~/.qwen/mcp-servers/laya.js`, registered in `~/.qwen/settings.json`), a fresh
session can drive it:

```
mcp__laya__laya_health        # GET  /health
mcp__laya__laya_stats         # GET  /v1/stats
mcp__laya__laya_system_one    # POST /system_one
mcp__laya__laya_unload        # POST /control/unload
```

The MCP does not manage service lifecycle — for start/stop/restart/status use
the `laya` CLI (Step 5).

---

## Step 5 — Day-to-day operations via the `laya` CLI

The CLI is the sole owner of systemd lifecycle and the human/script interface.

```bash
.venv/bin/laya status            # combined systemd + health; exit 0/1/2/3
.venv/bin/laya health            # exit 0=ok, 1=degraded, 2=unreachable
.venv/bin/laya stats
.venv/bin/laya query --state '{"subject":"Refund not received"}' \
                    --questions '{"urgency":{"type":"score","instructions":"?"}}'
.venv/bin/laya unload            # free ~2.4 GB VRAM; lazily reloads on next query
.venv/bin/laya restart          # systemctl start/stop/restart on the system unit
```

> `start`/`stop`/`restart` manage the **system** unit and require root (route
> through `sudo-gui`). `status` is readable by any user. The CLI calls plain
> `systemctl` (no `--user`), matching the system unit installed in Step 2.

> Endpoint resolution for the CLI: `--endpoint` flag → `LAYA_ENDPOINT` env →
> `http://127.0.0.1:11448`.

### Run the test suite (parity gate + contracts)

The parity gate confirms decisions are identical to the CPU reference. Run it
from the service directory with the CUDA env set:

```bash
cd /MASTERFOLDER/Services/laya-service
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m pytest tests/ -v
```

### Benchmark

```bash
cd /MASTERFOLDER/Services/laya-service
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python bench.py
```

---

## Uninstall / stop

```bash
# Stop the service (frees the card-1 VRAM). The timer is unaffected.
sudo-gui systemctl stop laya-service.service

# Fully remove the deployment (unit + timer + their state)
sudo-gui systemctl disable --now laya-service.service
sudo-gui systemctl disable --now laya-service-restart.timer
sudo-gui rm /etc/systemd/system/laya-service.service /etc/systemd/system/laya-service-restart.timer
sudo-gui systemctl daemon-reload
```

To also clear the result cache: stop the service, then
`rm ~/.laya-service/result_cache.db` (or set `"cache": {"enabled": false}` in
`config/laya_config.json`).

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `/health` → `providers` shows only `CPUExecutionProvider` | CUDA EP fell back to CPU. Read the actual log first: `sudo-gui journalctl -u laya-service.service -n 40`. Two causes seen: (a) `LD_LIBRARY_PATH` missing the CUDA-13 stack — but it's baked into the unit, so check `Environment="LD_LIBRARY_PATH=..."` in `/etc/systemd/system/laya-service.service` survived the `cp`; (b) the pinned card isn't actually visible to the process (see `GPU=-1` below). |
| Startup log shows `EP Error ... CUDA failure 100: no CUDA-capable device is detected ; GPU=-1` | ORT asked for a CUDA device index that doesn't exist in the driver's view. Diagnose with these exact checks (run them before theorizing): <br><br>`nvidia-smi --query-gpu=index,pci.bus_id,memory.total --format=csv,noheader` — how many cards does the kernel/`nvidia-smi` see?<br>`LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu .venv/bin/python -c "import ctypes;cu=ctypes.CDLL('libcuda.so.1');cu.cuInit(0);n=ctypes.c_int();cu.cuDeviceGetCount(ctypes.byref(n));print('visible:',n.value)"` — how many does the CUDA driver expose to the process?<br><br>If the driver count is **fewer than** the `nvidia-smi` count (observed on bobai 2026-09-28: driver sees 1, `nvidia-smi` sees 2), the process and `nvidia-smi` are in different device namespaces — a system-level CUDA/driver mismatch (stale `libcuda` vs kernel module, or a MIG/passthrough remap), **not** a bug in this service. Setting `CUDA_VISIBLE_DEVICES` to an index the driver doesn't expose (e.g. `1` when only `0` is visible) produces exactly this error. **Fix is at the OS/CUDA-driver layer, outside this repo** — get the driver and runtime to agree on the same device set, then set the unit's `CUDA_VISIBLE_DEVICES` to an index that exists in the driver's view. |
| Unit won't start; port conflict | Something is already bound to 11448. Find it: `ss -tlnp \| grep 11448`, then stop that process (usually a leftover foreground `laya-server`) and retry `sudo-gui systemctl start laya-service.service`. |
| `Model not loaded` / 500 on `/system_one` right after start | The model load failed at startup (the service degrades rather than crash-loops). Check `sudo-gui journalctl -u laya-service.service -n 60` for the `Failed to initialise model at startup` line. Usually a missing model bundle at `model_dir` or the card not being reachable. |
| OOM at startup on card 1 | Card 1 filled up since you checked (vLLM moved, or another tenant). Re-check `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader` and, if card 1 is now full, either wait for it to free or flip the pin back to card 0 (edit `CUDA_VISIBLE_DEVICES` in the installed unit, then `sudo-gui systemctl daemon-reload && sudo-gui systemctl restart laya-service.service`). |

### Parity note (do not "fix" small FP drift)

GPU vs CPU probabilities differ by ~1–4e-4 (up to ~6–7e-4 on the nonlinear
`confidence`/`score` aggregates). That is standard CPU-vs-CUDA FP32
nondeterminism — the **decision is invariant**. Acceptance bar: `input_tokens`
exact (267 on the fixture), same `choice`/argmax, numerics within `1e-3` abs.
Do not assert exact 4-decimal equality between CPU and GPU; it isn't achievable
and isn't the goal.

---

## After changing the model, ORT, or config

Re-verify before relying on it:

```bash
cd /MASTERFOLDER/Services/laya-service
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m pytest tests/ -v
curl -s http://127.0.0.1:11448/health   # confirm CUDAExecutionProvider first
```

---

## See also

- `QWEN.md` — full ops reference: paths, environment, the CUDA-13 gotcha in
  depth, resource footprint, deploy notes.
- `SERVICE.md` — components, API contract, data flow, tuning.
- `README.md` — overview + quick start.
- `ROADMAP.md` — planned work.
- `CHANGELOG.md` — dated change history.
