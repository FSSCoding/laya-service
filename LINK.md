# Laya GPU Inference Service — LINK

Canonical how-it-runs document: deploy paths, port, environment, CUDA-13 gotcha, systemd, and ops.
Read this first when operating or modifying the service on bobai.

---

## Identity

| Field | Value |
|-------|-------|
| Service | `laya-service` |
| Host | **bobai** (10.23.88.5) — for now; may move to another FSS server later |
| Port | **11448** (user-assigned 2026-09-28; verified free) |
| Interface | HTTP only (no Unix socket — not needed yet) |
| Model | Laya 421M-param ONNX (English), from `/MASTERFOLDER/Services/laya/onnx/` |
| Runtime | Python 3.12 + `onnxruntime-gpu==1.30.0` (CUDA EP) |
| GPU | One RTX 3090, pinned via `CUDA_VISIBLE_DEVICES=1` (card 1; card 0 hosts other tenants) |
| Venv | **Dedicated**: `/MASTERFOLDER/Services/laya-service/.venv` |

---

## Paths

| Thing | Path |
|-------|------|
| Service root | `/MASTERFOLDER/Services/laya-service/` |
| Python package | `/MASTERFOLDER/Services/laya-service/src/laya_server/` (`server.py`, `_config.py`, `worker.py`, `sequence.py`, `batcher.py`, `result_cache.py`) |
| Server entry point | `/MASTERFOLDER/Services/laya-service/.venv/bin/laya-server` (`laya_server.server:run_server`) |
| Config | `/MASTERFOLDER/Services/laya-service/config/laya_config.json` |
| Venv | `/MASTERFOLDER/Services/laya-service/.venv` |
| Model bundle (read-only) | `/MASTERFOLDER/Services/laya/onnx/` (`laya.onnx`, `laya.onnx.data`, `laya_config.json`, `tokenizer/`) |
| Result cache DB | `~/.laya-service/result_cache.db` (SQLite; created at runtime) |
| systemd unit (source) | `/MASTERFOLDER/Services/laya-service/systemd/laya-service.service` |
| systemd timer (source) | `/MASTERFOLDER/Services/laya-service/systemd/laya-service-restart.timer` |
| systemd unit (**installed**) | `/etc/systemd/system/laya-service.service` (system unit, `User=bob`) |
| systemd timer (**installed**) | `/etc/systemd/system/laya-service-restart.timer` |
| systemd drop-in override | `/etc/systemd/system/laya-service.service.d/override.conf` (re-pins `CUDA_VISIBLE_DEVICES=1`) |
| CLI entry point | `/MASTERFOLDER/Services/laya-service/.venv/bin/laya` (`laya_server.cli:cli`) |
| MCP server | `~/.qwen/mcp-servers/laya.js` (registered in `~/.qwen/settings.json`) |
| Gitea remote | `https://gitea.bobai.com.au/foxadmin/laya-service.git` |
| Tests | `/MASTERFOLDER/Services/laya-service/tests/` (pytest) |
| Benchmark | `/MASTERFOLDER/Services/laya-service/bench.py` |
| Research/design doc | `/MASTERFOLDER/Services/laya/docs/LAYA-GPU-SERVICE.md` |

> The model bundle is **not** owned by this repo — it lives in the `laya` library repo and is
> downloaded there. This service reads it read-only. If you move the service to another machine,
> the bundle must exist at `model_dir` (config) or be downloaded first.

---

## Environment

### The CUDA 13 library path (CRITICAL)

`onnxruntime-gpu==1.30.0` is built against **CUDA 13.0** (ORT ≥1.27 defaults to CUDA 13). Its
cuBLAS / cuBLAS-Lt **13** libraries are **not** on the default `LD_LIBRARY_PATH`. On bobai they live
in Ollama's CUDA-13 stack:

- cuBLAS/cuBLAS-Lt 13 + cuDART 13: `/usr/local/lib/ollama/cuda_v13/`
- cuDNN 9.25: `/usr/lib/x86_64-linux-gnu`

**Without this, the CUDA EP silently falls back to CPU** with:
```
Failed to load library .../libonnxruntime_providers_cuda.so
  libcublasLt.so.13: cannot open shared object file: No such file or directory
```

So every invocation (and the systemd unit) sets:
```
LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu
```

To verify CUDA is actually active after a change: `GET /health` → `providers` must list
`"CUDAExecutionProvider"` first. If it only shows `CPUExecutionProvider`, the lib path is wrong.

> If a future ORT upgrade moves to a CUDA 12.x build, point `LD_LIBRARY_PATH` at the matching cuBLAS
> 12 path instead (`ldconfig -p | grep cublasLt`). Don't guess.

### Other env (set by the systemd unit)

| Var | Value | Why |
|-----|-------|-----|
| `CUDA_VISIBLE_DEVICES` | `1` | Pin to one 3090. Card 1 is the pin as of 2026-09-28 (card 0 hosts Fss-RAG, fss-tts, etc.); switch to `0` if card 1 gets saturated. |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `2` | Prevent thread oversubscription (model is small; 2 threads is plenty). |
| `TOKENIZERS_PARALLELISM` | `false` | Avoid tokenizers warning + redundant threads. |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Inherited from embed-server; harmless (ORT doesn't use torch) — keep for consistency. |

### Application config resolution

The server reads **one** env var: `LAYA_CONFIG_PATH`. All other configuration comes from the JSON
config file. Resolution order (strict — no silent fallback):

1. `LAYA_CONFIG_PATH` env var, if set
2. `config/laya_config.json` relative to CWD, if it exists
3. Raise `FileNotFoundError`

The systemd unit sets `WorkingDirectory=/MASTERFOLDER/Services/laya-service`, so the CWD-relative
resolution finds `config/laya_config.json` without needing `LAYA_CONFIG_PATH`.

Do NOT set env vars like `LAYA_IDLE_UNLOAD_S` or `LAYA_PORT` — the server does not read them.
All runtime tunables are in the JSON config file.

---

## Running

### Foreground (dev / testing)

```bash
cd /MASTERFOLDER/Services/laya-service
.venv/bin/python -m pip install -e .

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/laya-server
```

Or via `python -m`:

```bash
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m laya_server
```

### systemd (production)

**Installed & active on bobai (2026-09-28).** The unit is a **system** unit at
`/etc/systemd/system/laya-service.service` (`User=bob`), with the weekly restart timer at
`/etc/systemd/system/laya-service-restart.timer` and a drop-in override at
`/etc/systemd/system/laya-service.service.d/override.conf` (re-pins `CUDA_VISIBLE_DEVICES=1`).
Journal confirms `systemd[1]` owns the process. Verify with `sudo-gui systemctl status
laya-service.service` and `curl -s http://127.0.0.1:11448/health`.

The unit runs the installed `laya-server` console command. The package is editable-installed in
the service venv (`.venv/bin/python -m pip install -e .`), so the running service tracks the
source tree.

**Re-install / update from source** (needs sudo — route through `sudo-gui`):

```bash
sudo-gui cp /MASTERFOLDER/Services/laya-service/systemd/laya-service.service /etc/systemd/system/laya-service.service
sudo-gui cp /MASTERFOLDER/Services/laya-service/systemd/laya-service-restart.timer /etc/systemd/system/laya-service-restart.timer
sudo-gui systemctl daemon-reload
sudo-gui systemctl enable --now laya-service.service
sudo-gui systemctl enable --now laya-service-restart.timer
```

> **Source ↔ installed drift (2026-09-28):** the installed `.service` differs from the repo source
> only in two comment lines (installed cites "vLLM" as the card-0 tenant; source says "other GPU
> tenants"). No behavioral difference. The effective runtime is the unit **plus** the drop-in
> override. If you re-`cp` the source, the override still applies (drop-ins survive unit
> replacement).

Resource caps in the unit: `MemoryMax=8G`, `CPUQuota=300%`, `Nice=-5`, `Restart=on-failure`,
`RestartSec=5`. The weekly timer reclaims fragmented CUDA memory (mirrors embed-server).

**Single uvicorn worker.** The unit does not set `workers=` (defaults to 1). Each worker process
loads its own ONNX session (~2.3 GB VRAM). Running multiple workers would duplicate the VRAM
footprint. This is intentional for a single-card GPU service.

> **Port conflict:** if a foreground instance is already bound to 11448, stop it before enabling the
> unit (two processes can't share the port). Check with `ss -tlnp | grep 11448`.

---

## Verification

```bash
# liveness + provider (must show CUDAExecutionProvider first)
curl -s http://127.0.0.1:11448/health

# stats (cache hits/misses, latency p50/p99, providers, uptime)
curl -s http://127.0.0.1:11448/v1/stats

# parity gate (decision-identical to CPU reference; run from the service dir)
cd /MASTERFOLDER/Services/laya-service
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m pytest tests/ -v

# benchmark
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:/usr/lib/x86_64-linux-gnu \
  .venv/bin/python bench.py
```

### Parity bar (correct across backends)

GPU vs CPU differ by ~1–4e-4 per probability (and ~6–7e-4 on the nonlinear `confidence`/`score`
aggregates) — standard CPU-vs-CUDA FP32 non-determinism. The **decision is invariant**. Acceptance:
`input_tokens` exact (267 on the fixture), same `choice`/argmax, numerics within `1e-3` abs.
Do NOT assert exact 4-decimal equality between CPU and GPU — that's not achievable and isn't the goal.

---

## Ops notes

- **Model lifecycle:** the model is pre-warmed at startup and held while busy. With the current
  config (`idle.unload_after_s=300`, `idle.enabled=true`) it **auto-unloads after 300 s idle** and
  **lazily reloads (~0.7 s) on the next `/system_one`**. Set `unload_after_s=0` /
  `idle.enabled=false` to keep it permanently resident. `/health` reports `model_loaded` and
  `status: "degraded"` while unloaded (normal after idle — not a fault). Idle cost while loaded is
  ~2.3 GB VRAM + ~736 MB RAM + ~0% GPU util.
- **Cache** is keyed on `sha256(state, questions)` — deterministic, so repeat requests are served
  from memory (~6 ms) instead of the GPU. Cache lives at `~/.laya-service/result_cache.db` + in-memory
  LRU. To clear: stop the service and delete the DB (or set `cache.enabled=false` in config).
- **Pinned to card 1** (`CUDA_VISIBLE_DEVICES=1`). If card 1 gets saturated by another tenant,
  switch the pin to `0` (edit `CUDA_VISIBLE_DEVICES` in the unit source and the drop-in, then
  `sudo-gui systemctl daemon-reload && sudo-gui systemctl restart laya-service.service`) and
  update the docs to match.
- **Re-verify after any ORT/model change**: run `pytest tests/` and check `/health` providers.
- This service is **read-only** with respect to the `laya` repo (it reads the bundle, never writes).
- **Config validation**: the server validates the config file at startup (port range, positive TTL,
  model_dir existence). A missing or invalid config fails at startup with a clear error message.
