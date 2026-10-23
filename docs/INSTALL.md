# Install — Laya GPU Inference Service

Fresh-machine install + deploy + verify for **Linux**. Every value below is concrete and
portable; machine-specific things (your GPU card, your CUDA library path, your install
directory) are the only things you fill in. For the foreground/dev loop and the API contract,
see [README.md](../README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

---

## What you're installing

| Component | Where it lives |
|-----------|----------------|
| Repo / service root | your checkout of this repository (`$LAYA_HOME`) |
| Python package (editable install) | `$LAYA_HOME/src/laya_server/` |
| Virtual environment | `$LAYA_HOME/.venv` |
| Server console command | `$LAYA_HOME/.venv/bin/laya-server` |
| CLI console command | `$LAYA_HOME/.venv/bin/laya` |
| Config file (read at startup) | `$LAYA_HOME/config/laya_config.json` |
| Model bundle (read-only, **not in this repo**) | whatever `model_dir` points at |
| Result cache DB (created at runtime) | `~/.laya-service/result_cache.db` |
| systemd unit template (main service) | `$LAYA_HOME/systemd/laya-service.service.template` |
| systemd restart companion (oneshot) | `$LAYA_HOME/systemd/laya-service-restart.service` |
| systemd weekly restart timer | `$LAYA_HOME/systemd/laya-service-restart.timer` |
| HTTP port | `11448` (change in config if needed) |

---

## Prerequisites (verify before installing)

Each one failing here means a later step fails for an obvious reason.

```bash
# 1. Python >= 3.12
python3 --version

# 2. Source tree present
ls -l "$LAYA_HOME/src/laya_server/server.py" \
      "$LAYA_HOME/config/laya_config.example.json" \
      "$LAYA_HOME/systemd/laya-service.service.template"

# 3. Model bundle exists (the service reads it read-only; it is NOT in this repo)
ls -l "$MODEL_DIR/laya.onnx" "$MODEL_DIR/laya_config.json" "$MODEL_DIR/tokenizer/"

# 4. A CUDA 13 cuBLAS/cuBLAS-Lt stack exists (onnxruntime-gpu 1.30 links against it)
ldconfig -p | grep -i cublasLt

# 5. Port 11448 is free
ss -tlnp | grep 11448 || echo "port 11448 free"

# 6. Which GPU card is free (pick a card index for CUDA_VISIBLE_DEVICES)
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

> **Why the model bundle isn't in this repo:** the weights are ~1.7 GB and are licensed
> separately (Apache-2.0, Convai Innovations). Download the **pre-exported ONNX bundle** from
> <https://huggingface.co/receptron/laya-onnx> and point `model_dir` at it.
>
> Do **not** use <https://huggingface.co/convaiinnovations/laya> — that is the source
> PyTorch/safetensors checkpoint (what the ONNX was exported from), which this service does not
> load. The service requires `laya.onnx`.

---

## Step 1 — Install the package into a venv

```bash
cd "$LAYA_HOME"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Confirm the entry points:

```bash
laya-server --help >/dev/null && echo "laya-server OK"
laya version
```

---

## Step 2 — Configure

```bash
cp config/laya_config.example.json config/laya_config.json
$EDITOR config/laya_config.json
```

Set, at minimum:

- `model_dir` → absolute path to the ONNX bundle directory (**required**; the service refuses
  to start without it).
- `gpu.visible_devices` → the card index you chose in the prerequisites.
- `port` → `11448` unless something else is using it.
- `host` → `127.0.0.1` for local-only, `0.0.0.0` to expose on the LAN (then protect it
  yourself; there is no auth).

For CPU-only, remove the `CUDAExecutionProvider` entry from `gpu.providers`.

---

## Step 3 — Run (foreground) and verify CUDA

```bash
# Make the CUDA 13 libs discoverable, pin your card, run:
CUDA_VISIBLE_DEVICES=<card> LD_LIBRARY_PATH=<cublas-13-dir>:/usr/lib/x86_64-linux-gnu laya-server
```

From another shell:

```bash
# `providers` MUST list "CUDAExecutionProvider" first for GPU to be active.
curl -s http://127.0.0.1:11448/health
curl -s http://127.0.0.1:11448/v1/stats
```

### Step 3b — Prove the inference path end-to-end

```bash
curl -s -X POST http://127.0.0.1:11448/system_one -H 'Content-Type: application/json' -d '{
  "state": {"subject": "Refund not received", "body": "I ordered last week and never got my money back."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "payments", "support": "help"}},
    "urgency":    {"type": "score", "instructions": "How urgent is this?",
                   "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
    "churn_risk": {"type": "noul", "instructions": "Likely to cancel?"}
  }
}'
```

Expected shape: `department` → a choice + probabilities, `urgency` → a 0–3 score + legend,
`churn_risk` → a `noul` probability, `usage.input_tokens` a small positive integer.

---

## Step 4 — (Optional) Install as a systemd service

Two units make up the service: the main service (`laya-service.service`) and a weekly
restart companion that reclaims fragmented CUDA memory
(`laya-service-restart.service`, triggered by `laya-service-restart.timer`). Install both.

Edit the template's five `__CONFIGURATION__` points, then install all three unit files:

```bash
sed -e "s|__SERVICE_USER__|$SUDO_USER|g" \
    -e "s|__INSTALL_DIR__|$LAYA_HOME|g" \
    -e "s|__GPU_CARD__|<card>|g" \
    -e "s|__CUDA_LIB_PATH__|<cublas-13-dir>|g" \
    systemd/laya-service.service.template > /etc/systemd/system/laya-service.service

sudo cp systemd/laya-service-restart.service /etc/systemd/system/
sudo cp systemd/laya-service-restart.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now laya-service.service
sudo systemctl enable --now laya-service-restart.timer
```

> **The timer needs its companion service.** `laya-service-restart.timer` activates
> `laya-service-restart.service` by the systemd same-basename convention (a timer with no
> explicit target defaults to the service with the matching name). Installing the timer
> without the companion leaves a timer that fires at the weekly slot with no service to
> activate — the reclaim silently never happens. Install the service alongside the timer.
>
> **Why a weekly restart timer:** it reclaims fragmented CUDA memory. The service itself is
> stable; the timer is a memory-hygiene measure, not a liveness mechanism.
>
> **Single uvicorn worker:** the unit does not set `workers=` (defaults to 1). Each worker loads
> its own ~2.3 GB ONNX session; multiple workers would duplicate the VRAM footprint. Intentional
> for a single-card service.

---

## Step 5 — Day-to-day operations

```bash
laya status            # combined systemd + health; exit 0/1/2/3
laya health            # exit 0=ok, 1=degraded, 2=unreachable
laya stats
laya choice  --state '{"subject":"Refund"}' --instructions "Which team?" \
             --options '{"billing":"payments","support":"help"}' --json
laya noul    --state '{"subject":"Refund"}' --statement "Likely to cancel?" --json
laya unload  # free VRAM; lazily reloads on next query
laya restart # systemctl on the system unit (needs root)
```

> `start`/`stop`/`restart` manage the **system** unit and need root. `status` is readable by any
> user. CLI endpoint resolution: `--endpoint` flag → `LAYA_ENDPOINT` env → `http://127.0.0.1:11448`.

### Test suite (parity gate + contracts)

```bash
pip install -e ".[dev]"
pytest                                  # 94 passed (parity skips without a GPU + model)
LAYA_MODEL_DIR="$MODEL_DIR" \
  CUDA_VISIBLE_DEVICES=<card> \
  LD_LIBRARY_PATH=<cublas-13-dir>:/usr/lib/x86_64-linux-gnu \
  pytest -v                              # includes the GPU/CPU parity gate
```

### Benchmark

```bash
LAYA_MODEL_DIR="$MODEL_DIR" \
  CUDA_VISIBLE_DEVICES=<card> \
  LD_LIBRARY_PATH=<cublas-13-dir>:/usr/lib/x86_64-linux-gnu \
  python bench.py
```

---

## Uninstall / stop

Remove the main service, the restart companion, and the timer — all three are installed:

```bash
sudo systemctl stop laya-service.service
sudo systemctl disable --now laya-service.service
sudo systemctl disable --now laya-service-restart.timer
sudo rm /etc/systemd/system/laya-service.service \
       /etc/systemd/system/laya-service-restart.service \
       /etc/systemd/system/laya-service-restart.timer
sudo systemctl daemon-reload
```

> If you added a drop-in override (e.g. to re-pin the GPU card), remove its directory too:
> `sudo rm -r /etc/systemd/system/laya-service.service.d`.

To clear the result cache: stop the service, then `rm ~/.laya-service/result_cache.db` (or set
`"cache": {"enabled": false}` in the config).

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `/health` → `providers` shows only `CPUExecutionProvider` | CUDA EP fell back to CPU. Read the real log first: `journalctl -u laya-service.service -n 40`. Usual cause: `LD_LIBRARY_PATH` is missing the CUDA-13 `libcublasLt.so.13`. Find it with `ldconfig -p \| grep -i cublasLt` and add its directory. |
| Startup log: `CUDA failure 100: no CUDA-capable device ... GPU=-1` | ORT asked for a CUDA index that doesn't exist in the driver's view. Check `nvidia-smi --query-gpu=index,pci.bus_id,memory.total --format=csv,noheader` and set `CUDA_VISIBLE_DEVICES` to an index the driver actually exposes. If the driver sees fewer cards than `nvidia-smi`, that is an OS/CUDA-driver mismatch, not a bug in this service. |
| Unit won't start; port conflict | Something is already bound to the port. `ss -tlnp \| grep <port>`, stop the leftover process, retry. |
| `Model not loaded` / 500 on `/system_one` right after start | Model load failed at startup (the service degrades rather than crash-loops). Check `journalctl -u laya-service.service -n 60` for `Failed to initialise model at startup`. Usually a missing `model_dir` or an unreachable card. |
| OOM at startup | The pinned card filled up since you checked. Re-check `nvidia-smi` and pick a freer card. |

### Parity note (do not "fix" small FP drift)

GPU vs CPU probabilities differ by ~1–4e-4 (up to ~6–7e-4 on the nonlinear `confidence`/`score`
aggregates). That is standard CPU-vs-CUDA FP32 nondeterminism — the **decision is invariant**.
Acceptance bar: `input_tokens` exact, same `choice`/argmax, numerics within `1e-3` abs. Do not
assert exact 4-decimal equality between CPU and GPU.

---

## After changing the model, ORT, or config

Re-verify before relying on it:

```bash
LAYA_MODEL_DIR="$MODEL_DIR" CUDA_VISIBLE_DEVICES=<card> \
  LD_LIBRARY_PATH=<cublas-13-dir>:/usr/lib/x86_64-linux-gnu pytest -v
curl -s http://127.0.0.1:11448/health   # confirm CUDAExecutionProvider first
```

---

## See also

- [QWEN.md](../QWEN.md) — full ops reference: architecture, environment, the CUDA-13 gotcha, resource footprint.
- [ARCHITECTURE.md](ARCHITECTURE.md) — components, API contract, data flow, tuning.
- [README.md](../README.md) — overview + quick start.
