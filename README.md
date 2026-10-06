# Laya GPU Inference Service

**GPU-accelerated Laya inference over HTTP.** A 421M-parameter System-1 *decision* model
(`onnxruntime-gpu`) served by a lean FastAPI app. Laya does not generate text: you give it a
*state* (a ticket, an email, a JSON object) and a set of typed questions, and it returns
calibrated probabilities for every answer in a single forward pass.

This repository is the HTTP service around that model — resident model loading, request
coalescing, a two-tier result cache, and a small CLI + playground. It is built by
[Fox Software Solutions](https://foxsoftwaresolutions.com.au).

![Fox Software Solutions](playground/static/fox-logo.png)

---

## What Laya is (and isn't)

Laya is a **bounded semantic decision tool**, not a chatbot and not a factual oracle. It answers
three kinds of question about a given state:

| Type | Meaning | Answer shape |
|------|---------|--------------|
| `choice` | pick one option from a set | argmax option + per-option probabilities |
| `score` | expected level on an ordered rubric | expected level (0…n-1) + per-level probabilities |
| `noul` | P(true) for a yes/no statement | a single probability |

The model produces **calibrated probabilities**. Treat them as confidence signals to gate and
rank actions on, not as ground-truth facts. For the safe-usage doctrine and how to phrase good
questions, read [docs/MODEL-DOCTRINE.md](docs/MODEL-DOCTRINE.md).

---

## Install (Linux)

**Prerequisites**

- Python **≥ 3.12**
- An NVIDIA GPU with a recent driver **and** a CUDA 13 runtime (`onnxruntime-gpu ≥ 1.30` is
  built against CUDA 13). CPU-only also works — see [CPU-only](#cpu-only) below.
- The Laya ONNX model bundle (obtain separately — see [Getting the model](#getting-the-model)).

**Steps**

```bash
git clone <your-mirror-of-this-repo> laya-service
cd laya-service

python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # pulls onnxruntime-gpu, fastapi, uvicorn, tokenizers, ...
```

**Configure**

```bash
cp config/laya_config.example.json config/laya_config.json
$EDITOR config/laya_config.json    # set model_dir (and gpu.visible_devices / port as needed)
```

`model_dir` **must** point at a directory that contains the model bundle
(see [Getting the model](#getting-the-model)). The service refuses to start if it is unset or
points at a missing directory.

**Run**

```bash
# CUDA (make the cuBLAS/cuBLAS-Lt/cuDNN 13 libraries discoverable first — see below):
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/path/to/cublas-13-libs:/usr/lib/x86_64-linux-gnu laya-server

# Verify:
curl -s http://127.0.0.1:11448/health
# "providers" must list "CUDAExecutionProvider" first for GPU to be active.
```

> **The CUDA library path matters.** `onnxruntime-gpu ≥ 1.30` needs the CUDA 13
> `libcublasLt.so.13` on the loader path. If it is missing, the CUDA execution provider
> **silently falls back to CPU** with a `Failed to load library ... libcublasLt.so.13` warning.
> Find your copy with `ldconfig -p | grep -i cublasLt` and add its directory to
> `LD_LIBRARY_PATH`. Verify via `/health` → `providers` after every change.

To run it as a background service on Linux, see
[systemd/laya-service.service.template](systemd/laya-service.service.template) (edit the five
marked configuration points) and [docs/INSTALL.md](docs/INSTALL.md).

### CPU-only

No NVIDIA GPU? Remove the `CUDAExecutionProvider` entry from `gpu.providers` in your config
(or just don't set `LD_LIBRARY_PATH` and let it fall through). Everything else is identical;
expect CPU latency to be roughly an order of magnitude slower than a warm GPU call.

---

## Getting the model

The model weights are **not** included in this repository (they are several hundred MB and are
licensed separately). Download the official Laya ONNX bundle from Hugging Face:

- **Model card:** https://huggingface.co/convaiinnovations/laya
- **Weights licence:** Apache-2.0 (Convai Innovations)

`laya_config.json` expects `model_dir` to contain:

```
laya.onnx            graph
laya.onnx.data       fp32 weights (large)
laya_config.json     max_len, head_max_len, temperatures
tokenizer/           the checkpoint's tokenizer
```

Point `model_dir` at that directory. The service reads it read-only.

---

## Make one request

**HTTP**

```bash
curl -s http://127.0.0.1:11448/system_one -H 'Content-Type: application/json' -d '{
  "state": { "subject": "Refund not received", "body": "I ordered last week and never got my money back." },
  "questions": {
    "department": { "type": "choice", "instructions": "Which team should handle this?",
                    "criteria": { "billing": "payments", "support": "help" } },
    "urgency":    { "type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"] },
    "churn_risk": { "type": "noul", "instructions": "Likely to cancel their account?" }
  }
}'
```

**Response** (the response shape follows the Jev `system_one` API; see
[docs/research/JEV-COMPATIBILITY-SCOPE.md](docs/research/JEV-COMPATIBILITY-SCOPE.md)):

```json
{
  "model": "laya",
  "answers": {
    "department": { "type": "choice", "choice": "billing",
                    "probabilities": { "billing": 0.94, "support": 0.03 },
                    "confidence": 0.76, "rl_agent": { "act_probability": 1.0, "escalate_probability": 0.0, "experimental": true } },
    "urgency":    { "type": "score", "score": 1.39, "legend": { "0": "not urgent", "1": "somewhat urgent", "2": "urgent", "3": "critical" },
                    "probabilities": { "0": 0.18, "1": 0.55, "2": 0.24, "3": 0.03 },
                    "confidence": 0.19, "rl_agent": { "act_probability": 1.0, "escalate_probability": 0.0, "experimental": true } },
    "churn_risk": { "type": "noul", "noul": 0.10, "rl_agent": { "act_probability": 1.0, "escalate_probability": 0.0, "experimental": true } }
  },
  "usage": { "input_tokens": 267, "output_tokens": 0 }
}
```

Errors use an OpenAI-style envelope: `{"error": {"message": "...", "type": "...", "param": null, "code": null}}`.

**CLI** (once the server is running):

```bash
laya choice --state '{"subject":"Refund"}' --instructions "Which team?" \
            --options '{"billing":"payments","support":"help"}' --json
laya score  --state '{"subject":"Refund"}' --instructions "How urgent?" \
            --levels '["not urgent","somewhat","urgent","critical"]' --json
laya noul   --state '{"subject":"Refund"}' --statement "Likely to cancel?" --json
laya batch  --state '{"subject":"Refund"}' --questions '<json>' --json
laya health                      # exit 0=ok, 1=degraded, 2=unreachable
laya stats
laya unload                      # free VRAM; lazily reloads on next query
```

**Python primitives** (importable without running the server's own process):

```python
from laya_server import laya_choice, laya_score, laya_noul, laya_batch
from laya_server import classify, multi_select, rank

r = laya_choice(state, "Which team?", {"billing": "payments", "support": "help"})
# → ChoiceResult(choice=..., probabilities=..., confidence=..., routing_signal=...)
r = laya_noul(state, "Is the customer likely to cancel?")
# → NoulResult(noul=..., routing_signal=...)
r = rank(state, "Urgency", ["ticket-A", "ticket-B", "ticket-C"],
         rubric=["low", "med", "high"])   # [(item, ScoreResult), ...] sorted desc
```

### Routing signal (experimental)

Every answer carries an `rl_agent` envelope from the model's separate RL-trained head:
`{act_probability, escalate_probability, experimental: true}`. This is an **experimental**
routing hint — do **not** build production routing logic on it without validating calibration
for your domain. See the doctrine doc.

---

## HTTP API

Base URL: `http://<host>:11448`

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/system_one` | Run Laya: `{state, questions}` → calibrated answers |
| `GET` | `/health` | Liveness + active providers + model state + uptime |
| `GET` | `/v1/stats` | Request / latency / cache / provider statistics |
| `POST` | `/control/unload` | Unload the model from GPU memory |

Full contract, data flow, and tuning knobs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Playground

A local, loopback-only browser workbench that teaches the three primitives interactively
(probability visualisations, "break it" probes, explore-more references). No persistence.

```bash
python playground/server.py --port 8090
# → http://127.0.0.1:8090  (proxies /api/* to the service at http://127.0.0.1:11448 by default)
```

If your service listens on a different port, pass `--upstream http://127.0.0.1:<port>`
(loopback-only). See [playground/README.md](playground/README.md) and
[docs/playground/SPEC.md](docs/playground/SPEC.md).

---

## Resource footprint (measured, RTX 3090)

| Resource | Value | Notes |
|----------|-------|-------|
| VRAM (loaded, idle) | ~2.3 GB | Auto-unloads after 300 s idle by default; lazily reloads ~0.7 s on next request. |
| System RAM (RSS) | ~736 MB | Worker process (tokenizer + ONNX session host). |
| Single warm call | ~15 ms | 3-question call, warm, GPU. |
| Batched (16 concurrent) | ~15 ms/request | Coalesced into a single forward pass. |

The model auto-unloads after `idle.unload_after_s` (default 300 s) and reloads on the next
request, so a left-running service doesn't hoard VRAM. `/health` reports `model_loaded` and
`status: "degraded"` while unloaded — that is normal after idle, not a fault.

---

## Testing & benchmarks

```bash
pip install -e ".[dev]"
pytest                       # 94 passed (parity tests skip without a GPU + model)
LAYA_MODEL_DIR=/path/to/model pytest -v   # includes the GPU/CPU parity gate
LAYA_MODEL_DIR=/path/to/model python bench.py
```

`benchmarks/` holds the evaluation + hardening harness and a canonical
[baseline snapshot](benchmarks/baseline/). Characterisation probes that document real
model behaviour live in [examples/probes/](examples/probes/).

---

## Documentation

| File | What it is |
|------|-----------|
| [README.md](README.md) | This file — overview, install, API, resources. |
| [QWEN.md](QWEN.md) | Canonical agent-operating context: architecture, package layout, request flow, conventions. |
| [docs/INSTALL.md](docs/INSTALL.md) | Fresh-machine Linux install (including systemd setup). |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, HTTP contract, data flow, tuning. |
| [docs/MODEL-DOCTRINE.md](docs/MODEL-DOCTRINE.md) | Mental model + safe-usage rules per primitive. |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Planned work. |
| [docs/playground/SPEC.md](docs/playground/SPEC.md) | Playground design. |
| [docs/research/](docs/research/) | Feasibility + compatibility research (no implementation). |
| [CHANGELOG.md](CHANGELOG.md) | Dated change log. |
| [LICENSE](LICENSE) / [NOTICE](NOTICE) | Apache-2.0 licence + third-party attribution. |

---

## Licence

Code: **Apache-2.0** (see [LICENSE](LICENSE)). Model weights: **Apache-2.0** (Convai
Innovations, [NOTICE](NOTICE)); they are not redistributed by this repository. The
"Fox Software Solutions" name and logo are used for attribution only and are not licensed for
other use.

---

## Status

- ✅ Built & verified: GPU/CPU parity (decision-identical), batching, caching, typed
  primitives, routing signal.
- ✅ Linux + NVIDIA GPU (CUDA EP) and CPU-only tested.
- ⬜ Windows / macOS: not yet supported — see
  [docs/research/WINDOWS-MACOS-FEASIBILITY.md](docs/research/WINDOWS-MACOS-FEASIBILITY.md).
- ⬜ Jev API compatibility: scoped, not implemented — see
  [docs/research/JEV-COMPATIBILITY-SCOPE.md](docs/research/JEV-COMPATIBILITY-SCOPE.md).
