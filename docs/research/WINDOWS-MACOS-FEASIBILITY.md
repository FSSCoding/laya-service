# Windows & macOS Feasibility (Research)

**Status:** Research / scoping only. No porting or code changes from this document.
**Date:** 2026-10-06.
**Supported platform today:** Linux (x86_64) with an NVIDIA GPU and CUDA 13 libraries.

## Scope

This document records what it would take to run `laya-service` on Windows and macOS, so the
decision to support (or not support) each platform is informed. It does **not** commit to
porting, and it does **not** add CI or code for either platform. The conclusion is at the end.

## Current Linux-specific surface

The parts of the service and its tooling that assume Linux are limited and well-understood:

| Concern | Linux assumption | Where it appears |
|---|---|---|
| **Startup** | Foreground process (`laya-server`) or systemd unit | `server.py`, `systemd/laya-service.service.template`, `cli.py` lifecycle verbs |
| **Config path resolution** | CWD-relative `config/laya_config.json` or `LAYA_CONFIG_PATH` | `_config.py:resolve_config_path` |
| **GPU pinning** | `CUDA_VISIBLE_DEVICES=<card>` | env, systemd template, launch commands |
| **CUDA 13 libs** | `LD_LIBRARY_PATH` must include the directory with `libcublasLt.so.13` | systemd template, launch commands, docs/INSTALL.md |
| **Result cache DB** | `~/.laya-service/result_cache.db` (home-dir path, OS-portable) | `result_cache.py` |
| **Playground** | Loopback HTTP on `127.0.0.1:<port>`, stdlib + httpx (pure Python, no OS-specific code) | `playground/server.py` |
| **Benchmark / probes** | `nvidia-smi` for VRAM snapshots; `PATH` lookups | `benchmarks/harness/systems.py`, `bench.py` |

Notable: **none of the Python source hardcodes a Linux path or calls a Linux-only syscall.** The
package itself (`laya_server`) is pure Python + onnxruntime and is platform-agnostic at the code
level. Every Linux assumption lives in *deployment* (systemd, env vars, lib paths), not in the
library.

## macOS

### Blocker: no CUDA / no `onnxruntime-gpu`

- `onnxruntime-gpu` has **no macOS wheel**. Apple GPUs have no NVIDIA driver, so the CUDA EP
  cannot load. The only onnxruntime GPU backends on macOS are Metal (via `onnxruntime-silicon`),
  which is a *different* wheel, *different* provider (`CoreMLExecutionProvider` / `MetalExecutionProvider`),
  and is **not** what this service is tested or tuned against.
- The CUDA-13 library-path gotcha does not apply (there is no cuBLAS on macOS), but that also means
  the entire "verify `/health` providers shows CUDAExecutionProvider" flow is meaningless.
- The parity bar (GPU vs CPU, CUDA EP first) cannot be expressed on macOS at all.

### What would be needed

1. A separate, untested `onnxruntime-silicon` build path (different dependency, different provider).
2. Rewriting the CUDA-specific docs, systemd→launchd unit, and the `/health` provider check.
3. Re-validating model output parity on the Metal EP (FP32 nondeterminism characteristics differ).
4. A dedicated CI/test matrix leg (no Apple hardware in the current setup).

### Conclusion (macOS)

**Not feasible without a new, untested backend and a separate maintenance line.** The model,
tokenizer, and service logic would carry over, but the GPU runtime would be entirely different and
unverified. Recommend **do not port**. CPU execution (standard `onnxruntime`) would run on macOS
but is slow and is not a supported/verified configuration of this service.

## Windows

### Partially feasible — CUDA exists on Windows, but the deployment model differs

- `onnxruntime-gpu` **does** publish Windows x64 wheels, and NVIDIA drivers + CUDA on Windows are
  normal. So the *GPU inference path* is theoretically portable.
- However, the CUDA 13 library story differs: on Windows the cuBLAS/cuDNN DLLs are loaded from the
  CUDA toolkit install (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`) or from
  `PATH` — there is no `LD_LIBRARY_PATH`. The "find `libcublasLt.so.13`" step becomes "find
  `cublasLt64_13.dll` on `PATH`".
- `CUDA_VISIBLE_DEVICES` works identically on Windows (same env-var contract), so GPU pinning is fine.

### What would be needed

1. **No systemd on Windows.** The deployment model changes to one of: a Windows service
   (NSSM or `sc.exe`), a Task Scheduler task, or a foreground `laya-server` process under a process
   supervisor. The `systemd/` template and every `cli.py` systemd lifecycle verb become
   inapplicable; a separate Windows-service path would be required.
2. **Different env plumbing.** `LD_LIBRARY_PATH` → `PATH`; the CUDA-13 lib-location guidance
   rewritten for the Windows CUDA install layout.
3. **Path conventions.** `~/.laya-service/result_cache.db` → `%USERPROFILE%\.laya-service\...`
   (already handled by Python's `os.path.expanduser`, so this is likely fine unmodified — but must
   be verified).
4. **Benchmark/probes.** `nvidia-smi` exists on Windows (NVIDIA ships it), so `benchmarks/harness/systems.py`
   VRAM snapshots should mostly carry over; the VRAM-unit parsing may need checking.
5. **A dedicated CI/test matrix leg** on Windows runners.

### Conclusion (Windows)

**Feasible in principle** (CUDA + onnxruntime-gpu both exist on Windows), but it is a real
deployment change, not a drop-in: the systemd lifecycle must be replaced, the CUDA lib-path docs
rewritten for Windows, and the whole thing validated on a Windows CI leg. Recommend **do not port
yet** — it is a discrete effort with ongoing maintenance cost, and there is no current demand signal.
If/when demand appears, the cleanest path is a `docs/INSTALL-WINDOWS.md` + a Windows service recipe,
keeping the library untouched.

## Summary table

| Platform | CUDA EP available? | Drop-in? | Effort | Recommendation |
|---|---|---|---|---|
| **Linux (x86_64)** | ✅ (target) | — | — | **Supported.** Fully tested. |
| **macOS** | ❌ (needs Metal EP, different wheel) | No | High (new backend + parity re-validation) | **Do not port.** |
| **Windows (x64)** | ✅ (Windows wheels exist) | No (systemd → Win service; `LD_LIBRARY_PATH` → `PATH`) | Medium | **Not yet.** Feasible if demand appears; needs its own deployment path + CI leg. |

## Guardrails for this doc

- This is a **research** artifact. Nothing here authorises code changes, CI additions, or
  dependency additions.
- Any future port decision must start from a fresh, explicit user instruction and a separate plan.
- The "supported platform" claim in README/INSTALL remains **Linux with an NVIDIA GPU + CUDA 13**.
