# Contributing to laya-service

Bug reports and pull requests are welcome. This is a small, focused service; keep changes tight and
in the spirit of the existing code.

## Good contributions

- Bugs in the primitives, caching, batching, or HTTP layer (with a repro).
- Docs fixes that improve the stranger-facing install/use story.
- Test coverage that pins current behaviour (parity bar: decisions invariant, numerics within `1e-3`).
- Small, self-contained improvements that don't change the public API or the model.

## Not in scope (yet)

- Porting to Windows/macOS — see `docs/research/WINDOWS-MACOS-FEASIBILITY.md` (research only).
- Jev `system_one` client adapter — see `docs/research/JEV-COMPATIBILITY-SCOPE.md` (scoped, not built).
- New model versions, retraining, or weight changes — the model is a bounded semantic decision tool;
  its characterisation is documented, and behaviour changes need a real adoption use case.

## Before you open a PR

1. **Set up** per `docs/INSTALL.md` (Linux, NVIDIA GPU + CUDA 13, or CPU for parity testing).
2. **Point `model_dir`** at a valid Laya ONNX bundle in `config/laya_config.json`.
3. **Run the checks** and make sure they're green:

   ```bash
   .venv/bin/python -m ruff check src tests
   .venv/bin/python -m pytest tests/ -v
   ```

4. Keep commits focused. A coherent, well-described change beats several scattered ones.

## How to file an issue

- Reproduce on a clean checkout with the current `config/laya_config.example.json` pattern.
- Note the provider from `GET /health` (CUDA vs CPU) — many "weird" results are just CPU fallback.
- Include the exact request (`state` + `questions`) and response, plus version.

## Support

This project does not offer a support SLA. Issues are triaged on the maintainer's schedule. For
questions about Laya's *behaviour* (not service bugs), read `docs/MODEL-DOCTRINE.md` first — a lot of
what looks like a bug is documented model character.

## Code of conduct

Be kind. Disagree with the code, not the person.
