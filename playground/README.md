# Laya Playground

Local-only learning workbench for the Laya inference service. Type a state,
ask a typed question, press **Run**, see what Laya actually returns.

## Start

```bash
# from the repository root, with the venv activated:
python playground/server.py --port <PORT>
```

Pick a free loopback port yourself (the server refuses to start without an
explicit `--port`). By default the upstream is `http://127.0.0.1:11448`; if your
service listens on a different port, pass `--upstream http://127.0.0.1:<port>`
(loopback-only).

Then open `http://127.0.0.1:<PORT>/` in your browser.

## What it does

- Serves a static HTML/CSS/JS learning interface from `playground/static/`
- Proxies `GET /api/health` → Laya `/health`
- Proxies `POST /api/query` → Laya `/system_one` (exact body passthrough)
- Loopback-only (127.0.0.1). No authentication, no persistence, no history.
- Separate HTTP requests are **not** coalesced into one inference pass.
  Multiple questions inside one `/system_one` request **are** answered in one
  forward pass — that is what the Batch tab means.

## Tests

```bash
.venv/bin/python -m pytest tests/test_playground_server.py -v
```

## Files

| File | Purpose |
|---|---|
| `server.py` | Loopback-only static + proxy server (stdlib + httpx) |
| `static/index.html` | Single-page learning workbench |
| `static/playground.css` | FSS dark/light theme, chart/routing/batch styles |
| `static/playground.js` | Interactive layer: tabs, forms, fetch, rendering, probes |
| `tests/test_playground_server.py` | Proxy server tests (mocked upstream) |
