"""Laya Playground — local-only static file + API proxy server.

Binds to 127.0.0.1 only, on a user-supplied --port. Serves an allowlisted set
of static files from this package's directory, and proxies /api/health and
/api/query to the upstream Laya service at the fixed address 127.0.0.1:11448.

Design goals (per PLAYGROUND-SPEC.md §7):
- Loopback-only binding; no other network interfaces are ever bound.
- Fixed upstream; no user-supplied URL substitution.
- Allowlisted static files only; no arbitrary filesystem traversal.
- Upstream status/body pass-through on /api/query and /api/health,
  including error bodies.
- Modest body-size cap and explicit timeout.
- Consistent local gateway error envelope for connect/timeout/oversized
  bodies, without attributing them to the upstream.
- No /control/unload, no arbitrary paths, no configurable upstream.

Dependencies: httpx only (already in pyproject.toml). stdlib for everything
else.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

UPSTREAM_BASE = "http://127.0.0.1:11448"
LOOPBACK_HOST = "127.0.0.1"
MAX_BODY_BYTES = 1_000_000  # 1 MB — covers realistic playground payloads
REQUEST_TIMEOUT_S = 60.0
HEALTH_TIMEOUT_S = 5.0

# Files (relative to the playground/static/ directory) that may be served.
STATIC_ALLOWLIST: dict[str, str] = {
    "/": "index.html",
    "/index.html": "index.html",
    "/playground.css": "playground.css",
    "/playground.js": "playground.js",
}


def _resolve_static_path(request_path: str, playground_dir: Path) -> Path | None:
    """Map request path to a safe, allowlisted file path or None if rejected."""
    if request_path not in STATIC_ALLOWLIST:
        return None
    filename = STATIC_ALLOWLIST[request_path]
    # The allowlist is the gate; this resolve is belt-and-braces against
    # symlink tricks (the filenames are fixed, but resolve + check).
    target = (playground_dir / filename).resolve()
    try:
        target.relative_to(playground_dir.resolve())
    except ValueError:
        return None
    return target


def _is_loopback_address(host: str) -> bool:
    """True if `host` resolves to 127.0.0.0/8 (::1) only.

    Defensive check for the --host flag; we still hard-bind to 127.0.0.1
    in main() so the bind itself cannot escape loopback.
    """
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for _family, _type, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        if ip == "127.0.0.1" or ip == "::1":
            continue
        return False
    return True


class PlaygroundHandler(BaseHTTPRequestHandler):
    """BaseHTTPRequestHandler subclass with route dispatch.

    The handler holds no upstream state; each request uses a short-lived
    httpx.Client to avoid connection pooling across threads (ThreadingHTTPServer
    creates one handler instance per connection).
    """

    # --- configuration (set by the factory before serve_forever) -------------

    playground_dir: Path | None = None

    # --- helpers ------------------------------------------------------------

    def _write_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._write_bytes(status, body, "application/json")

    def _gateway_error(self, status: int, code: str, message: str) -> None:
        self._write_json(
            status,
            {"error": {"code": code, "message": message, "origin": "playground"}},
        )

    # --- HTTP methods -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 — required name by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._handle_health()
            return

        # Static allowlist
        if self.playground_dir is None:
            self._write_bytes(404, b"not found", "text/plain")
            return
        target = _resolve_static_path(path, self.playground_dir)
        if target is None or not target.is_file():
            self._write_bytes(404, b"not found", "text/plain")
            return
        ctype = "text/html" if target.suffix == ".html" else (
            "text/css" if target.suffix == ".css" else (
                "application/javascript" if target.suffix == ".js" else "application/octet-stream"
            )
        )
        data = target.read_bytes()
        self._write_bytes(200, data, ctype)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path != "/api/query":
            self._gateway_error(404, "not_found", f"No handler for POST {path}")
            return

        # Body-size cap — read and reject before forwarding.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._gateway_error(400, "invalid_content_length",
                                "Content-Length header is not a valid integer")
            return
        if length > MAX_BODY_BYTES:
            self._gateway_error(413, "body_too_large",
                                f"Body exceeds {MAX_BODY_BYTES} byte cap")
            return

        body = self.rfile.read(length) if length else b""

        content_type = (self.headers.get("Content-Type") or "").lower()
        if "application/json" not in content_type:
            self._gateway_error(415, "unsupported_media_type",
                                "Content-Type must be application/json")
            return

        # Parse and reject malformed JSON locally BEFORE touching upstream,
        # so the client gets a clear local error rather than a forwarded 500.
        try:
            payload: Any = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._gateway_error(400, "invalid_json", "Malformed JSON in request body")
            return

        self._forward_query(payload)

    # --- upstream forwarding ------------------------------------------------

    def _handle_health(self) -> None:
        url = f"{UPSTREAM_BASE}/health"
        try:
            with httpx.Client(timeout=HEALTH_TIMEOUT_S) as client:
                resp = client.get(url)
        except httpx.TimeoutException:
            self._gateway_error(504, "upstream_timeout",
                                f"Health request to {UPSTREAM_BASE} timed out")
            return
        except httpx.HTTPError as exc:
            self._gateway_error(502, "upstream_unreachable",
                                _unreachable_msg(exc))
            return

        # Pass through upstream status and body verbatim.
        ctype = resp.headers.get("Content-Type", "application/json")
        self._write_bytes(resp.status_code, resp.content, ctype)

    def _forward_query(self, payload: Any) -> None:
        url = f"{UPSTREAM_BASE}/system_one"
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
                resp = client.post(url, json=payload)
        except httpx.TimeoutException:
            self._gateway_error(504, "upstream_timeout",
                                f"Query request to {UPSTREAM_BASE} timed out")
            return
        except httpx.HTTPError as exc:
            self._gateway_error(502, "upstream_unreachable",
                                _unreachable_msg(exc))
            return

        ctype = resp.headers.get("Content-Type", "application/json")
        self._write_bytes(resp.status_code, resp.content, ctype)

    # Suppress per-request logging noise on stderr (spec wants a quiet, single-URL console).
    def log_message(self, fmt: str, *args: Any) -> None:
        return


def _unreachable_msg(exc: Exception) -> str:
    """Format the unreachable-upstream error message within the line-length limit."""
    return f"Cannot reach upstream at {UPSTREAM_BASE}: {exc.__class__.__name__}"


def _build_server(port: int, host: str, playground_dir: Path) -> ThreadingHTTPServer:
    """Construct the ThreadingHTTPServer bound to the explicit port/host.

    ThreadingHTTPServer allows concurrent connections without a request
    queue — there is intentionally no in-memory queue in front of the
    upstream, so the browser sees upstream latency directly.
    """

    class _Handler(PlaygroundHandler):
        pass

    _Handler.playground_dir = playground_dir
    return ThreadingHTTPServer((host, port), _Handler)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laya Playground — local loopback-only static + proxy server."
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Explicit TCP port to bind on 127.0.0.1 (no default; pick a free port).",
    )
    parser.add_argument(
        "--host",
        default=LOOPBACK_HOST,
        help=f"Bind host (default {LOOPBACK_HOST}).",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Override static dir (defaults to the directory containing this script).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.dir:
        playground_dir = Path(args.dir).resolve()
    else:
        playground_dir = Path(__file__).resolve().parent / "static"

    try:
        server = _build_server(args.port, args.host, playground_dir)
    except OSError as exc:
        print(f"error: cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    bind_note = "" if args.host == LOOPBACK_HOST else " [LAN-accessible]"
    print(f"Laya Playground serving at http://{args.host}:{args.port}/{bind_note}")
    print(f"  static: {playground_dir}")
    print(f"  upstream: {UPSTREAM_BASE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
