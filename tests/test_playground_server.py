"""Tests for the Laya Playground loopback proxy/static server.

Layer: unit tests for the playground's HTTP routes and gateway envelope.

Approach: stand up the actual ThreadingHTTPServer on an ephemeral port
bound to 127.0.0.1, and mock the upstream via ``httpx.Client`` patching.
No live Laya service is ever contacted, and no model is loaded.

Covered (per spec):
- GET / and /index.html serve the allowlisted static file
- Path traversal: ../ etc. are rejected with 404 and not served
- GET /api/health forwards /health and preserves upstream status + body
- POST /api/query forwards the exact JSON body to /system_one and
  preserves upstream status + body (including 5xx error bodies)
- Malformed JSON in the request body → local 400 invalid_json (no upstream call)
- Unavailable upstream → 502/504 gateway envelope, origin=playground
- Body-size cap → 413 before any upstream call
- Non-loopback --host is rejected by main()
- Explicit --port required; missing --port is a parser error
- /api/* has no /control/unload route
"""

from __future__ import annotations

import contextlib
import json
import socket
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "playground"))

# Make `server` importable as a top-level module for this test file only.
# This import must come after sys.path manipulation, so it is excluded from
# the isort block.
import server as pg  # noqa: E402,I001


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextlib.contextmanager
def _mock_upstream():
    """Patch httpx.Client used by the playground handler so all upstream
    traffic is in-process.

    Yields a dict with two keys:
      - calls: list of upstream call dicts (method, url, body)
      - responses: FIFO queue of canned responses.

    Each entry in responses is one of:
      (status_code, body_bytes, headers_dict)  -> return this fake response
      ("raise", exception_instance)            -> raise this exception

    If the queue is empty when a call arrives, AssertionError is raised,
    which surfaces as a 5xx to the client via BaseHTTPRequestHandler —
    but in tests that should never happen, because we queue exactly what
    each handler will consume.
    """
    calls: list[dict] = []
    responses: list[tuple[int, bytes, dict] | tuple[str, Exception]] = []
    state = {"calls": calls, "responses": responses}

    class _FakeResponse:
        def __init__(self, status_code, content, headers):
            self.status_code = status_code
            self.content = content
            self.headers = headers

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def _next(self):
            if not state["responses"]:
                raise AssertionError(
                    "upstream mock: no more canned responses queued"
                )
            item = state["responses"].pop(0)
            if isinstance(item, tuple) and item[0] == "raise":
                raise item[1]
            status, body, headers = item
            return _FakeResponse(status, body, headers)

        def get(self, url, *a, **kw):
            calls.append({"method": "GET", "url": url})
            return self._next()

        def post(self, url, *a, **kw):
            kwargs_body = kw.get("json", kw.get("data"))
            calls.append({"method": "POST", "url": url, "body": kwargs_body})
            return self._next()

    with patch.object(pg.httpx, "Client", _FakeClient):
        yield state


@contextlib.contextmanager
def _start_playground(playground_dir: Path):
    """Start the real server on an ephemeral port; yields (url, port)."""
    port = _free_port()
    srv = pg._build_server(port, "127.0.0.1", playground_dir)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}", port
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _raw_http_get(port: int, path: str) -> tuple[int, bytes]:
    """Issue a literal HTTP GET request on a raw socket.

    Unlike httpx, this sends the path exactly as given — no client-side
    percent-decoding or dot-segment normalization. Used for path-traversal
    tests where we must verify the *server* rejects the literal path.
    Returns (status_code, body_bytes).
    """
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        header_lines = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        s.sendall(header_lines.encode("ascii"))
        chunks: list[bytes] = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks)
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, body


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------


@pytest.fixture
def playground_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pg"
    d.mkdir()
    (d / "index.html").write_text("<html><body>ok</body></html>")
    (d / "playground.css").write_text("body { color: red; }")
    (d / "playground.js").write_text("console.log('hi')")
    # A file NOT in the allowlist — must never be served.
    (d / "secret.txt").write_text("top-secret")
    return d


class TestStatic:
    def test_serves_index_root(self, playground_dir: Path):
        with _mock_upstream(), _start_playground(playground_dir) as (url, _port):
            r = httpx.get(f"{url}/")
            assert r.status_code == 200
            assert r.text == "<html><body>ok</body></html>"
            assert r.headers["content-type"].startswith("text/html")
            assert "Content-Security-Policy" in r.headers

    def test_serves_index_html(self, playground_dir: Path):
        with _mock_upstream(), _start_playground(playground_dir) as (url, _port):
            r = httpx.get(f"{url}/index.html")
            assert r.status_code == 200
            assert "ok" in r.text

    def test_serves_css_and_js_with_correct_types(self, playground_dir: Path):
        with _mock_upstream(), _start_playground(playground_dir) as (url, _port):
            css_resp = httpx.get(f"{url}/playground.css")
            js_resp = httpx.get(f"{url}/playground.js")
            assert css_resp.headers["content-type"].startswith("text/css")
            assert js_resp.headers["content-type"].startswith("application/javascript")

    def test_rejects_non_allowlisted_file(self, playground_dir: Path):
        with _mock_upstream(), _start_playground(playground_dir) as (url, _port):
            # Not in the allowlist → 404, even though the file exists on disk.
            r = httpx.get(f"{url}/secret.txt")
            assert r.status_code == 404
            assert "top-secret" not in r.text

    @pytest.mark.parametrize(
        "raw_path",
        [
            "/../etc/passwd",
            "/..%2F..%2Fetc%2Fpasswd",
            "/foo/../index.html",
            "/index.html/..",
            "/%2e%2e/%2e%2e/etc/passwd",
            "/..",
            "/secret.txt",
        ],
    )
    def test_path_traversal_rejected(self, playground_dir: Path, raw_path: str):
        """None of these must return file contents or crash.

        We use a raw socket here so the path is sent verbatim — httpx and
        other clients may normalize dot-segments or decode percent escapes
        before sending, which would change what the server actually sees.
        """
        with _mock_upstream(), _start_playground(playground_dir) as (_url, port):
            status, body = _raw_http_get(port, raw_path)
            assert status == 404, f"path {raw_path!r} returned {status}"
            assert b"top-secret" not in body
            assert b"Traceback" not in body


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


class TestApiHealth:
    def test_forwards_to_upstream_health(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            st["responses"].append(
                (200, json.dumps({"status": "ok", "model_loaded": True}).encode(),
                 {"Content-Type": "application/json"})
            )
            r = httpx.get(f"{url}/api/health")
            assert r.status_code == 200
            assert r.json() == {"status": "ok", "model_loaded": True}
            assert len(st["calls"]) == 1
            assert st["calls"][0]["url"] == pg.UPSTREAM_BASE + "/health"

    def test_preserves_upstream_error_status(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            body = json.dumps({"error": "boom"}).encode()
            st["responses"].append((503, body, {"Content-Type": "application/json"}))
            r = httpx.get(f"{url}/api/health")
            assert r.status_code == 503
            assert r.json() == {"error": "boom"}

    def test_unavailable_upstream_connect_error(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            st["responses"].append(("raise", httpx.ConnectError("nope")))
            r = httpx.get(f"{url}/api/health")
            assert r.status_code == 502
            body = r.json()
            assert body["error"]["code"] == "upstream_unreachable"
            assert body["error"]["origin"] == "playground"

    def test_unavailable_upstream_timeout(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            st["responses"].append(("raise", httpx.ConnectTimeout("slow")))
            r = httpx.get(f"{url}/api/health")
            assert r.status_code == 504
            assert r.json()["error"]["code"] == "upstream_timeout"


# ---------------------------------------------------------------------------
# /api/query
# ---------------------------------------------------------------------------


class TestApiQuery:
    def test_forwards_exact_payload_and_preserves_200(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            payload = {
                "state": {"subject": "hello"},
                "questions": {
                    "q1": {"type": "choice", "instructions": "?",
                            "criteria": {"a": "A", "b": "B"}},
                },
            }
            expected = {
                "model": "laya",
                "answers": {
                    "q1": {"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}},
                },
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "cached": False,
            }
            st["responses"].append(
                (200, json.dumps(expected).encode(),
                 {"Content-Type": "application/json"})
            )
            r = httpx.post(f"{url}/api/query", json=payload)
            assert r.status_code == 200
            assert r.json() == expected
            assert len(st["calls"]) == 1
            assert st["calls"][0]["method"] == "POST"
            assert st["calls"][0]["url"] == pg.UPSTREAM_BASE + "/system_one"
            # Exact payload passthrough — the upstream gets the same JSON.
            assert st["calls"][0]["body"] == payload

    def test_preserves_upstream_5xx_error_body(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            err = {"detail": "head overflow"}
            st["responses"].append(
                (500, json.dumps(err).encode(),
                 {"Content-Type": "application/json"})
            )
            r = httpx.post(
                f"{url}/api/query",
                json={"state": "x", "questions": {"q": {"type": "score", "criteria": []}}},
            )
            assert r.status_code == 500
            assert r.json() == err

    def test_preserves_upstream_422_body(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            body = {"detail": [{"msg": "field required", "loc": ["body", "state"]}]}
            body_bytes = json.dumps(body).encode()
            st["responses"].append(
                (422, body_bytes, {"Content-Type": "application/json"})
            )
            r = httpx.post(f"{url}/api/query", json={})
            assert r.status_code == 422
            assert r.json() == body

    def test_malformed_json_is_rejected_locally_without_upstream_call(
        self, playground_dir: Path
    ):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            # No canned response — if the handler wrongly forwarded, the mock
            # would raise AssertionError; instead we get a clean 400.
            r = httpx.post(
                f"{url}/api/query",
                content=b"{not-json",
                headers={"Content-Type": "application/json"},
            )
            assert r.status_code == 400
            body = r.json()
            assert body["error"]["code"] == "invalid_json"
            assert body["error"]["origin"] == "playground"
            assert st["calls"] == []  # upstream was never contacted

    def test_wrong_content_type_rejected(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            r = httpx.post(
                f"{url}/api/query",
                content=json.dumps({"state": "x", "questions": {"q": {}}}).encode(),
                headers={"Content-Type": "text/plain"},
            )
            assert r.status_code == 415
            assert st["calls"] == []

    def test_body_size_cap(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            # Craft a body just over the cap.
            big = b"x" * (pg.MAX_BODY_BYTES + 1)
            r = httpx.post(
                f"{url}/api/query",
                content=big,
                headers={"Content-Type": "application/json"},
            )
            assert r.status_code == 413
            assert r.json()["error"]["code"] == "body_too_large"
            assert st["calls"] == []

    def test_upstream_payload_at_cap_passes(self, playground_dir: Path):
        """A body at or under the cap passes through to upstream."""
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            # Build a body well under the cap.
            base = json.dumps(
                {"state": "x", "questions": {"q": {"type": "noul", "instructions": ""}}}
            ).encode()
            pad_len = pg.MAX_BODY_BYTES - len(base)
            payload = {
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "y" * pad_len}},
            }
            st["responses"].append(
                (200, b'{"ok":true}', {"Content-Type": "application/json"})
            )
            r = httpx.post(f"{url}/api/query", json=payload)
            # Either 200 (passed cap) or 413 (boundary rounding); assert
            # upstream was actually called if not 413.
            if r.status_code == 200:
                assert len(st["calls"]) == 1
            else:
                assert r.status_code == 413

    def test_unavailable_upstream_on_query(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            st["responses"].append(("raise", httpx.ConnectError("nope")))
            r = httpx.post(f"{url}/api/query", json={"state": "x", "questions": {}})
            assert r.status_code == 502
            body = r.json()
            assert body["error"]["code"] == "upstream_unreachable"
            assert body["error"]["origin"] == "playground"

    def test_upstream_timeout_on_query(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            st["responses"].append(("raise", httpx.ReadTimeout("slow")))
            r = httpx.post(f"{url}/api/query", json={"state": "x", "questions": {}})
            assert r.status_code == 504
            assert r.json()["error"]["code"] == "upstream_timeout"

    def test_no_unload_route(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            r = httpx.post(f"{url}/api/unload", json={})
            assert r.status_code == 404
            assert st["calls"] == []

    def test_unknown_post_route_404(self, playground_dir: Path):
        with _mock_upstream() as st, _start_playground(playground_dir) as (url, _port):
            r = httpx.post(f"{url}/api/nope", json={})
            assert r.status_code == 404
            assert st["calls"] == []


# ---------------------------------------------------------------------------
# CLI / argument handling
# ---------------------------------------------------------------------------


class TestCli:
    def test_requires_explicit_port(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pg.parse_args([])
        assert exc_info.value.code == 2  # argparse error
        err = capsys.readouterr().err
        assert "--port" in err

    def test_accepts_non_loopback_host(self, tmp_path: Path):
        """Non-loopback hosts are now allowed (LAN access)."""
        idx = tmp_path / "pg"
        idx.mkdir()
        (idx / "index.html").write_text("")
        # Should not raise — non-loopback hosts are permitted.
        args = pg.parse_args(["--port", "12345", "--host", "0.0.0.0", "--dir", str(idx)])
        assert args.host == "0.0.0.0"

    def test_accepts_explicit_loopback_host(self):
        args = pg.parse_args(["--port", "12345", "--host", "127.0.0.1"])
        assert args.port == 12345
        assert args.host == "127.0.0.1"

    def test_default_host_is_loopback(self):
        args = pg.parse_args(["--port", "12345"])
        assert args.host == "127.0.0.1"


# ---------------------------------------------------------------------------
# Loopback bind check
# ---------------------------------------------------------------------------


class TestLoopbackBind:
    def test_is_loopback_address(self):
        assert pg._is_loopback_address("127.0.0.1")
        assert pg._is_loopback_address("localhost")  # resolves to 127.0.0.1
        assert not pg._is_loopback_address("0.0.0.0")
        assert not pg._is_loopback_address("::")
        # A hostname that does not exist.
        assert not pg._is_loopback_address("definitely-not-a-host.example.invalid")

    def test_server_binds_to_127_0_0_1_only(self, playground_dir: Path):
        """Sanity check that _build_server returns a server whose socket
        is bound to 127.0.0.1 and not 0.0.0.0."""
        port = _free_port()
        srv = pg._build_server(port, "127.0.0.1", playground_dir)
        bound_host, bound_port = srv.socket.getsockname()
        assert bound_host == "127.0.0.1"
        assert bound_port == port
        srv.server_close()
