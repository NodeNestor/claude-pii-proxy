"""
Claude PII Proxy

Transparent proxy between Claude Code and the Anthropic API. Detects PII in
outbound requests, replaces it with deterministic short tokens
(``ludde -> name-a3f2b1``), and restores the real values in the response —
streaming or not — before they reach the client.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import ssl
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from redactor import Redactor, StreamRestorer


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


_log_path = os.path.join(os.path.expanduser("~"), ".claude", "pii-proxy-debug.log")
_log_handler = FlushFileHandler(_log_path, mode="a")
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), _log_handler],
)
log = logging.getLogger("pii-proxy")


UPSTREAM_URL = os.environ.get("PII_PROXY_UPSTREAM") or "https://api.anthropic.com"
LISTEN_PORT = int(os.environ.get("PII_PROXY_PORT") or "5599")
WARMUP = os.environ.get("PII_PROXY_WARMUP", "1") not in ("0", "false", "False", "")
MIN_SCORE = float(os.environ.get("PII_PROXY_MIN_SCORE", "0.5"))
MODEL_ID = os.environ.get("PII_PROXY_MODEL", "openai/privacy-filter")

ssl_ctx = ssl.create_default_context()
_parsed_upstream = urlparse(UPSTREAM_URL)
UPSTREAM_PATH = _parsed_upstream.path or ""

redactor = Redactor(model_id=MODEL_ID, min_score=MIN_SCORE)


def _join_path(upstream_path: str, request_path: str) -> str:
    if not upstream_path:
        return request_path
    if not request_path or request_path == "/":
        return upstream_path
    if upstream_path.endswith("/") and request_path.startswith("/"):
        return upstream_path[:-1] + request_path
    if not upstream_path.endswith("/") and not request_path.startswith("/"):
        return upstream_path + "/" + request_path
    return upstream_path + request_path


def _upstream_conn():
    if _parsed_upstream.scheme == "https":
        return http.client.HTTPSConnection(
            _parsed_upstream.hostname,
            _parsed_upstream.port or 443,
            context=ssl_ctx,
            timeout=600,
        )
    return http.client.HTTPConnection(
        _parsed_upstream.hostname,
        _parsed_upstream.port or 80,
        timeout=600,
    )


def _forward_headers(req_headers: dict, body: bytes | None = None, strip_encoding: bool = False) -> dict:
    headers: dict[str, str] = {}
    for key, value in req_headers.items():
        lower = key.lower()
        if lower in ("host", "transfer-encoding", "connection", "content-length"):
            continue
        if strip_encoding and lower == "accept-encoding":
            continue
        headers[key] = value
    if body is not None:
        headers["content-length"] = str(len(body))
    return headers


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # silence default access log
        pass

    # ------- request plumbing -------

    def _read_body(self) -> bytes:
        length = int(self.headers.get("content-length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _get_headers_dict(self) -> dict:
        return {key: value for key, value in self.headers.items()}

    def do_POST(self):
        log.info(f"[REQ] POST {self.path}")
        if self.path.startswith("/v1/messages"):
            self._handle_messages()
        else:
            self._proxy_raw("POST")

    def do_GET(self):
        log.info(f"[REQ] GET {self.path}")
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/debug/map":
            self._handle_debug_map()
        else:
            self._proxy_raw("GET")

    def do_PUT(self):
        self._proxy_raw("PUT")

    def do_DELETE(self):
        self._proxy_raw("DELETE")

    def do_PATCH(self):
        self._proxy_raw("PATCH")

    def do_OPTIONS(self):
        self._proxy_raw("OPTIONS")

    # ------- helpers -------

    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_health(self):
        self._send_json(200, {
            "status": "ok",
            "upstream": UPSTREAM_URL,
            "model_id": redactor.model_id,
            "mappings": len(redactor.map),
            "min_score": redactor.min_score,
        })

    def _handle_debug_map(self):
        # Don't dump real values — only counts and token examples.
        sample = list(redactor.map._token_to_real.keys())[:10]
        self._send_json(200, {"count": len(redactor.map), "tokens_sample": sample})

    def _proxy_raw(self, method: str):
        body = self._read_body()
        headers = _forward_headers(self._get_headers_dict(), body if body else None)
        try:
            conn = _upstream_conn()
            conn.request(method, _join_path(UPSTREAM_PATH, self.path), body=body or None, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            has_cl = False
            for key, value in resp.getheaders():
                lk = key.lower()
                if lk in ("connection", "transfer-encoding"):
                    continue
                if lk == "content-length":
                    has_cl = True
                self.send_header(key, value)
            if not has_cl:
                self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            conn.close()
        except Exception as e:
            log.error(f"[RAW] {e}", exc_info=True)
            self._send_json(502, {"error": str(e)})

    # ------- /v1/messages -------

    def _handle_messages(self):
        raw_body = self._read_body()
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        # Redact in place
        try:
            redactor.redact_request_payload(payload)
        except Exception as e:
            log.error(f"[MSG] Redaction failed (forwarding original body): {e}", exc_info=True)

        body_out = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        is_streaming = bool(payload.get("stream", False))

        req_headers = self._get_headers_dict()
        out_headers = _forward_headers(req_headers, body_out, strip_encoding=True)

        log.info(f"[MSG] -> {UPSTREAM_URL}{self.path} stream={is_streaming} bytes={len(body_out):,} mappings={len(redactor.map)}")

        try:
            conn = _upstream_conn()
            conn.request("POST", _join_path(UPSTREAM_PATH, self.path), body=body_out, headers=out_headers)
            resp = conn.getresponse()
        except Exception as e:
            log.error(f"[MSG] Upstream error: {e}", exc_info=True)
            self._send_json(502, {"error": str(e)})
            return

        try:
            if is_streaming:
                self._forward_sse(resp)
            else:
                self._forward_full(resp)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _forward_full(self, resp):
        body = resp.read()
        modified = body
        try:
            data = json.loads(body)
            data = redactor.restore_in_obj(data)
            modified = json.dumps(data, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            log.warning(f"[MSG] Could not restore non-streaming response: {e}")
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            lk = k.lower()
            if lk in ("connection", "transfer-encoding", "content-length"):
                continue
            self.send_header(k, v)
        self.send_header("content-length", str(len(modified)))
        self.end_headers()
        self.wfile.write(modified)
        self.wfile.flush()

    # ------- streaming SSE -------

    def _forward_sse(self, resp):
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            lk = k.lower()
            if lk in ("connection", "transfer-encoding", "content-length"):
                continue
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()

        # Per-content-block state for safe streaming restoration.
        block_restorers: dict[int, StreamRestorer] = {}
        block_delta_type: dict[int, str] = {}  # idx -> "text_delta" | "input_json_delta"

        def write_line(line: bytes):
            self.wfile.write(line + b"\n")
            self.wfile.flush()

        def write_event(evt: dict):
            write_line(b"data: " + json.dumps(evt, ensure_ascii=False).encode("utf-8"))

        def emit_synthetic(idx: int, text: str):
            if not text:
                return
            dtype = block_delta_type.get(idx, "text_delta")
            key = "partial_json" if dtype == "input_json_delta" else "text"
            write_event({
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": dtype, key: text},
            })

        def process_data_line(data_str: str, raw_line: bytes):
            try:
                evt = json.loads(data_str)
            except Exception:
                write_line(raw_line)
                return

            t = evt.get("type", "")

            if t == "content_block_start":
                idx = int(evt.get("index", 0))
                block = evt.get("content_block", {}) or {}
                btype = block.get("type")
                block_delta_type[idx] = "input_json_delta" if btype == "tool_use" else "text_delta"
                block_restorers[idx] = StreamRestorer(redactor)
                evt = redactor.restore_in_obj(evt)
                write_event(evt)
                return

            if t == "content_block_delta":
                idx = int(evt.get("index", 0))
                delta = evt.get("delta", {}) or {}
                dtype = delta.get("type", "")
                if dtype:
                    block_delta_type.setdefault(idx, dtype)
                restorer = block_restorers.get(idx)
                if restorer is None:
                    restorer = StreamRestorer(redactor)
                    block_restorers[idx] = restorer
                if dtype == "text_delta":
                    delta["text"] = restorer.feed(delta.get("text", ""))
                    evt["delta"] = delta
                elif dtype == "input_json_delta":
                    delta["partial_json"] = restorer.feed(delta.get("partial_json", ""))
                    evt["delta"] = delta
                else:
                    evt = redactor.restore_in_obj(evt)
                write_event(evt)
                return

            if t == "content_block_stop":
                idx = int(evt.get("index", 0))
                restorer = block_restorers.pop(idx, None)
                if restorer:
                    tail = restorer.flush()
                    if tail:
                        emit_synthetic(idx, tail)
                evt = redactor.restore_in_obj(evt)
                write_event(evt)
                return

            # Other event types: just restore tokens in any string fields.
            evt = redactor.restore_in_obj(evt)
            write_event(evt)

        line_buf = b""
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                line_buf += chunk
                while b"\n" in line_buf:
                    line, _, line_buf = line_buf.partition(b"\n")
                    if line.endswith(b"\r"):
                        line = line[:-1]
                    if line.startswith(b"data: "):
                        data_str = line[6:].decode("utf-8", errors="replace")
                        if data_str.strip() == "[DONE]":
                            write_line(line)
                        else:
                            process_data_line(data_str, line)
                    else:
                        write_line(line)
            if line_buf:
                if line_buf.startswith(b"data: "):
                    process_data_line(line_buf[6:].decode("utf-8", errors="replace"), line_buf)
                else:
                    write_line(line_buf)
            # flush any remaining tails (rare; should be drained by content_block_stop)
            for idx, restorer in list(block_restorers.items()):
                tail = restorer.flush()
                if tail:
                    emit_synthetic(idx, tail)
        except Exception as e:
            log.error(f"[SSE] Stream forwarding error: {e}", exc_info=True)


class ThreadedHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    log.info(f"Starting Claude PII Proxy on port {LISTEN_PORT}")
    log.info(f"  Upstream: {UPSTREAM_URL}")
    log.info(f"  Model:    {redactor.model_id}")
    log.info(f"  Min score: {redactor.min_score}")
    log.info(f"  Mappings loaded: {len(redactor.map)}")
    if WARMUP:
        log.info("  Warming up model...")
        try:
            redactor.warmup()
            log.info("  Warmup done.")
        except Exception as e:
            log.warning(f"[WARMUP] Failed (will load on first request): {e}")
    server = ThreadedHTTPServer(("127.0.0.1", LISTEN_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
