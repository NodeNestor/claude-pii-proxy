"""
Mock Anthropic API for end-to-end testing of the PII proxy.

* Captures the last received /v1/messages body (so the test can verify the
  proxy redacted PII before forwarding).
* Echoes the user's last text content back in the assistant response so the
  test can verify the round-trip restoration on the way back.
"""
from __future__ import annotations

import json
import logging
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MOCK] %(message)s", stream=sys.stdout)
log = logging.getLogger("mock")

LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9212

# Module-level capture: the last raw body received at /v1/messages.
LAST_BODY: bytes = b""


def _last_user_text(payload: dict) -> str:
    """Return the last user-message text from a /v1/messages payload."""
    for msg in reversed(payload.get("messages", []) or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
    return ""


class MockAnthropicHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "mock-ok"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/last":
            body = LAST_BODY
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        global LAST_BODY
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length) if length > 0 else b""
        LAST_BODY = raw
        log.info(f"POST {self.path} ({length} bytes)")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}

        is_stream = bool(payload.get("stream", False))
        model = payload.get("model", "mock-model")
        user_text = _last_user_text(payload)

        # Echo the user's text back inside the assistant response. If the proxy
        # redacted the user text, that redacted form appears here, and on the
        # way back the proxy must restore it for the test to pass.
        echo_text = (
            f"You said: {user_text}\n"
            f"(received {len(payload.get('messages', []) or [])} messages)"
        )
        input_tokens = max(len(raw) // 4, 50)
        output_tokens = max(len(echo_text) // 4, 1)

        if is_stream:
            self._send_sse(model, input_tokens, output_tokens, echo_text)
        else:
            self._send_json(model, input_tokens, output_tokens, echo_text)

    def _send_sse(self, model, input_tokens, output_tokens, text):
        events = []
        events.append(json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_mock_001",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        }))
        events.append(json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }))
        # Stream in 3-char chunks to exercise the proxy's tail buffer
        chunk_size = 3
        for i in range(0, len(text), chunk_size):
            events.append(json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text[i : i + chunk_size]},
            }))
        events.append(json.dumps({
            "type": "content_block_stop",
            "index": 0,
        }))
        events.append(json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": output_tokens},
        }))
        events.append(json.dumps({"type": "message_stop"}))

        body = "".join(f"data: {e}\n\n" for e in events).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_json(self, model, input_tokens, output_tokens, text):
        resp = {
            "id": "msg_mock_001",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    log.info(f"Mock Anthropic API on port {LISTEN_PORT}")
    server = HTTPServer(("0.0.0.0", LISTEN_PORT), MockAnthropicHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
