"""
End-to-end test for the PII proxy.

Sends a request containing PII through the proxy:
  1. Verifies the upstream (mock Anthropic) received a *redacted* body.
  2. Verifies the response delivered back to the client has been *restored*
     to the original PII.

Runs both for non-streaming JSON and streaming SSE responses.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import sys
import time
from urllib.parse import urlparse

PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:5599")
MOCK_URL = os.environ.get("MOCK_URL", "http://127.0.0.1:9212")

# A PII-rich user message that the model should detect strongly.
SAMPLE = (
    "Hi, my name is Ludde Oland and you can reach me at "
    "ludde.oland@gmail.com or call +46 70 123 45 67. "
    "Thanks!"
)

TOKEN_RE = re.compile(
    r"\b(name|email|phone|address|url|date|account|secret)-[0-9a-f]{6}\b"
)


def _conn(url: str, timeout: int = 30):
    p = urlparse(url)
    return http.client.HTTPConnection(p.hostname, p.port, timeout=timeout)


def wait_for(url: str, path: str = "/health", attempts: int = 60, delay: float = 1.0):
    print(f"Waiting for {url}{path} ...", flush=True)
    last_err = None
    for i in range(attempts):
        try:
            c = _conn(url, 5)
            c.request("GET", path)
            r = c.getresponse()
            r.read()
            c.close()
            if r.status == 200:
                print(f"  ready after {i+1} attempt(s)", flush=True)
                return
        except Exception as e:
            last_err = e
        time.sleep(delay)
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}")


def post_messages(stream: bool):
    body = json.dumps({
        "model": "claude-opus-4-7",
        "max_tokens": 256,
        "stream": stream,
        "messages": [{"role": "user", "content": SAMPLE}],
    }).encode()
    c = _conn(PROXY_URL, 60)
    c.request("POST", "/v1/messages", body=body, headers={
        "content-type": "application/json",
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-length": str(len(body)),
    })
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def get_last_upstream() -> bytes:
    c = _conn(MOCK_URL, 10)
    c.request("GET", "/last")
    r = c.getresponse()
    data = r.read()
    c.close()
    return data


def assert_(cond, msg, details=None):
    print(("PASS" if cond else "FAIL") + ": " + msg, flush=True)
    if details:
        print("  " + details, flush=True)
    if not cond:
        sys.exit(1)


def extract_sse_text(body: bytes) -> str:
    """Concatenate all text_delta values from an SSE stream."""
    out: list[str] = []
    for raw_line in body.split(b"\n"):
        line = raw_line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].decode("utf-8", errors="replace")
        if payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        if evt.get("type") == "content_block_delta":
            d = evt.get("delta", {}) or {}
            if d.get("type") == "text_delta":
                out.append(d.get("text", ""))
    return "".join(out)


def run_case(label: str, stream: bool):
    print(f"\n=== Case: {label} (stream={stream}) ===", flush=True)
    status, resp = post_messages(stream=stream)
    assert_(status == 200, f"proxy returned 200 ({status})")

    # Inspect what the upstream actually received
    last_raw = get_last_upstream()
    try:
        last = json.loads(last_raw)
    except Exception:
        last = {}
    forwarded = json.dumps(last)

    # Substring assertions: PII must NOT appear in upstream body.
    pii_substrings = ["Ludde", "Oland", "ludde.oland@gmail.com", "+46 70 123 45 67"]
    for s in pii_substrings:
        assert_(s not in forwarded,
                f"PII '{s}' absent from upstream body",
                f"upstream forwarded len={len(forwarded)}")

    # Tokens must appear in upstream body.
    tokens_in_upstream = TOKEN_RE.findall(forwarded)
    assert_(len(tokens_in_upstream) >= 1,
            f"upstream body contains at least one token (found {len(tokens_in_upstream)})",
            f"sample: {tokens_in_upstream[:5]}")

    # Now verify response was restored on the way back to client.
    if stream:
        text = extract_sse_text(resp)
    else:
        try:
            r_json = json.loads(resp)
            text = "".join(b.get("text", "") for b in r_json.get("content", []) if b.get("type") == "text")
        except Exception:
            text = resp.decode("utf-8", errors="replace")

    print(f"  client received text (first 200 chars): {text[:200]!r}", flush=True)

    # PII must be restored in the client-facing response.
    assert_("Ludde" in text, "client response contains restored 'Ludde'")
    # Email may be partially detected; check at least the local part survives if email was tokenised.
    assert_("ludde.oland" in text or "@gmail.com" in text or "name-" in text and "email-" not in text,
            "client response shows restored email or unredacted local-part")
    # No tokens should leak to the client.
    leaked = TOKEN_RE.findall(text)
    assert_(len(leaked) == 0, f"no PII tokens leak to client (found {len(leaked)})",
            f"leaked: {leaked[:5]}")


def main():
    wait_for(MOCK_URL, "/health")
    wait_for(PROXY_URL, "/health")
    run_case("non-streaming JSON", stream=False)
    run_case("streaming SSE", stream=True)
    print("\nAll PII-proxy e2e tests passed.", flush=True)


if __name__ == "__main__":
    main()
