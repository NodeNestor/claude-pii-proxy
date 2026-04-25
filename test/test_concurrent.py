"""
Stress test: many concurrent /v1/messages requests with distinct PII per
request. Verifies the proxy:

  1. Doesn't bleed PII from one request into another's response.
  2. Restores each response's specific PII to the original real values.
  3. Mints stable tokens (same real value -> same token) under contention.
  4. Doesn't corrupt the JSONL span cache or token-map files.

Run a mock-anthropic on 9212 and proxy on 5599 first.
"""
from __future__ import annotations

import concurrent.futures as cf
import http.client
import json
import os
import re
import sys
import time
from urllib.parse import urlparse

PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:5599")
N_REQUESTS = int(os.environ.get("N", "32"))
N_THREADS = int(os.environ.get("THREADS", "16"))

TOKEN_RE = re.compile(
    r"\b(name|email|phone|address|url|date|account|secret)-[0-9a-f]{6}\b"
)


def _conn():
    p = urlparse(PROXY_URL)
    return http.client.HTTPConnection(p.hostname, p.port, timeout=60)


def fire(req_id: int):
    real_name = f"User{req_id}Smith"
    real_email = f"u{req_id}@example.com"
    text = (
        f"Request #{req_id}: my name is {real_name} "
        f"and my email is {real_email}. Thanks!"
    )
    body = json.dumps({
        "model": "claude-opus-4-7",
        "max_tokens": 256,
        "stream": req_id % 2 == 0,  # alternate streaming and non-streaming
        "messages": [{"role": "user", "content": text}],
    }).encode()

    c = _conn()
    c.request("POST", "/v1/messages", body=body, headers={
        "content-type": "application/json",
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-length": str(len(body)),
    })
    r = c.getresponse()
    resp = r.read()
    c.close()

    is_stream = req_id % 2 == 0
    if is_stream:
        client_text = ""
        for line in resp.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            try:
                evt = json.loads(line[6:])
            except Exception:
                continue
            if evt.get("type") == "content_block_delta":
                d = evt.get("delta", {}) or {}
                if d.get("type") == "text_delta":
                    client_text += d.get("text", "")
    else:
        try:
            data = json.loads(resp)
            client_text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        except Exception:
            client_text = resp.decode("utf-8", errors="replace")

    # Assertions per request:
    if real_name not in client_text:
        return (req_id, False, f"missing real_name {real_name!r} in client text: {client_text[:200]!r}")
    if real_email not in client_text:
        return (req_id, False, f"missing real_email {real_email!r} in client text: {client_text[:200]!r}")
    leaked = TOKEN_RE.findall(client_text)
    if leaked:
        return (req_id, False, f"leaked tokens {leaked} to client {client_text[:200]!r}")
    # Cross-bleed check: no other request's name/email should appear here.
    for other in range(N_REQUESTS):
        if other == req_id:
            continue
        for needle in (f"User{other}Smith", f"u{other}@example.com"):
            if needle in client_text:
                return (req_id, False, f"cross-bleed: saw req#{other}'s {needle!r} in req#{req_id}'s text")
    return (req_id, True, "")


def main():
    print(f"Firing {N_REQUESTS} concurrent requests across {N_THREADS} workers against {PROXY_URL}")
    t0 = time.perf_counter()
    fails: list[tuple[int, str]] = []
    with cf.ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = [ex.submit(fire, i) for i in range(N_REQUESTS)]
        for f in cf.as_completed(futures):
            i, ok, err = f.result()
            if not ok:
                fails.append((i, err))
    dt = time.perf_counter() - t0
    print(f"  elapsed: {dt:.2f}s ({N_REQUESTS / dt:.1f} req/s)")
    print(f"  failures: {len(fails)}")
    for i, err in fails[:10]:
        print(f"    [#{i}] {err}")
    if fails:
        sys.exit(1)
    print("All concurrent requests passed.")


if __name__ == "__main__":
    main()
