"""
Real end-to-end test using the user's Claude Code subscription auth.

Starts the PII proxy pointing at api.anthropic.com, then runs `claude -p`
through it. Verifies that:

  1. The proxy received the request, detected PII, and minted tokens.
  2. Anthropic accepted the redacted body (subscription auth passes through).
  3. The response Claude Code prints — after the proxy restores it —
     contains the *original* PII, with no tokens leaking to the client.

This is the same pattern claude-rolling-context uses for its real-API e2e.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time

PROXY_PORT = int(os.environ.get("PII_PROXY_PORT", "5599"))
PROXY_PY = os.environ.get(
    "PII_PROXY_PY",
    str(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "proxy", ".venv", "Scripts", "python.exe")))
    if os.name == "nt"
    else str(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "proxy", ".venv", "bin", "python"))),
)
PROXY_SCRIPT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "proxy", "server.py"))
TOKEN_RE = re.compile(
    r"\b(name|email|phone|address|url|date|account|secret)-[0-9a-f]{6}\b"
)


def start_proxy() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PII_PROXY_PORT"] = str(PROXY_PORT)
    env["PII_PROXY_UPSTREAM"] = "https://api.anthropic.com"
    env.pop("ANTHROPIC_BASE_URL", None)  # avoid recursion
    print(f"[TEST] Starting proxy with {PROXY_PY} {PROXY_SCRIPT}")
    proc = subprocess.Popen(
        [PROXY_PY, PROXY_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def stream():
        for line in proc.stdout:
            print("[PROXY]", line.decode(errors="replace").rstrip())

    threading.Thread(target=stream, daemon=True).start()

    for _ in range(60):
        try:
            c = http.client.HTTPConnection("127.0.0.1", PROXY_PORT, timeout=2)
            c.request("GET", "/health")
            r = c.getresponse()
            body = json.loads(r.read())
            c.close()
            if r.status == 200:
                print(f"[TEST] proxy ready: {body}")
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("proxy failed to start within 30s")


def get_health() -> dict:
    c = http.client.HTTPConnection("127.0.0.1", PROXY_PORT, timeout=5)
    c.request("GET", "/health")
    r = c.getresponse()
    data = json.loads(r.read())
    c.close()
    return data


def _settings_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".claude", "settings.json")


def patch_settings_for_proxy(proxy_url: str) -> dict:
    """Set ANTHROPIC_BASE_URL in ~/.claude/settings.json so the claude CLI
    routes through our proxy. Returns the prior env block for restoration."""
    path = _settings_path()
    settings: dict = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            settings = json.load(f)
    prior_env = dict(settings.get("env", {}) or {})
    env = settings.setdefault("env", {})
    env["ANTHROPIC_BASE_URL"] = proxy_url
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    return prior_env


def restore_settings_env(prior_env: dict) -> None:
    path = _settings_path()
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    settings["env"] = prior_env
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")


def run_claude(prompt: str, model: str = "haiku") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{PROXY_PORT}"
    return subprocess.run(
        ["claude", "-p", "--model", model, prompt],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def main():
    print("=" * 60)
    print(" PII proxy real-API e2e — Claude Code subscription auth")
    print("=" * 60)

    creds = os.path.expanduser("~/.claude/.credentials.json")
    if not os.path.exists(creds):
        print(f"[TEST] WARNING: no credentials at {creds}")

    proxy = start_proxy()
    prior_env = patch_settings_for_proxy(f"http://127.0.0.1:{PROXY_PORT}")
    print(f"[TEST] settings.json patched; prior env keys: {list(prior_env.keys())}")
    # Wipe persistent map so we can verify *new* tokens get minted on this run
    map_path = os.path.expanduser("~/.claude/pii-proxy-map.json")
    if os.path.exists(map_path):
        os.replace(map_path, map_path + ".bak")
        print(f"[TEST] backed up existing map to {map_path}.bak")
    try:
        before = get_health()["mappings"]
        prompt = (
            "Repeat the following sentence back exactly, nothing else: "
            "'My name is Ludde Oland and you can email me at "
            "ludde.oland@gmail.com.'"
        )
        print(f"\n[TEST] prompt: {prompt}\n")
        result = run_claude(prompt)
        print(f"[TEST] claude exit code: {result.returncode}")
        print(f"[TEST] stdout: {result.stdout[:600]!r}")
        if result.stderr:
            print(f"[TEST] stderr: {result.stderr[:300]!r}")

        after = get_health()["mappings"]
        delta = after - before
        leaked = TOKEN_RE.findall(result.stdout)
        contains_real = "Ludde" in result.stdout

        print("\n--- assertions ---")
        ok = True
        for cond, label in [
            (result.returncode == 0, "claude -p exited cleanly"),
            (delta > 0, f"proxy minted tokens (mapping delta={delta})"),
            (contains_real, "client output contains restored 'Ludde'"),
            (len(leaked) == 0, f"no tokens leaked to client ({leaked})"),
        ]:
            status = "PASS" if cond else "FAIL"
            print(f"  {status}: {label}")
            ok = ok and cond

        if not ok:
            sys.exit(1)
        print("\nReal-API e2e test passed: Claude API saw only tokens, user sees real PII.")

    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=5)
        except Exception:
            proxy.kill()
        try:
            restore_settings_env(prior_env)
            print("[TEST] settings.json env restored")
        except Exception as e:
            print(f"[TEST] WARNING: could not restore settings env: {e}")
        # Restore map backup if we made one
        map_path = os.path.expanduser("~/.claude/pii-proxy-map.json")
        if os.path.exists(map_path + ".bak"):
            try:
                os.replace(map_path + ".bak", map_path)
                print("[TEST] restored prior pii-proxy map")
            except Exception:
                pass


if __name__ == "__main__":
    main()
