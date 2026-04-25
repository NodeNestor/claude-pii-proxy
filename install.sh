#!/usr/bin/env bash
# Install the PII Proxy plugin for Claude Code (Linux/macOS)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_DIR="$SCRIPT_DIR/proxy"
VENV_DIR="$PROXY_DIR/.venv"
PORT="${PII_PROXY_PORT:-5599}"
PROXY_URL="http://127.0.0.1:$PORT"
CLAUDE_DIR="$HOME/.claude"

echo "=== PII Proxy Installer ==="
echo ""

echo "[1/4] Checking Python..."
PY=""
if command -v python3 >/dev/null 2>&1; then PY="python3"
elif command -v python >/dev/null 2>&1; then PY="python"
else
    echo "  ERROR: Python not found. Install Python 3.9+ and try again."
    exit 1
fi
echo "  Found $($PY --version 2>&1)"

echo "[2/4] Creating venv and installing deps..."
if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$PY" -m venv "$VENV_DIR"
fi
VPY="$VENV_DIR/bin/python"
"$VPY" -m pip install --upgrade pip --quiet
"$VPY" -m pip install -r "$PROXY_DIR/requirements.txt" --quiet
echo "  Deps installed in $VENV_DIR"
echo "  (For NVIDIA GPUs install onnxruntime-gpu instead:"
echo "     $VPY -m pip uninstall -y onnxruntime && $VPY -m pip install onnxruntime-gpu)"

echo "[3/4] Configuring Claude Code settings.json..."
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
mkdir -p "$CLAUDE_DIR"

"$PY" - "$SETTINGS_FILE" "$PROXY_URL" "$PORT" <<'PYEOF'
import json, os, sys
settings_file, proxy_url, port = sys.argv[1], sys.argv[2], sys.argv[3]
settings = {}
if os.path.exists(settings_file):
    try:
        with open(settings_file) as f:
            settings = json.load(f)
    except Exception:
        settings = {}
env = settings.setdefault("env", {})
existing = env.get("ANTHROPIC_BASE_URL", "")
if not existing:
    env["ANTHROPIC_BASE_URL"] = proxy_url
    print(f"  Set ANTHROPIC_BASE_URL={proxy_url}")
elif f"127.0.0.1:{port}" not in existing:
    env["PII_PROXY_UPSTREAM"] = existing
    env["ANTHROPIC_BASE_URL"] = proxy_url
    print(f"  Chaining: ANTHROPIC_BASE_URL={proxy_url} -> upstream={existing}")
else:
    print("  ANTHROPIC_BASE_URL already set")
defaults = {
    "PII_PROXY_PORT": "5599",
    "PII_PROXY_MIN_SCORE": "0.5",
    "PII_PROXY_MODEL": "openai/privacy-filter",
    "PII_PROXY_WARMUP": "1",
}
for k, v in defaults.items():
    env.setdefault(k, v)
with open(settings_file, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")
print(f"  Settings written to {settings_file}")
PYEOF

echo "[4/4] Registering Claude Code plugin..."
PLUGIN_LINK="$CLAUDE_DIR/plugins/pii-proxy"
mkdir -p "$CLAUDE_DIR/plugins"
if [ -L "$PLUGIN_LINK" ] || [ -d "$PLUGIN_LINK" ]; then
    rm -rf "$PLUGIN_LINK"
fi
ln -s "$SCRIPT_DIR" "$PLUGIN_LINK"
echo "  Plugin linked at $PLUGIN_LINK"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The proxy will auto-start when you launch Claude Code."
echo "Manual start:  $VPY $PROXY_DIR/server.py"
echo ""
echo "First request will download the openai/privacy-filter ONNX weights"
echo "(quantized variant preferred — usually 50-150MB)."
echo ""
echo "Start a new Claude Code session to activate."
