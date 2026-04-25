#!/usr/bin/env bash
# Ensure the PII proxy is running (Linux/macOS)
# Uses the venv created by install.sh (proxy/.venv/bin/python)

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_DIR="$SCRIPT_DIR/../proxy"
VENV_PY="$PROXY_DIR/.venv/bin/python"
CLAUDE_DIR="$HOME/.claude"
PID_FILE="$CLAUDE_DIR/pii-proxy.pid"
VER_FILE="$CLAUDE_DIR/pii-proxy.version"
HOOK_LOG="$CLAUDE_DIR/pii-proxy-hook.log"
PROXY_LOG="$CLAUDE_DIR/pii-proxy.log"
PORT="${PII_PROXY_PORT:-5599}"
PROXY_URL="http://127.0.0.1:$PORT"
PLUGIN_JSON="$SCRIPT_DIR/../.claude-plugin/plugin.json"

mkdir -p "$CLAUDE_DIR"

log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $*" >> "$HOOK_LOG"
}

log "Hook started. PROXY_DIR=$PROXY_DIR"

CURRENT_VERSION="unknown"
if [ -f "$PLUGIN_JSON" ] && command -v python3 >/dev/null 2>&1; then
    CURRENT_VERSION=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('version','unknown'))" "$PLUGIN_JSON" 2>/dev/null || echo "unknown")
fi

# Update settings.json
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
PY_CMD=""
if command -v python3 >/dev/null 2>&1; then PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then PY_CMD="python"
fi

if [ -n "$PY_CMD" ]; then
    "$PY_CMD" - "$SETTINGS_FILE" "$PROXY_URL" "$PORT" <<'PYEOF' 2>>"$HOOK_LOG"
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
elif f"127.0.0.1:{port}" not in existing:
    env["PII_PROXY_UPSTREAM"] = existing
    env["ANTHROPIC_BASE_URL"] = proxy_url
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
PYEOF
fi

# Pick interpreter
if [ -x "$VENV_PY" ]; then
    PYTHON="$VENV_PY"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi
log "Using interpreter: $PYTHON"

# Already running?
if [ -f "$PID_FILE" ]; then
    SAVED_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$SAVED_PID" ] && kill -0 "$SAVED_PID" 2>/dev/null; then
        RUNNING_VERSION=""
        [ -f "$VER_FILE" ] && RUNNING_VERSION=$(cat "$VER_FILE" 2>/dev/null || echo "")
        if [ "$RUNNING_VERSION" = "$CURRENT_VERSION" ]; then
            log "Proxy already running (PID $SAVED_PID, v$RUNNING_VERSION)"
            exit 0
        fi
        log "Version changed ($RUNNING_VERSION -> $CURRENT_VERSION), restarting (PID $SAVED_PID)"
        kill "$SAVED_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE" "$VER_FILE"
fi

log "Starting proxy with $PYTHON ..."
cd "$PROXY_DIR"
nohup "$PYTHON" server.py >>"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
echo -n "$PROXY_PID" > "$PID_FILE"
echo -n "$CURRENT_VERSION" > "$VER_FILE"
log "Proxy started with PID $PROXY_PID (v$CURRENT_VERSION)"

exit 0
