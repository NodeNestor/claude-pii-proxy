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

# Wire ourselves into Claude Code via HTTPS_PROXY + NODE_EXTRA_CA_CERTS (NOT
# ANTHROPIC_BASE_URL, which trips the Remote Control / GrowthBook gate). CA
# generation, HTTPS_PROXY single-owner ownership, chaining with rolling-context,
# plugin defaults and stale-base_url cleanup all live in wire.py — one tested
# implementation shared with the PowerShell hook and rolling-context.
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
PY_CMD=""
if command -v python3 >/dev/null 2>&1; then PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then PY_CMD="python"
fi

if [ -n "$PY_CMD" ]; then
    WIRE_OUT=$("$PY_CMD" "$PROXY_DIR/wire.py" --name pii-proxy --settings "$SETTINGS_FILE" 2>&1)
    if [ $? -eq 0 ]; then
        while IFS= read -r line; do [ -n "$line" ] && log "wire:$line"; done <<< "$WIRE_OUT"
    else
        log "WARNING: wire.py failed to update settings.json: $WIRE_OUT"
    fi
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
