---
description: Switch the PII proxy to GPU inference (NVIDIA/CUDA, AMD/Intel via DirectML, or Apple CoreML). Reinstalls onnxruntime in the proxy venv and updates settings.json. Restart Claude Code afterwards. Note — for typical short-message Claude Code conversations CPU is usually faster; CUDA mainly helps for cold-start on long pastes / first 1M-context turn (~3-4× speedup measured on RTX 5060 Ti).
disable-model-invocation: true
allowed-tools: Bash
---

VENV_PY=""
if [ -x "${CLAUDE_PLUGIN_ROOT}/proxy/.venv/Scripts/python.exe" ]; then
  VENV_PY="${CLAUDE_PLUGIN_ROOT}/proxy/.venv/Scripts/python.exe"
elif [ -x "${CLAUDE_PLUGIN_ROOT}/proxy/.venv/bin/python" ]; then
  VENV_PY="${CLAUDE_PLUGIN_ROOT}/proxy/.venv/bin/python"
else
  echo "Cannot locate proxy venv at \${CLAUDE_PLUGIN_ROOT}/proxy/.venv"; exit 1
fi

case "${ARGUMENTS:-}" in
  cuda|nvidia)
    PKG="onnxruntime-gpu"
    PROV="CUDAExecutionProvider,CPUExecutionProvider"
    ;;
  directml|dml|amd|intel)
    PKG="onnxruntime-directml"
    PROV="DmlExecutionProvider,CPUExecutionProvider"
    ;;
  coreml|apple|mac)
    PKG="onnxruntime"
    PROV="CoreMLExecutionProvider,CPUExecutionProvider"
    ;;
  cpu|"")
    PKG="onnxruntime"
    PROV="CPUExecutionProvider"
    ;;
  *)
    echo "Usage: /pii-gpu <cuda|directml|coreml|cpu>"
    exit 1
    ;;
esac

echo "Switching to: $PKG  (providers: $PROV)"
"$VENV_PY" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml 2>&1 | tail -3
"$VENV_PY" -m pip install --quiet "$PKG"

# Update settings.json
SETTINGS="$HOME/.claude/settings.json"
"$VENV_PY" - "$SETTINGS" "$PROV" <<'PYEOF'
import json, os, sys
sf, prov = sys.argv[1], sys.argv[2]
s = json.load(open(sf)) if os.path.exists(sf) else {}
s.setdefault("env", {})["PII_PROXY_PROVIDERS"] = prov
json.dump(s, open(sf, "w"), indent=2)
print(f"Wrote PII_PROXY_PROVIDERS={prov} into {sf}")
PYEOF

echo
echo "Restart Claude Code (or kill the proxy PID in ~/.claude/pii-proxy.pid)"
echo "for the new provider to take effect."
