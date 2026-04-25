---
description: DANGEROUS — wipe the token <-> real-value map. Any token still alive in past conversation history will no longer be restorable; Claude will keep responding with the placeholder. Use only after wiping the conversation too.
disable-model-invocation: true
allowed-tools: Bash
---

echo "This will permanently drop all token<->value mappings."
echo "Past conversations referencing those tokens will show placeholders forever."
echo "Re-run with the --confirm flag to proceed:"
echo "  /pii-clear-tokens --confirm"
echo
case "$ARGUMENTS" in
  *--confirm*)
    curl -s -X POST "http://127.0.0.1:${PII_PROXY_PORT:-5599}/admin/tokens/clear" \
        || echo "PII proxy is not running on port ${PII_PROXY_PORT:-5599}."
    ;;
  *)
    echo "(no --confirm given; nothing was changed)"
    ;;
esac
