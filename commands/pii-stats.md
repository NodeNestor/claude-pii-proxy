---
description: Show PII proxy stats — model, providers, cache sizes, mapping count
disable-model-invocation: true
allowed-tools: Bash
---

curl -s "http://127.0.0.1:${PII_PROXY_PORT:-5599}/stats" || echo "PII proxy is not running on port ${PII_PROXY_PORT:-5599}. Start a new Claude Code session to launch it."
