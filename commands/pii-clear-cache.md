---
description: Drop the cached PII detection results. Safe — keeps token map so past tokens still restore. The cache repopulates on use; first-pass detection on each unique chunk runs again.
disable-model-invocation: true
allowed-tools: Bash
---

curl -s -X POST "http://127.0.0.1:${PII_PROXY_PORT:-5599}/admin/cache/clear" || echo "PII proxy is not running on port ${PII_PROXY_PORT:-5599}."
