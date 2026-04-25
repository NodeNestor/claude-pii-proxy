# Claude PII Proxy

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Plugin: nestor-plugins](https://img.shields.io/badge/plugin-nestor--plugins-blue)](https://github.com/NodeNestor/nestor-plugins)

Round-trip PII redaction for Claude Code. A transparent local proxy sits
between Claude Code and the Anthropic API:

* **Outbound** — names, emails, phones, addresses, URLs, dates, account
  numbers, and secrets are detected by [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)
  and replaced with deterministic short tokens like `name-a3f2b1`.
* **Inbound** — when Claude's response (streaming or not) refers to those
  tokens, they are restored to the real values before the response reaches
  Claude Code. From your perspective, nothing changed.

The Anthropic API never sees the real values; Claude operates on opaque
typed tokens and can still reason about them (`name-a3f2b1` ≠ `name-9c7e21`,
`email-fad6db` is "an email", `secret-aabbcc` is "a credential"). A
deterministic HMAC keeps the same real value mapping to the same token
across requests, sessions, and proxy restarts.

## Install

### Via the marketplace (recommended)

```
/plugin marketplace add https://github.com/NodeNestor/nestor-plugins
/plugin install pii-proxy
```

Restart your terminal and start a new Claude Code session. The `SessionStart`
hook will:

1. Create `proxy/.venv` and install `onnxruntime`, `tokenizers`, `huggingface_hub`, `numpy`.
2. Set `ANTHROPIC_BASE_URL=http://127.0.0.1:5599` in `~/.claude/settings.json` (chains transparently if rolling-context or another proxy is already configured).
3. Start the proxy in the background.

First request downloads ~1.5 GB of int8 ONNX weights from Hugging Face
(or ~770 MB if you pin `PII_PROXY_QUANT=model_q4f16.onnx`). After that,
detection is local — no network calls outside of the Anthropic upstream
itself.

### Manual install

```bash
git clone https://github.com/NodeNestor/claude-pii-proxy
cd claude-pii-proxy
# Windows
powershell -ExecutionPolicy Bypass -File install.ps1
# Linux / macOS
./install.sh
```

The installer:

1. Creates `proxy/.venv` and installs `onnxruntime`, `tokenizers`, `huggingface_hub`, `numpy`.
2. Sets `ANTHROPIC_BASE_URL=http://127.0.0.1:5599` in `~/.claude/settings.json` (chains transparently if another proxy was already configured).
3. Symlinks the repo into `~/.claude/plugins/pii-proxy` so the `SessionStart` hook keeps the proxy alive.

### Switch to GPU inference

```
/pii-gpu cuda          # NVIDIA + CUDA via onnxruntime-gpu
/pii-gpu directml      # Win/AMD/Intel via DirectX 12 (note: broken on q4 graphs)
/pii-gpu coreml        # Apple Silicon
/pii-gpu cpu           # back to default
```

The command reinstalls onnxruntime in the proxy venv and writes
`PII_PROXY_PROVIDERS` into `settings.json`. Restart Claude Code afterwards.

## How it works

```
Claude Code  ──►  PII Proxy (:5599)  ──►  Anthropic API
                       │  detect & redact text fields
                       │  persist real ↔ token map
                       │
                       ◄── restore tokens in SSE stream / JSON response
```

* **Detection**: `openai/privacy-filter` (1.5 B total / 50 M active params, MoE,
  Apache-2.0). Runs locally via **onnxruntime** — no PyTorch.
* **Default quant**: int8 `model_quantized.onnx`, the fastest variant on CPU
  (~45 ms / single, ~6 ms / string when batched). Override with
  `PII_PROXY_QUANT=model_q4f16.onnx` for smallest disk (~770 MB) at higher
  latency.
* **Length bucketing**: power-of-two padded lengths so a single 2 K-token
  system prompt doesn't drag short messages along to the same forward-pass
  size.
* **Volatile-tag-aware caching**: `<system-reminder>`, `<local-command-stdout>`,
  `<local-command-caveat>` and `<available-deferred-tools>` are split out of
  the cache key. Stable message content survives across turns even though
  Claude Code refreshes the timestamps each request.
* **1M-context support**: inputs >8 000 chars are split into overlapping
  sub-chunks (320-char overlap) and detected per chunk. Each sub-chunk
  hashes/caches independently, so re-sending the same 100 KB code paste is
  0 ms after the first time.
* **Persistent span cache** at `~/.claude/pii-proxy-spans.jsonl`: detection
  results survive proxy restarts and are shared across Claude Code
  sessions. Slow first-time inference happens at most once per unique chunk
  on this machine, **ever**. Append-only JSONL — unbounded by design, no
  eviction; we paid the inference cost once, we keep the result forever.
  Use `/pii-clear-cache` if you ever want to reclaim disk.

## Slash commands

| Command | What it does |
|---|---|
| `/pii-stats` | Show config, providers, cache sizes, mapping count |
| `/pii-config` | Show every tunable + how to change it |
| `/pii-clear-cache` | Drop the span cache (token map kept — safe, just slows the next request on already-seen content) |
| `/pii-clear-tokens --confirm` | DANGEROUS — wipe token↔value map. Past tokens in conversation history will stop restoring |
| `/pii-gpu cuda\|directml\|coreml\|cpu` | Switch ORT package + provider |

## Configuration

Every tunable can be set in `~/.claude/settings.json` under `env`:

| Var | Default | Meaning |
|---|---|---|
| `PII_PROXY_PORT` | `5599` | Listen port |
| `PII_PROXY_UPSTREAM` | `https://api.anthropic.com` | Upstream URL (auto-set when chaining behind another proxy) |
| `PII_PROXY_MODEL` | `openai/privacy-filter` | HF model id |
| `PII_PROXY_QUANT` | `model_quantized.onnx` (int8) | Pin a specific ONNX variant — e.g. `model_q4f16.onnx` for smallest disk |
| `PII_PROXY_PROVIDERS` | auto | Comma-separated ORT providers in priority order, e.g. `CUDAExecutionProvider,CPUExecutionProvider` |
| `PII_PROXY_THREADS` | `min(8, cpu/2)` | ORT `intra_op_num_threads`. 8 is the sweet spot on most CPUs for this model |
| `PII_PROXY_MIN_SCORE` | `0.5` | Minimum classifier confidence to redact |
| `PII_PROXY_WARMUP` | `1` | Pre-load the model at startup |

## Storage layout

| Path | What | Safe to delete? |
|---|---|---|
| `~/.claude/pii-proxy-spans.jsonl` | Cached PII spans — append-only JSONL, **unbounded**. Each unique chunk takes ~200–1000 B. Restoring from this file is essentially free, so we never evict; large files are not a performance problem — appends are O(new entries), not O(file size). | **Yes** — `/pii-clear-cache` |
| `~/.claude/pii-proxy-map.json` | Token ↔ real-value map | **Not while past conversations exist** — old `name-XXXXXX` tokens become unrestorable forever |
| `~/.claude/pii-proxy.secret` | HMAC seed for deterministic tokens | If deleted, future tokens differ; old map still works for old tokens |
| `~/.claude/pii-proxy.log` | Proxy stdout/stderr | Yes |
| `~/.claude/pii-proxy-debug.log` | Request lifecycle | Yes |

## Endpoints

```
GET  /health                     JSON: status, upstream, model, mapping count
GET  /stats                      Human-readable config + cache report
POST /admin/cache/clear          Wipe span cache (in-memory + disk)
POST /admin/tokens/clear         Wipe token map (in-memory + disk) — destructive
GET  /debug/map                  Counts only, never reveals real values
```

All other paths under `/v1/*` are forwarded to the Anthropic upstream after redaction.

## Scope of redaction

Conservative by default — only **text** content blocks and **tool_result** text
get redacted. Tool definitions, tool_use input dicts, model IDs, and other
metadata are left alone so file paths and command arguments aren't mangled.
A name embedded inside a path like `C:\Users\ludde\foo.py` will *not* be
redacted there, because the substitution would break the path.

## Speed (CPU, int8, defaults)

| Scenario | Latency |
|---|---|
| Cold model load | ~10 s (one-time per process; `SessionStart` hook hides it) |
| Single short string | ~45 ms |
| Batch of 16 short strings | ~89 ms (5.5 ms / string) |
| Batch of 32 short strings | ~280 ms (8.8 ms / string) |
| 25 KB doc — first time ever | ~3.8 s |
| 25 KB doc — after proxy restart | **0 ms** (loads from disk cache) |
| Warmed-up 10-turn conversation, per turn | ~175 ms |
| 1M-token-equivalent conversation, **subsequent turns** | **0 ms** |

GPU via `/pii-gpu cuda` typically delivers a 10–30× cold-path speedup on
NVIDIA hardware.

## Limitations

* **Detection misses are possible.** The model card warns about uncommon
  names, non-English text, and novel credential formats. For high-stakes
  redaction, pair with a regex pass for known token formats (`sk-ant-*`,
  AWS keys, etc.) and review.
* **Reverse map loss = unrecoverable.** Deleting `pii-proxy-map.json`
  orphans any tokens still alive in conversation history — they will
  appear in chat as `name-a3f2b1` instead of being restored. The HMAC
  keeps minting consistent for new occurrences.
* **Streaming token restoration** uses a small per-block tail buffer.
  Tokens that span SSE delta boundaries are still restored correctly;
  output is delayed by at most one token's worth of streaming (~16 chars).
* **Not a compliance tool.** Use as part of a privacy-by-design approach,
  not as standalone anonymization for regulated workloads.

## Layout

```
proxy/
  server.py        HTTP proxy + SSE stream rewriter + admin endpoints
  redactor.py      payload walker, token map, HMAC minting, streaming restorer
  engine.py        onnxruntime + tokenizers, length bucketing, BIOES decoder
  requirements.txt
commands/          plugin slash commands (/pii-stats, /pii-gpu, ...)
hooks/             SessionStart hook + Windows/POSIX starters
test/
  mock_anthropic.py    capture-and-echo upstream for round-trip assertions
  test_proxy.py        18-assertion mock e2e (streaming + non-streaming)
  e2e_real.py          real-API e2e using your Claude Code subscription
  Dockerfile.mock, Dockerfile.test
.claude-plugin/plugin.json
docker-compose.test.yml
install.ps1, install.sh
```

## License

MIT. See [LICENSE](LICENSE).

The `openai/privacy-filter` model is Apache-2.0; this repo does not
redistribute it — it is downloaded on first use from Hugging Face.
