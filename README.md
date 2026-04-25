# Claude PII Proxy

Round-trip PII redaction for Claude Code. A transparent local proxy sits
between Claude Code and the Anthropic API:

* **Outbound** — names, emails, phones, addresses, URLs, dates, account
  numbers, and secrets are detected by [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)
  and replaced with deterministic short tokens like `name-a3f2b1`.
* **Inbound** — when Claude's response (streaming or not) refers to those
  tokens, they are restored to the real values before the response reaches
  Claude Code. From the user's perspective, nothing changed.

The Anthropic API never sees the real values; Claude operates on the tokens
the same way it would operate on real names. A persistent map keeps the
mapping deterministic across requests and sessions.

## How it works

```
Claude Code  ──►  PII Proxy (:5599)  ──►  Anthropic API
                       │  redact text fields, persist real ↔ token map
                       │
                       ◄── restore tokens in SSE stream / JSON response
```

* PII detection: `openai/privacy-filter` (1.5B total / 50M active params,
  Apache-2.0). Runs locally via **onnxruntime** — no PyTorch.
* Inference: defaults to int8 `model_quantized.onnx` on CPU, which is the
  fastest variant for this MoE model (~45 ms / single, ~6 ms / string when
  batched). Strings are length-bucketed (power-of-two padded length) so a
  single 2k-token system prompt doesn't drag short messages along with it.
* Caching: stable parts of messages survive across conversation turns
  even when Claude Code refreshes `<system-reminder>` timestamps each
  request. After warmup, a 10-message conversation reduces from ~10 cache
  misses per turn to ~1.
* Override quantization with `PII_PROXY_QUANT` (e.g. `model_q4f16.onnx` for
  ~770 MB on disk vs 1.5 GB int8, at ~10× higher latency).
* Override providers with `PII_PROXY_PROVIDERS=CUDAExecutionProvider,...`
  (CUDA is preferred when present; DirectML is intentionally last because it
  silently produces zero predictions on this model's q4 graphs).
* Tokens: `<category>-<6 hex>` where category is one of `name`, `email`,
  `phone`, `address`, `url`, `date`, `account`, `secret`. Hash is HMAC-SHA256
  with a per-machine secret in `~/.claude/pii-proxy.secret`.
* Map: `~/.claude/pii-proxy-map.json` (token → real value). Atomic writes.

## Install

### Windows

```powershell
git clone https://github.com/NodeNestor/claude-pii-proxy
cd claude-pii-proxy
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Linux / macOS

```bash
git clone https://github.com/NodeNestor/claude-pii-proxy
cd claude-pii-proxy
./install.sh
```

The installer:

1. Creates `proxy/.venv` and installs `onnxruntime`, `tokenizers`,
   `huggingface_hub`, `numpy`.
2. Sets `ANTHROPIC_BASE_URL=http://127.0.0.1:5599` in `~/.claude/settings.json`
   (chains transparently if another proxy is already configured).
3. Symlinks the repo into `~/.claude/plugins/pii-proxy` so a `SessionStart`
   hook keeps the proxy running.

Restart Claude Code — first request will download the model
(quantized variant, typically 50–150MB).

### GPU acceleration (optional)

```bash
proxy/.venv/bin/python -m pip uninstall -y onnxruntime
proxy/.venv/bin/python -m pip install onnxruntime-gpu     # NVIDIA / CUDA
# or:
proxy/.venv/bin/python -m pip install onnxruntime-directml # Windows / DirectML
```

## Configuration

All env vars can be set in `~/.claude/settings.json` under `env`:

| Var | Default | Meaning |
|---|---|---|
| `PII_PROXY_PORT` | `5599` | Port to listen on |
| `PII_PROXY_UPSTREAM` | `https://api.anthropic.com` | Upstream URL (auto-set when chaining) |
| `PII_PROXY_MODEL` | `openai/privacy-filter` | HF model id |
| `PII_PROXY_QUANT` | `model_quantized.onnx` (int8) | Pin a specific ONNX variant — e.g. `model_q4f16.onnx` for smallest disk |
| `PII_PROXY_PROVIDERS` | (auto) | Comma-separated ORT providers, e.g. `CUDAExecutionProvider,CPUExecutionProvider` |
| `PII_PROXY_THREADS` | `min(8, cpu/2)` | ORT `intra_op_num_threads`. 8 is the sweet spot on most CPUs for this model |
| `PII_PROXY_MIN_SCORE` | `0.5` | Minimum classifier confidence to redact |
| `PII_PROXY_WARMUP` | `1` | Pre-load the model at startup |

## Health & debug

```bash
curl http://127.0.0.1:5599/health
curl http://127.0.0.1:5599/debug/map
```

Logs:

* `~/.claude/pii-proxy.log` — proxy stdout/stderr
* `~/.claude/pii-proxy-debug.log` — request lifecycle
* `~/.claude/pii-proxy-hook.log` — SessionStart hook log

## Scope of redaction

Conservative by default — only **text** content blocks and **tool_result** text
get redacted. Tool definitions, tool_use input dicts, model IDs, and other
metadata are left alone, so file paths and command arguments aren't
mangled. (A name embedded inside a path like `C:\Users\ludde` will not be
redacted there, because the substitution would break the path.)

## Limitations

* **Detection misses are possible.** The model card warns about uncommon
  names, non-English text, and novel credential formats. For high-stakes
  redaction, pair with a regex pass for known token formats (`sk-ant-*`,
  AWS keys, etc.) and review.
* **Reverse map loss = unrecoverable.** Deleting `pii-proxy-map.json` orphans
  any tokens still alive in conversation history — they will appear in chat
  as `name-a3f2b1` instead of being restored. The HMAC keeps minting
  consistent for new occurrences though.
* **Streaming partial_json deltas** are restored with a small per-block tail
  buffer (24 chars). Tokens that span deltas are restored correctly; output
  is delayed by ~24 chars worth of streaming.
* **Not a compliance tool.** Use as part of a privacy-by-design approach,
  not as standalone anonymization for regulated workloads.

## Layout

```
proxy/
  server.py        — HTTP proxy + SSE stream rewriter
  redactor.py      — payload walker, token map, HMAC minting
  engine.py        — ONNX runtime + tokenizers + BIOES decoder
  requirements.txt
hooks/
  hooks.json       — SessionStart hook
  start-proxy.ps1  — Windows starter
  start-proxy.sh   — POSIX starter
.claude-plugin/
  plugin.json
install.ps1
install.sh
```

## License

MIT. See [LICENSE](LICENSE).

The `openai/privacy-filter` model is Apache-2.0; this repo does not
redistribute it — it is downloaded on first use from Hugging Face.
