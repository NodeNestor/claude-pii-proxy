---
description: Show how to change PII proxy settings — quantization, providers (CPU/GPU), threads, cache location. Settings live in ~/.claude/settings.json and require a Claude Code session restart.
disable-model-invocation: true
allowed-tools: Bash
---

PORT="${PII_PROXY_PORT:-5599}"
echo "PII proxy live config:"
echo "----------------------"
curl -s "http://127.0.0.1:${PORT}/stats" 2>/dev/null | grep -E "Model|Quant|Min score|Providers|Threads|Inputs|Path|Mappings|Size|In memory" || echo "(proxy not reachable)"
echo
echo "How to change settings"
echo "----------------------"
echo "Edit ~/.claude/settings.json — under \"env\" you can set:"
echo
echo "  PII_PROXY_QUANT       Pin a specific ONNX variant. Options:"
echo "                        model_quantized.onnx  (int8, default — fastest CPU)"
echo "                        model_q4f16.onnx      (smallest disk, ~770 MB)"
echo "                        model_q4.onnx"
echo "                        model_fp16.onnx"
echo "                        model.onnx            (fp32, biggest)"
echo
echo "  PII_PROXY_PROVIDERS   Comma-separated ORT providers, in priority order."
echo "                        Examples:"
echo "                          CUDAExecutionProvider,CPUExecutionProvider   (NVIDIA, requires onnxruntime-gpu)"
echo "                          CoreMLExecutionProvider,CPUExecutionProvider (Apple Silicon)"
echo "                          DmlExecutionProvider,CPUExecutionProvider    (Win/AMD/Intel — currently broken on this model's q4 graph)"
echo
echo "  PII_PROXY_THREADS     intra_op_num_threads for CPU. Default min(8, cpu/2)."
echo
echo "  PII_PROXY_MIN_SCORE   Detection confidence threshold (0..1). Default 0.5."
echo
echo "  PII_PROXY_PORT        Listen port. Default 5599."
echo
echo "Cache locations"
echo "---------------"
echo "  Span cache : ~/.claude/pii-proxy-spans.json   (delete to force re-detection)"
echo "  Token map  : ~/.claude/pii-proxy-map.json     (DO NOT delete in active sessions)"
echo "  HMAC seed  : ~/.claude/pii-proxy.secret"
echo
echo "After editing settings.json, start a new Claude Code session for changes to take effect."
