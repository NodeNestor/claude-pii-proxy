FROM python:3.12-slim

WORKDIR /app

# System deps for tokenizers (Rust binding wheel works without these on glibc,
# but keeping ca-certs for HTTPS to HF/Anthropic.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY proxy/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY proxy/ ./proxy/
COPY .claude-plugin/ ./.claude-plugin/

# Pre-download quantized model weights so the container is self-contained
# and the first request doesn't pay the download cost. Falls back through
# q4 -> int8 -> fp16 -> fp32 priority order from engine.py.
RUN python -c "import sys; sys.path.insert(0, '/app/proxy'); \
    from engine import _download_model_assets; \
    paths = _download_model_assets('openai/privacy-filter'); \
    print('Pre-downloaded:', paths[2])"

ENV PII_PROXY_PORT=5599 \
    PII_PROXY_MIN_SCORE=0.5 \
    PII_PROXY_MODEL=openai/privacy-filter \
    PII_PROXY_WARMUP=1

EXPOSE 5599

CMD ["python", "proxy/server.py"]
