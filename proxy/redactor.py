"""
PII Redactor — detects PII via openai/privacy-filter, replaces with deterministic
tokens (e.g. "Ludde" -> "name-a3f2b1"), maintains a reverse map so responses
containing those tokens are restored before reaching the client.

Tokens are deterministic via HMAC(local_secret, real_value) so the same real
value always produces the same token across requests and sessions. The
reverse map is persisted to disk so tokens survive proxy restarts.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets as pysecrets
import threading

from engine import PrivacyFilterEngine

log = logging.getLogger("pii-proxy.redactor")

CLAUDE_DIR = os.path.join(os.path.expanduser("~"), ".claude")
SECRET_PATH = os.path.join(CLAUDE_DIR, "pii-proxy.secret")
MAP_PATH = os.path.join(CLAUDE_DIR, "pii-proxy-map.json")

# Matches any token we may have minted, e.g. "name-a3f2b1"
TOKEN_PATTERN = re.compile(
    r"\b(name|email|phone|address|url|date|account|secret)-[0-9a-f]{6}\b"
)


def _ensure_claude_dir() -> None:
    os.makedirs(CLAUDE_DIR, exist_ok=True)


def _load_or_create_secret() -> bytes:
    _ensure_claude_dir()
    if os.path.exists(SECRET_PATH):
        with open(SECRET_PATH, "rb") as f:
            data = f.read().strip()
            if data:
                return data
    data = pysecrets.token_hex(32).encode()
    with open(SECRET_PATH, "wb") as f:
        f.write(data)
    try:
        os.chmod(SECRET_PATH, 0o600)
    except OSError:
        pass
    log.info(f"[INIT] Generated new HMAC secret at {SECRET_PATH}")
    return data


class TokenMap:
    """Persistent bidirectional map between real values and minted tokens."""

    def __init__(self, path: str = MAP_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._token_to_real: dict[str, str] = {}
        self._real_to_token: dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for token, real in data.items():
                self._token_to_real[token] = real
                self._real_to_token[real] = token
            log.info(f"[INIT] Loaded {len(self._token_to_real)} mapping(s) from {self.path}")
        except Exception as e:
            log.warning(f"[INIT] Could not load map ({e}); starting fresh")

    def _save_locked(self) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._token_to_real, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._dirty = False
        except Exception as e:
            log.warning(f"[SAVE] Could not save map: {e}")

    def get_token(self, real: str) -> str | None:
        with self._lock:
            return self._real_to_token.get(real)

    def add(self, real: str, token: str) -> None:
        with self._lock:
            if real in self._real_to_token:
                return
            self._token_to_real[token] = real
            self._real_to_token[real] = token
            self._save_locked()

    def lookup(self, token: str) -> str | None:
        with self._lock:
            return self._token_to_real.get(token)

    def __len__(self) -> int:
        return len(self._token_to_real)


class Redactor:
    """Detects PII spans, mints deterministic tokens, walks Anthropic API payloads."""

    def __init__(
        self,
        model_id: str = "openai/privacy-filter",
        min_score: float = 0.5,
        cache_size: int = 4096,
        engine: PrivacyFilterEngine | None = None,
    ) -> None:
        self.model_id = model_id
        self.min_score = min_score
        self._cache_size = cache_size
        self._secret = _load_or_create_secret()
        self.map = TokenMap()
        self._engine = engine or PrivacyFilterEngine(model_id=model_id, min_score=min_score)
        self._span_cache: dict[str, list[dict]] = {}

    def warmup(self) -> None:
        """Pre-load the ONNX model so first request doesn't pay the load cost."""
        self._engine.load()

    # ---------- token minting ----------

    def _make_token(self, real: str, category: str) -> str:
        digest = hmac.new(self._secret, real.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{category}-{digest[:6]}"

    # ---------- detection ----------

    def _detect_spans(self, text: str) -> list[dict]:
        if not text or len(text) < 3:
            return []
        # quick reject: no letters -> no names/emails/etc.
        if not any(c.isalpha() for c in text):
            return []
        h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        cached = self._span_cache.get(h)
        if cached is not None:
            return cached

        try:
            spans = self._engine.detect(text)
        except Exception as e:
            log.warning(f"[DETECT] engine failed on {len(text)}-char input: {e}")
            spans = []

        # crude LRU: trim oldest 25% if over cap
        if len(self._span_cache) >= self._cache_size:
            for k in list(self._span_cache.keys())[: self._cache_size // 4]:
                self._span_cache.pop(k, None)
        self._span_cache[h] = spans
        return spans

    # ---------- redact / restore ----------

    def redact_text(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        spans = self._detect_spans(text)
        if not spans:
            return text
        spans = sorted(spans, key=lambda s: s["start"])
        parts: list[str] = []
        cursor = 0
        for s in spans:
            if s["start"] < cursor:
                continue  # overlap; skip
            real = s["real"]
            token = self.map.get_token(real)
            if token is None:
                token = self._make_token(real, s["category"])
                self.map.add(real, token)
            parts.append(text[cursor : s["start"]])
            parts.append(token)
            cursor = s["end"]
        parts.append(text[cursor:])
        return "".join(parts)

    def restore_text(self, text: str) -> str:
        if not isinstance(text, str) or "-" not in text:
            return text

        def _sub(m: re.Match) -> str:
            token = m.group(0)
            real = self.map.lookup(token)
            return real if real is not None else token

        return TOKEN_PATTERN.sub(_sub, text)

    # ---------- payload walkers ----------

    def redact_request_payload(self, payload: dict) -> dict:
        """In-place redact relevant text fields of an Anthropic /v1/messages payload.

        Collects all text fields, runs detection in a single batched forward
        pass, then writes the redacted text back. This is dramatically faster
        than calling the model once per text field. Conservative scope: only
        rewrites text-content blocks and tool_result text — leaves tool_use
        inputs, tool definitions, and metadata untouched so we don't break
        file paths, command args, or schema strings.
        """
        # (1) Walk the payload and collect references to every text field we
        #     plan to redact, alongside its current text and a setter.
        targets: list[tuple[str, "callable[[str], None]"]] = []

        def _collect(text: str, setter):
            if isinstance(text, str) and text:
                targets.append((text, setter))

        sys = payload.get("system")
        if isinstance(sys, str):
            _collect(sys, lambda v: payload.__setitem__("system", v))
        elif isinstance(sys, list):
            for idx, b in enumerate(sys):
                if isinstance(b, dict) and b.get("type") == "text":
                    _collect(b.get("text", ""), lambda v, _b=b: _b.__setitem__("text", v))

        for msg in payload.get("messages", []) or []:
            content = msg.get("content")
            if isinstance(content, str):
                _collect(content, lambda v, _m=msg: _m.__setitem__("content", v))
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    _collect(block.get("text", ""), lambda v, _b=block: _b.__setitem__("text", v))
                elif btype == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        _collect(inner, lambda v, _b=block: _b.__setitem__("content", v))
                    elif isinstance(inner, list):
                        for ib in inner:
                            if isinstance(ib, dict) and ib.get("type") == "text":
                                _collect(ib.get("text", ""), lambda v, _ib=ib: _ib.__setitem__("text", v))

        if not targets:
            return payload

        # (2) Resolve cached spans first, batch-detect the cache misses.
        texts = [t for t, _ in targets]
        cached_spans: list[list[dict] | None] = []
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for i, text in enumerate(texts):
            if not text or len(text) < 3 or not any(c.isalpha() for c in text):
                cached_spans.append([])
                continue
            h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
            cached = self._span_cache.get(h)
            if cached is not None:
                cached_spans.append(cached)
            else:
                cached_spans.append(None)
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            try:
                batch_spans = self._engine.detect_batch(miss_texts)
            except Exception as e:
                log.warning(f"[DETECT] batch failed: {e}; falling back to per-text")
                batch_spans = [self._engine.detect(t) for t in miss_texts]
            # Cache and slot back in
            if len(self._span_cache) >= self._cache_size:
                for k in list(self._span_cache.keys())[: self._cache_size // 4]:
                    self._span_cache.pop(k, None)
            for j, idx in enumerate(miss_indices):
                spans = batch_spans[j]
                cached_spans[idx] = spans
                h = hashlib.sha1(miss_texts[j].encode("utf-8", errors="replace")).hexdigest()
                self._span_cache[h] = spans

        # (3) Apply redactions back to each target.
        for (text, setter), spans in zip(targets, cached_spans):
            if not spans:
                continue
            redacted = self._apply_spans(text, spans)
            if redacted != text:
                setter(redacted)
        return payload

    def _apply_spans(self, text: str, spans: list[dict]) -> str:
        if not spans:
            return text
        spans = sorted(spans, key=lambda s: s["start"])
        parts: list[str] = []
        cursor = 0
        for s in spans:
            if s["start"] < cursor:
                continue
            real = s["real"]
            token = self.map.get_token(real)
            if token is None:
                token = self._make_token(real, s["category"])
                self.map.add(real, token)
            parts.append(text[cursor : s["start"]])
            parts.append(token)
            cursor = s["end"]
        parts.append(text[cursor:])
        return "".join(parts)

    def restore_in_obj(self, obj):
        """Walk arbitrary JSON-like obj and restore tokens in every string."""
        if isinstance(obj, str):
            return self.restore_text(obj)
        if isinstance(obj, list):
            return [self.restore_in_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.restore_in_obj(v) for k, v in obj.items()}
        return obj


class StreamRestorer:
    """Token-aware tail buffer for streaming text deltas.

    Detects if the trailing edge of the buffer is *inside* a partially-streamed
    token (``cat-`` + 0..5 hex, or ``cat-`` + exactly 6 hex with no boundary
    char yet) and holds from the token's start. Anything before that point is
    safe to regex-restore and emit immediately. No length-based heuristic — we
    only hold what could still grow.
    """

    # Matches a trailing prefix of any *possible* token at end-of-buffer:
    #   - any non-empty prefix of a category word (e.g. "n", "na", "nam"), so
    #     we hold long enough to recognise an incoming "name-..."
    #   - a full category word optionally followed by "-" and 0..6 hex
    #   - a full category-dash-6hex token (boundary char hasn't arrived yet)
    _PARTIAL_AT_END = re.compile(
        r"\b(?:"
        r"n|na|nam|name(?:-[0-9a-f]{0,6})?"
        r"|e|em|ema|emai|email(?:-[0-9a-f]{0,6})?"
        r"|p|ph|pho|phon|phone(?:-[0-9a-f]{0,6})?"
        r"|a|ad|add|addr|addre|addres|address(?:-[0-9a-f]{0,6})?"
        r"|u|ur|url(?:-[0-9a-f]{0,6})?"
        r"|d|da|dat|date(?:-[0-9a-f]{0,6})?"
        r"|ac|acc|acco|accou|accoun|account(?:-[0-9a-f]{0,6})?"
        r"|s|se|sec|secr|secre|secret(?:-[0-9a-f]{0,6})?"
        r")\Z"
    )

    def __init__(self, redactor: "Redactor") -> None:
        self.redactor = redactor
        self._buf = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        m = self._PARTIAL_AT_END.search(self._buf)
        safe_end = m.start() if m else len(self._buf)
        if safe_end == 0:
            return ""
        emit = self._buf[:safe_end]
        self._buf = self._buf[safe_end:]
        return self.redactor.restore_text(emit)

    def flush(self) -> str:
        out = self.redactor.restore_text(self._buf)
        self._buf = ""
        return out
