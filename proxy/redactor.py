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

# Tags that Claude Code injects whose content changes between turns
# (timestamps, command output, deferred-tool lists, etc.). Hashing the text
# *with* these tags causes a cache miss every turn for the same logical
# message; stripping them for the cache key gives us reliable hits across
# turns. Tag content still gets PII-detected on the full text.
_VOLATILE_TAGS_RE = re.compile(
    r"<(?:system-reminder|local-command-caveat|local-command-stdout|"
    r"available-deferred-tools)>.*?</(?:system-reminder|local-command-caveat|"
    r"local-command-stdout|available-deferred-tools)>",
    re.DOTALL,
)


def _split_by_volatile_tags(text: str) -> tuple[list[tuple[str, bool, int]], str]:
    """Return (chunks, stable_concat) where chunks is a list of
    ``(chunk_text, is_volatile, offset_in_original)`` and stable_concat is the
    concatenation of stable (non-volatile) chunks — a stable cache key under
    Claude Code's tag-injection behaviour.
    """
    chunks: list[tuple[str, bool, int]] = []
    cursor = 0
    stable_parts: list[str] = []
    for m in _VOLATILE_TAGS_RE.finditer(text):
        if m.start() > cursor:
            chunk = text[cursor : m.start()]
            chunks.append((chunk, False, cursor))
            stable_parts.append(chunk)
        chunks.append((text[m.start() : m.end()], True, m.start()))
        cursor = m.end()
    if cursor < len(text):
        chunk = text[cursor:]
        chunks.append((chunk, False, cursor))
        stable_parts.append(chunk)
    return chunks, "".join(stable_parts)

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

        # (2) Split each target into stable + volatile chunks. Stable chunks
        #     cache by content hash and survive Claude Code's tag-injection;
        #     volatile chunks (system-reminder, command-stdout, etc.) get
        #     detected fresh each call but are typically tiny.
        per_target_chunks: list[list[tuple[str, bool, int]]] = []
        for text, _ in targets:
            chunks, _ = _split_by_volatile_tags(text)
            per_target_chunks.append(chunks)

        chunk_spans: dict[tuple[int, int], list[dict]] = {}
        miss_jobs: list[tuple[int, int, str, bool]] = []  # (target_idx, chunk_idx, text, is_volatile)

        for ti, chunks in enumerate(per_target_chunks):
            for ci, (ctext, is_volatile, _off) in enumerate(chunks):
                if not ctext or len(ctext) < 3 or not any(c.isalpha() for c in ctext):
                    chunk_spans[(ti, ci)] = []
                    continue
                if is_volatile:
                    miss_jobs.append((ti, ci, ctext, True))
                    continue
                h = hashlib.sha1(ctext.encode("utf-8", errors="replace")).hexdigest()
                cached = self._span_cache.get(h)
                if cached is not None:
                    chunk_spans[(ti, ci)] = cached
                else:
                    miss_jobs.append((ti, ci, ctext, False))

        if miss_jobs:
            miss_texts = [j[2] for j in miss_jobs]
            try:
                batch_spans = self._engine.detect_batch(miss_texts)
            except Exception as e:
                log.warning(f"[DETECT] batch failed: {e}; falling back to per-text")
                batch_spans = [self._engine.detect(t) for t in miss_texts]
            # Trim cache before inserting
            if len(self._span_cache) >= self._cache_size:
                for k in list(self._span_cache.keys())[: self._cache_size // 4]:
                    self._span_cache.pop(k, None)
            for (ti, ci, ctext, is_volatile), spans in zip(miss_jobs, batch_spans):
                chunk_spans[(ti, ci)] = spans
                if not is_volatile:
                    h = hashlib.sha1(ctext.encode("utf-8", errors="replace")).hexdigest()
                    self._span_cache[h] = spans

        # (3) Combine chunk spans into target-relative spans, then apply.
        for ti, (text, setter) in enumerate(targets):
            all_spans: list[dict] = []
            for ci, (_ctext, _is_vol, offset) in enumerate(per_target_chunks[ti]):
                for sp in chunk_spans.get((ti, ci), ()):
                    shifted = dict(sp)
                    shifted["start"] += offset
                    shifted["end"] += offset
                    all_spans.append(shifted)
            if not all_spans:
                continue
            redacted = self._apply_spans(text, all_spans)
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
