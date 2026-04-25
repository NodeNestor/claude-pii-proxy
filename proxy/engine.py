"""
Hyper-lightweight inference engine for openai/privacy-filter.

No PyTorch, no transformers — just onnxruntime + tokenizers + huggingface_hub.

* Picks CUDA / DirectML provider when available, falls back to CPU.
* Defaults to the smallest available quantization (q4 → int8 → fp16 → fp32).
* Total installed footprint is ~50–200MB depending on which onnxruntime
  package the user has (CPU vs GPU).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Iterable

log = logging.getLogger("pii-proxy.engine")

DEFAULT_MODEL_ID = "openai/privacy-filter"

# Default order: best speed/size tradeoff first. For openai/privacy-filter:
#
#   variant               disk     CPU single latency (Win, x86-64)
#   ------                ----     --------
#   model_quantized.onnx  ~1.5 GB  ~70 ms     <- fastest CPU kernels (int8)
#   model_q4.onnx         ~875 MB  ~450 ms
#   model_q4f16.onnx      ~772 MB  ~580 ms    smallest, broken on DirectML
#   model_fp16.onnx       ~2.7 GB
#   model.onnx            ~5.4 GB
#
# Override with PII_PROXY_QUANT to pin a specific file (e.g. "model_q4f16.onnx"
# if you care more about disk than latency).
_QUANT_PRIORITY = (
    "model_quantized.onnx",
    "model_q4.onnx",
    "model_q4f16.onnx",
    "model_int8.onnx",
    "model_fp16.onnx",
    "model.onnx",
)

# Override via env var: pin a specific quantization, e.g. "model_quantized.onnx"
_QUANT_ENV = "PII_PROXY_QUANT"


def _pick_providers() -> list[str]:
    """Return onnxruntime execution providers in preference order.

    For this model the int8 ``model_quantized.onnx`` runs faster on CPU than
    on DirectML (the MoE/4-bit ops fall back to slow paths on DML). CUDA
    and CoreML are still preferred when present. Set ``PII_PROXY_PROVIDERS``
    (comma-separated) to override.
    """
    import onnxruntime as ort

    override = os.environ.get("PII_PROXY_PROVIDERS")
    available = set(ort.get_available_providers())
    if override:
        chosen = [p.strip() for p in override.split(",") if p.strip() in available]
    else:
        preferred = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            # DmlExecutionProvider intentionally last — slow + sometimes
            # produces all-zero predictions for this model's q4f16 graph.
            "CPUExecutionProvider",
            "DmlExecutionProvider",
        ]
        chosen = [p for p in preferred if p in available]
    if not chosen:
        chosen = ["CPUExecutionProvider"]
    log.info(f"[ENGINE] Available ORT providers: {sorted(available)}")
    log.info(f"[ENGINE] Using providers: {chosen}")
    return chosen


def _download_model_assets(model_id: str) -> tuple[str, str, str]:
    """Download tokenizer.json, config.json, and the chosen ONNX weights.

    Returns (tokenizer_path, config_path, onnx_path).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    tokenizer_path = hf_hub_download(model_id, "tokenizer.json")
    config_path = hf_hub_download(model_id, "config.json")

    pinned = os.environ.get(_QUANT_ENV)
    candidates: tuple[str, ...] = (pinned,) if pinned else _QUANT_PRIORITY

    last_err: Exception | None = None
    for fname in candidates:
        if not fname:
            continue
        for sub in (f"onnx/{fname}", fname):
            try:
                path = hf_hub_download(model_id, sub)
                # Some quantized exports have a sibling weights file
                _maybe_pull_external_data(model_id, sub)
                log.info(f"[ENGINE] Using ONNX weights: {sub}")
                return tokenizer_path, config_path, path
            except EntryNotFoundError:
                continue
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(
        f"Could not locate any ONNX weights in {model_id}. Last error: {last_err}"
    )


def _maybe_pull_external_data(model_id: str, onnx_subpath: str) -> None:
    """ONNX models >2GB store weights in sibling ``<name>_data`` /
    ``<name>_data_<n>`` shard files. Discover them by listing the repo and
    download all matching siblings so the model loads cleanly."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.errors import EntryNotFoundError

    candidates: list[str] = []
    try:
        files = list_repo_files(model_id)
        for f in files:
            if f.startswith(onnx_subpath) and f != onnx_subpath:
                candidates.append(f)
    except Exception:
        # Fallback: try common naming patterns
        candidates = [onnx_subpath + sfx for sfx in (".data", "_data", "_data_1", "_data_2", "_data_3")]
    for cand in candidates:
        try:
            hf_hub_download(model_id, cand)
            log.info(f"[ENGINE]   pulled sidecar: {cand}")
        except EntryNotFoundError:
            pass
        except Exception as e:
            log.warning(f"[ENGINE]   sidecar pull failed for {cand}: {e}")


def _load_id2label(config_path: str) -> dict[int, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    raw = cfg.get("id2label") or {}
    return {int(k): str(v) for k, v in raw.items()}


def _softmax(x):
    import numpy as np

    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# Power-of-two-ish bucketing to limit padding waste while keeping the number
# of distinct shapes (and therefore ORT's compiled kernels) bounded.
_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _bucket_size(n: int) -> int:
    for b in _BUCKETS:
        if n <= b:
            return b
    # Round up to the next 1024 for very long inputs
    return ((n + 1023) // 1024) * 1024


# ---------------------------------------------------------------------------
# BIOES decoding
# ---------------------------------------------------------------------------

# Map model categories -> short token category we mint
_CATEGORY_MAP = {
    "private_person": "name",
    "person": "name",
    "private_email": "email",
    "email": "email",
    "private_phone": "phone",
    "phone": "phone",
    "private_address": "address",
    "address": "address",
    "private_url": "url",
    "url": "url",
    "private_date": "date",
    "date": "date",
    "account_number": "account",
    "secret": "secret",
}


def _short_category(label_tail: str) -> str | None:
    return _CATEGORY_MAP.get(label_tail.lower())


def _decode_bioes(
    label_ids,
    scores,
    offsets,
    id2label: dict[int, str],
    text: str,
) -> list[dict]:
    """Convert per-token BIOES predictions into character-level spans."""
    spans: list[dict] = []
    open_span: dict | None = None

    def _close(s):
        if s and s["end"] > s["start"]:
            s["real"] = text[s["start"] : s["end"]]
            spans.append(s)

    for label_id, score, (start, end) in zip(label_ids, scores, offsets):
        # Special tokens have offsets (0,0); skip them.
        if start == 0 and end == 0:
            continue
        label = id2label.get(int(label_id), "O")
        if label == "O":
            _close(open_span)
            open_span = None
            continue
        prefix, _, tail = label.partition("-")
        cat = _short_category(tail) if tail else None
        if not cat:
            _close(open_span)
            open_span = None
            continue
        if prefix == "B":
            _close(open_span)
            open_span = {"start": int(start), "end": int(end), "category": cat, "score": float(score)}
        elif prefix == "I":
            if open_span and open_span["category"] == cat:
                open_span["end"] = int(end)
                open_span["score"] = min(open_span["score"], float(score))
            else:
                _close(open_span)
                open_span = {"start": int(start), "end": int(end), "category": cat, "score": float(score)}
        elif prefix == "E":
            if open_span and open_span["category"] == cat:
                open_span["end"] = int(end)
                open_span["score"] = min(open_span["score"], float(score))
                _close(open_span)
                open_span = None
            else:
                _close(open_span)
                open_span = None
                _close({"start": int(start), "end": int(end), "category": cat, "score": float(score)})
        elif prefix == "S":
            _close(open_span)
            open_span = None
            _close({"start": int(start), "end": int(end), "category": cat, "score": float(score)})
        else:
            # Unknown prefix; treat as O
            _close(open_span)
            open_span = None
    _close(open_span)
    return spans


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PrivacyFilterEngine:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        min_score: float = 0.5,
        max_seq_len: int = 4096,
    ) -> None:
        self.model_id = model_id
        self.min_score = min_score
        self.max_seq_len = max_seq_len
        self._lock = threading.Lock()
        self._tokenizer = None
        self._session = None
        self._id2label: dict[int, str] = {}
        self._input_names: list[str] = []
        self._pad_id: int = 0
        self._pad_token: str = "[PAD]"

    # ---- loading ----

    def load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            log.info(f"[ENGINE] Loading {self.model_id} ...")
            tokenizer_path, config_path, onnx_path = _download_model_assets(self.model_id)
            self._id2label = _load_id2label(config_path)
            log.info(f"[ENGINE] {len(self._id2label)} labels: e.g. {list(self._id2label.values())[:6]}")

            # Pull pad token info from config / tokenizer config
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("pad_token_id") is not None:
                    self._pad_id = int(cfg["pad_token_id"])
                tok_cfg_path = os.path.join(os.path.dirname(tokenizer_path), "tokenizer_config.json")
                if os.path.exists(tok_cfg_path):
                    with open(tok_cfg_path, "r", encoding="utf-8") as f:
                        tcfg = json.load(f)
                    if tcfg.get("pad_token"):
                        self._pad_token = str(tcfg["pad_token"])
                log.info(f"[ENGINE] pad_id={self._pad_id} pad_token={self._pad_token!r}")
            except Exception as e:
                log.warning(f"[ENGINE] Could not load pad token info: {e}")

            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(tokenizer_path)
            tok.enable_truncation(max_length=self.max_seq_len)
            self._tokenizer = tok

            import onnxruntime as ort

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3
            # Tune CPU threading. 0 = let ORT decide (sometimes under-uses cores).
            # Override with PII_PROXY_THREADS=N for explicit control.
            threads_env = os.environ.get("PII_PROXY_THREADS")
            if threads_env:
                try:
                    so.intra_op_num_threads = int(threads_env)
                except ValueError:
                    pass
            else:
                # Default: half of logical cores, capped at 8 — beyond that
                # int8 GEMM kernels usually contend more than they parallelise.
                cpu_count = os.cpu_count() or 4
                so.intra_op_num_threads = min(8, max(2, cpu_count // 2))
            so.inter_op_num_threads = 1
            providers = _pick_providers()
            self._session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
            self._input_names = [i.name for i in self._session.get_inputs()]
            log.info(
                f"[ENGINE] Ready. Inputs: {self._input_names}  "
                f"intra_op_threads={so.intra_op_num_threads}"
            )

    # ---- inference ----

    def _tokenize(self, text: str):
        enc = self._tokenizer.encode(text)
        return enc

    def detect(self, text: str) -> list[dict]:
        """Detect PII spans in a single text. Convenience wrapper around batch."""
        if not text or not text.strip():
            return []
        return self.detect_batch([text])[0]

    def detect_batch(self, texts: list[str]) -> list[list[dict]]:
        """Run forward passes over a batch of texts, length-bucketed.

        Plain padding-to-max wastes work when batch lengths are uneven (a
        single 2k-token system prompt would force ten 30-token messages to
        also be padded to 2k). Instead, we bucket by power-of-two padded
        length and run one forward pass per bucket — short strings stay
        short, long strings don't get split.

        Returns a list of span lists index-aligned with ``texts``.
        """
        import numpy as np

        if not texts:
            return []
        self.load()

        # Tokenise once, no padding (we'll pad per bucket below).
        encodings = [self._tokenizer.encode(t) for t in texts]

        # Group indices by bucket size.
        buckets: dict[int, list[int]] = {}
        empty_indices: list[int] = []
        for i, e in enumerate(encodings):
            n = len(e.ids)
            if n == 0:
                empty_indices.append(i)
                continue
            b = _bucket_size(n)
            buckets.setdefault(b, []).append(i)

        results: list[list[dict] | None] = [None] * len(texts)
        for i in empty_indices:
            results[i] = []

        for bucket_len, idxs in buckets.items():
            bsize = len(idxs)
            ids = np.full((bsize, bucket_len), self._pad_id, dtype=np.int64)
            mask = np.zeros((bsize, bucket_len), dtype=np.int64)
            for row, src in enumerate(idxs):
                e = encodings[src]
                n = len(e.ids)
                ids[row, :n] = e.ids
                mask[row, :n] = e.attention_mask

            feed: dict[str, "np.ndarray"] = {}
            if "input_ids" in self._input_names:
                feed["input_ids"] = ids
            if "attention_mask" in self._input_names:
                feed["attention_mask"] = mask
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)

            outputs = self._session.run(None, feed)
            logits = outputs[0]
            probs = _softmax(logits)
            label_ids = probs.argmax(axis=-1)
            scores = probs.max(axis=-1)

            for row, src in enumerate(idxs):
                e = encodings[src]
                real_len = len(e.ids)
                spans = _decode_bioes(
                    label_ids=label_ids[row, :real_len],
                    scores=scores[row, :real_len],
                    offsets=e.offsets[:real_len],
                    id2label=self._id2label,
                    text=texts[src],
                )
                cleaned: list[dict] = []
                for s in spans:
                    if s["score"] < self.min_score:
                        continue
                    real = s["real"]
                    stripped = real.strip()
                    if not stripped:
                        continue
                    lead = len(real) - len(real.lstrip())
                    trail = len(real) - len(real.rstrip())
                    cleaned.append(
                        {
                            "start": s["start"] + lead,
                            "end": s["end"] - trail,
                            "category": s["category"],
                            "real": stripped,
                            "score": s["score"],
                        }
                    )
                results[src] = cleaned

        # All indices should be filled by now
        return [r if r is not None else [] for r in results]
