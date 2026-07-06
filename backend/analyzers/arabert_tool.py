from __future__ import annotations

import logging
import os
import ctypes
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from app.utils.helpers import strip_diacritics

logger = logging.getLogger(__name__)

_ARABERT_MODEL_ID = os.environ.get("ARABERT_MODEL_ID", "aubmindlab/bert-base-arabertv2")
_ARABERT_LOCAL_ONLY = os.environ.get("ARABERT_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
_ARABERT_LOCAL_PATH = os.environ.get("ARABERT_MODEL_PATH")
if not _ARABERT_LOCAL_PATH:
    _cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--aubmindlab--bert-base-arabertv2"
    _ref_main = _cache_root / "refs" / "main"
    if _ref_main.exists():
        try:
            _snapshot = _ref_main.read_text(encoding="utf-8").strip()
            _candidate = _cache_root / "snapshots" / _snapshot
            if (_candidate / "config.json").exists():
                _ARABERT_LOCAL_PATH = str(_candidate)
        except Exception:
            _ARABERT_LOCAL_PATH = None

_arabert_pipeline = None
_arabert_tokenizer = None
_arabert_model = None
_arabert_loaded = False
_arabert_loading = False
_arabert_last_error: Optional[str] = None
_arabert_lock = threading.Lock()


def _available_physical_memory_mb() -> Optional[float]:
    try:
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return float(status.ullAvailPhys) / (1024.0 * 1024.0)
    except Exception:
        return None


def _should_defer_cold_model_load() -> bool:
    if _arabert_loaded or _arabert_pipeline is not None:
        return False
    if os.environ.get("ARABERT_FORCE_FULL_LOAD", "").lower() in {"1", "true", "yes"}:
        return False
    mode = (os.environ.get("ARABIC_NLP_MODE") or os.environ.get("ARABIC_NLP_RUN_MODE") or "demo").strip().lower()
    if mode == "demo":
        return True

    available_mb = _available_physical_memory_mb()
    if available_mb is None:
        return False
    threshold_mb = float(os.environ.get("ARABERT_DEMO_MIN_COLD_LOAD_MB", "1024"))
    return available_mb < threshold_mb


def _deferred_model_result(text: str, reason: str) -> Dict[str, Any]:
    tokens_text = _simple_tokenize(text)
    return {
        "tool": "arabert",
        "status": "ok",
        "reason": reason,
        "model": _ARABERT_MODEL_ID,
        "approach": "demo fallback tokenization; contextual model load deferred",
        "input": text,
        "word_count": len(tokens_text),
        "tokens": [_build_token(token, 0.0, []) for token in tokens_text],
        "meta": {
            "model": _ARABERT_MODEL_ID,
            "method": "resource-present-demo-fallback",
            "role": "contextual support / disambiguation",
            "local_files_only": _ARABERT_LOCAL_ONLY,
            "model_path": _ARABERT_LOCAL_PATH,
            "available_memory_mb": _available_physical_memory_mb(),
        },
    }


def _confidence_level(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _simple_tokenize(text: str) -> List[str]:
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize

        tokens = [tok for tok in simple_word_tokenize(text or "") if tok and tok.strip()]
        return tokens
    except Exception:
        return [part for part in str(text or "").split() if part.strip()]


def _build_token(
    surface: str,
    score: float = 0.0,
    candidates: Optional[List[Dict[str, Any]]] = None,
    *,
    surface_token_score: Optional[float] = None,
    top_candidate_score: Optional[float] = None,
) -> Dict[str, Any]:
    unsupported_note = (
        "AraBERT base model does not provide lemma/root/POS without a fine-tuned head."
    )

    return {
        "surface": surface,
        "lemma": None,
        "root": None,
        "pos": None,
        "gloss": None,
        "features": {
            "gender": None,
            "number": None,
            "tense": None,
            "person": None,
            "case": None,
            "definite": None,
            "voice": None,
        },
        "segmentation": [surface],
        "dependency": {"head": None, "head_text": None, "deprel": None},
        "confidence": {"score": round(float(score), 4), "level": _confidence_level(float(score))},
        "meta": {
            "model": _ARABERT_MODEL_ID,
            "method": "fill-mask",
            "role": "contextual support / disambiguation",
            "morphology_supported": False,
            "supported_features": ["contextual"],
            "unsupported_features": [
                "lemma",
                "root",
                "pos",
                "segmentation",
                "dependency",
            ],
            "display_note": unsupported_note,
            "candidates": candidates or [],
            "surface_token_score": None
            if surface_token_score is None
            else round(float(surface_token_score), 6),
            "top_candidate_score": None
            if top_candidate_score is None
            else round(float(top_candidate_score), 6),
        },
        "capabilities": {
            "contextual": True,
            "lemma": False,
            "root": False,
            "pos": False,
            "segmentation": False,
            "dependency": False,
        },
        "analyses": [
            {
                "lemma": None,
                "root": None,
                "pos": None,
                "gender": None,
                "number": None,
                "tense": None,
                "gloss": None,
            }
        ],
    }



def _normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "token": candidate.get("token_str") or candidate.get("sequence") or "",
        "score": round(float(candidate.get("score") or 0.0), 6),
        "token_id": candidate.get("token_id"),
        "sequence": candidate.get("sequence"),
    }


def _predict_mask(masked_text: str, target_token: str) -> Tuple[float, List[Dict[str, Any]]]:
    if _arabert_pipeline is None:
        return 0.0, []

    try:
        result = _arabert_pipeline(masked_text, top_k=5)
    except Exception as exc:
        logger.debug("[AraBERT] fill-mask scoring failed: %s", exc)
        return 0.0, []

    if not result:
        return 0.0, []

    if not isinstance(result, list):
        result = [result]

    candidates = [_normalize_candidate(candidate) for candidate in result]
    target = strip_diacritics(str(target_token or "")).strip()
    for candidate in candidates:
        if strip_diacritics(str(candidate["token"])).strip() == target:
            return float(candidate["score"]), candidates

    return float(candidates[0]["score"] or 0.0), candidates


def _actual_surface_token_score(masked_text: str, target_token: str) -> Optional[float]:
    if _arabert_model is None or _arabert_tokenizer is None:
        return None

    try:
        encoded = _arabert_tokenizer(masked_text, return_tensors="pt")
        mask_token_id = getattr(_arabert_tokenizer, "mask_token_id", None)
        input_ids = encoded.get("input_ids")
        if input_ids is None or mask_token_id is None:
            return None

        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=False).flatten().tolist()
        if not mask_positions:
            return None

        target_ids = _arabert_tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_ids) != 1:
            return None

        with torch.no_grad():
            outputs = _arabert_model(**encoded)
            logits = outputs.logits[0, mask_positions[0]]
            probabilities = torch.softmax(logits, dim=-1)
            return float(probabilities[target_ids[0]].item())
    except Exception as exc:
        logger.debug("[AraBERT] surface token probability failed: %s", exc)
        return None


def load_arabert() -> bool:
    global _arabert_pipeline, _arabert_tokenizer, _arabert_model, _arabert_loaded, _arabert_loading, _arabert_last_error

    with _arabert_lock:
        if _arabert_loaded and _arabert_pipeline is not None:
            return True
        if _arabert_loading:
            return False
        _arabert_loading = True

    try:
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, pipeline

        def _load_model_from_local_path(source: str):
            try:
                return AutoModelForMaskedLM.from_pretrained(source, local_files_only=_ARABERT_LOCAL_ONLY)
            except Exception:
                if not _ARABERT_LOCAL_PATH:
                    raise

                from safetensors.torch import load_file

                config = AutoConfig.from_pretrained(source, local_files_only=True)
                model = AutoModelForMaskedLM.from_config(config)
                state_dict = load_file(str(Path(_ARABERT_LOCAL_PATH) / "model.safetensors"))
                model.load_state_dict(state_dict, strict=False)
                model.tie_weights()
                return model

        source = _ARABERT_LOCAL_PATH or _ARABERT_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=_ARABERT_LOCAL_ONLY)
        model = _load_model_from_local_path(source)

        _arabert_tokenizer = tokenizer
        _arabert_model = model
        _arabert_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1)
        _arabert_loaded = True
        _arabert_last_error = None
        logger.info("AraBERT loaded and cached: %s", _ARABERT_MODEL_ID)
        return True
    except Exception as exc:
        _arabert_pipeline = None
        _arabert_tokenizer = None
        _arabert_model = None
        _arabert_loaded = False
        _arabert_last_error = str(exc)
        logger.warning("AraBERT failed to load: %s", exc)
        return False
    finally:
        with _arabert_lock:
            _arabert_loading = False


def get_arabert_status_detail() -> Dict[str, Any]:
    try:
        import importlib.util

        has_transformers = importlib.util.find_spec("transformers") is not None
        has_torch = importlib.util.find_spec("torch") is not None
    except Exception:
        has_transformers = False
        has_torch = False

    if _arabert_loaded and _arabert_pipeline is not None:
        return {
            "status": "ok",
            "reason": f"AraBERT loaded and cached from {_ARABERT_MODEL_ID}.",
            "loaded": True,
            "model_id": _ARABERT_MODEL_ID,
            "model_path": _ARABERT_LOCAL_PATH,
            "local_files_only": _ARABERT_LOCAL_ONLY,
            "has_transformers": has_transformers,
            "has_torch": has_torch,
        }

    if _arabert_loading:
        return {
            "status": "loading",
            "reason": f"AraBERT is loading {_ARABERT_MODEL_ID} on demand.",
            "loaded": False,
            "model_id": _ARABERT_MODEL_ID,
            "model_path": _ARABERT_LOCAL_PATH,
            "local_files_only": _ARABERT_LOCAL_ONLY,
            "has_transformers": has_transformers,
            "has_torch": has_torch,
        }

    if _arabert_last_error:
        return {
            "status": "unavailable",
            "reason": f"AraBERT failed to load {_ARABERT_MODEL_ID}: {_arabert_last_error}",
            "loaded": False,
            "model_id": _ARABERT_MODEL_ID,
            "model_path": _ARABERT_LOCAL_PATH,
            "local_files_only": _ARABERT_LOCAL_ONLY,
            "has_transformers": has_transformers,
            "has_torch": has_torch,
        }

    local_model_ready = bool(_ARABERT_LOCAL_PATH and (Path(_ARABERT_LOCAL_PATH) / "config.json").exists())
    if has_transformers and has_torch and (local_model_ready or not _ARABERT_LOCAL_ONLY):
        return {
            "status": "ok",
            "reason": (
                f"AraBERT model {_ARABERT_MODEL_ID} is available"
                + (f" in the local cache at {_ARABERT_LOCAL_PATH}." if local_model_ready else " for on-demand loading.")
            ),
            "loaded": False,
            "model_id": _ARABERT_MODEL_ID,
            "model_path": _ARABERT_LOCAL_PATH,
            "local_files_only": _ARABERT_LOCAL_ONLY,
            "has_transformers": has_transformers,
            "has_torch": has_torch,
        }

    return {
        "status": "missing_model",
        "reason": f"AraBERT model {_ARABERT_MODEL_ID} is not loaded yet. The first request will use the local cache only.",
        "loaded": False,
        "model_id": _ARABERT_MODEL_ID,
        "model_path": _ARABERT_LOCAL_PATH,
        "local_files_only": _ARABERT_LOCAL_ONLY,
        "has_transformers": has_transformers,
        "has_torch": has_torch,
    }


def get_arabert_status() -> str:
    return get_arabert_status_detail()["status"]


def arabert_analyze(text: str) -> Dict[str, Any]:
    try:
        if _should_defer_cold_model_load():
            return _deferred_model_result(
                text,
                "AraBERT local model found; cold model load deferred because available RAM is below the demo budget.",
            )

        if _arabert_pipeline is None and not load_arabert():
            detail = get_arabert_status_detail()
            return {
                "tool": "arabert",
                "status": detail["status"],
                "reason": detail["reason"],
                "input": text,
                "word_count": 0,
                "tokens": [],
                "meta": detail,
            }

        tokens_text = _simple_tokenize(text)
        tokens: List[Dict[str, Any]] = []

        with _arabert_lock:
            for word in tokens_text:
                masked = text.replace(word, "[MASK]", 1)
                score, candidates = _predict_mask(masked, word)
                actual_score = _actual_surface_token_score(masked, word)
                tokens.append(
                    _build_token(
                        word,
                        score,
                        candidates,
                        surface_token_score=actual_score,
                        top_candidate_score=score,
                    )
                )

        return {
            "tool": "arabert",
            "status": "ok",
            "reason": "",
            "model": _ARABERT_MODEL_ID,
            "approach": "contextual fill-mask (BERT)",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
            "meta": {
                "model": _ARABERT_MODEL_ID,
                "method": "fill-mask",
                "role": "contextual support / disambiguation",
                "local_files_only": _ARABERT_LOCAL_ONLY,
            },
        }
    except Exception as exc:
        logger.exception("[AraBERT] error")
        return {
            "tool": "arabert",
            "status": "error",
            "reason": str(exc),
            "input": text,
            "word_count": 0,
            "tokens": [],
        }


__all__ = [
    "arabert_analyze",
    "get_arabert_status",
    "get_arabert_status_detail",
    "load_arabert",
]
