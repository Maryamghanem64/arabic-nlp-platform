from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from app.utils.helpers import strip_diacritics

logger = logging.getLogger(__name__)

_ARABERT_MODEL_ID = os.environ.get("ARABERT_MODEL_ID", "aubmindlab/bert-base-arabertv2")

_arabert_pipeline = None
_arabert_tokenizer = None
_arabert_model = None
_arabert_loaded = False
_arabert_loading = False
_arabert_last_error: Optional[str] = None
_arabert_lock = threading.Lock()


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


def _build_token(surface: str, score: float = 0.0) -> Dict[str, Any]:
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


def _mask_score(masked_text: str, target_token: str) -> float:
    if _arabert_pipeline is None:
        return 0.0

    try:
        result = _arabert_pipeline(masked_text, top_k=5)
    except Exception as exc:
        logger.debug("[AraBERT] fill-mask scoring failed: %s", exc)
        return 0.0

    if not result:
        return 0.0

    if not isinstance(result, list):
        result = [result]

    target = strip_diacritics(str(target_token or "")).strip()
    for candidate in result:
        token_text = candidate.get("token_str") or candidate.get("sequence") or ""
        score = candidate.get("score")
        if score is None:
            continue
        if strip_diacritics(str(token_text)).strip() == target:
            return float(score)

    first = result[0]
    return float(first.get("score") or 0.0)


def load_arabert() -> bool:
    global _arabert_pipeline, _arabert_tokenizer, _arabert_model, _arabert_loaded, _arabert_loading, _arabert_last_error

    with _arabert_lock:
        if _arabert_loaded and _arabert_pipeline is not None:
            return True
        if _arabert_loading:
            return False

        _arabert_loading = True

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(_ARABERT_MODEL_ID)
        model = AutoModelForMaskedLM.from_pretrained(_ARABERT_MODEL_ID)

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
        }

    if _arabert_loading:
        return {
            "status": "loading",
            "reason": f"AraBERT is loading {_ARABERT_MODEL_ID} on demand.",
            "loaded": False,
            "model_id": _ARABERT_MODEL_ID,
        }

    if _arabert_last_error:
        return {
            "status": "unavailable",
            "reason": f"AraBERT failed to load {_ARABERT_MODEL_ID}: {_arabert_last_error}",
            "loaded": False,
            "model_id": _ARABERT_MODEL_ID,
        }

    return {
        "status": "missing_model",
        "reason": f"AraBERT model {_ARABERT_MODEL_ID} is not loaded yet. The first request will attempt to download it if network access and cache allow.",
        "loaded": False,
        "model_id": _ARABERT_MODEL_ID,
    }


def get_arabert_status() -> str:
    return get_arabert_status_detail()["status"]


def arabert_analyze(text: str) -> Dict[str, Any]:
    try:
        if _arabert_pipeline is None and not load_arabert():
            detail = get_arabert_status_detail()
            return {
                "tool": "arabert",
                "status": detail["status"],
                "reason": detail["reason"],
                "input": text,
                "word_count": 0,
                "tokens": [],
            }

        tokens_text = _simple_tokenize(text)
        tokens: List[Dict[str, Any]] = []

        with _arabert_lock:
            for word in tokens_text:
                masked = text.replace(word, "[MASK]", 1)
                score = _mask_score(masked, word)
                tokens.append(_build_token(word, score))

        return {
            "tool": "arabert",
            "status": "ok",
            "reason": "",
            "model": _ARABERT_MODEL_ID,
            "approach": "contextual fill-mask (BERT)",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
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
