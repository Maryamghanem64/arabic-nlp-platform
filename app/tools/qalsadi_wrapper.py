from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional

from app.utils.logger import logger


_thread_local = threading.local()


def _normalize_arabic(t: str) -> str:
    t = re.sub(r"[أإآ]", "ا", str(t or ""))
    t = re.sub(r"ـ", "", t)
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)
    return t.strip()


def _fallback_tokenize(text: str) -> List[str]:
    return [part for part in str(text or "").split() if part]


def _get_simple_tokenizer():
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize

        return simple_word_tokenize
    except Exception:
        return None


def _load_lemmas_for_thread():
    """Create a fresh qalsadi Lemmatizer instance for the current thread.

    This avoids sharing SQLite-backed objects across FastAPI threads.
    """
    try:
        import qalsadi.lemmatizer as qalsadi_lem

        return qalsadi_lem.Lemmatizer()
    except Exception as exc:
        logger.warning("Qalsadi Lemmatizer load failed: %s", exc)
        return None


def _get_analyzer() -> Optional[Any]:
    # Recreate analyzer per thread to avoid sharing SQLite-backed state.
    if getattr(_thread_local, "analyzer", None) is None:
        _thread_local.analyzer = _load_lemmas_for_thread()
    return _thread_local.analyzer



def qalsadi_analyze(text: str) -> Dict[str, Any]:
    tool = "qalsadi"
    t0 = time.time()
    try:
        normalized = _normalize_arabic(text)
        if not normalized:
            return {"tool": tool, "status": "unavailable", "tokens": [], "lemmas": [], "reason": "Empty text after normalization."}

        # Hard isolation: always create a fresh analyzer per request.
        # This fully avoids any possibility of SQLite objects being reused across threads.
        analyzer = _load_lemmas_for_thread()
        if analyzer is None:
            return {"tool": tool, "status": "unavailable", "tokens": [], "lemmas": [], "reason": "Qalsadi is unavailable (load failed)."}

        tokenizer = getattr(_thread_local, "tokenizer", None)
        if tokenizer is None:
            tokenizer = _get_simple_tokenizer() or _fallback_tokenize
            _thread_local.tokenizer = tokenizer

        tokens_text = tokenizer(normalized) if callable(tokenizer) else _fallback_tokenize(normalized)
        lemmas = analyzer.lemmatize_text(normalized)


        tokens: List[Dict[str, Any]] = []
        lemmas_out: List[str] = []
        for i, word in enumerate(tokens_text):
            lemma = lemmas[i] if i < len(lemmas) else word
            tokens.append({"surface": word, "lemma": lemma, "pos": None, "stem": None})
            lemmas_out.append(lemma)

        # Unified schema: keep tokens list and also supply `lemmas`.
        return {
            "tool": tool,
            "status": "ok",
            "tokens": tokens,
            "lemmas": lemmas_out,
            "reason": "",
        }

    except Exception as exc:
        logger.warning("[QALSADI] error: %s", exc)
        return {"tool": tool, "status": "error", "tokens": [], "lemmas": [], "reason": str(exc)}
    finally:
        _ = time.time() - t0

