from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List

from app.core.tool_registry import unavailable_result
from app.tools.base_tool import BaseTool
from app.utils.logger import logger, log_time


qalsadi_analyzer = None
qalsadi_thread_local = threading.local()
simple_word_tokenize = None
qalsadi_import_error = None


def _fallback_tokenize(text: str) -> List[str]:
    return [part for part in str(text or "").split() if part]


def _ensure_imports() -> bool:
    global simple_word_tokenize, qalsadi_import_error
    try:
        if simple_word_tokenize is None:
            from camel_tools.tokenizers.word import simple_word_tokenize as _simple_word_tokenize

            simple_word_tokenize = _simple_word_tokenize
        import qalsadi.lemmatizer  # noqa: F401

        qalsadi_import_error = None
        return True
    except Exception as exc:
        qalsadi_import_error = str(exc)
        return False


def load_qalsadi() -> None:
    global qalsadi_analyzer
    if not _ensure_imports():
        logger.warning("Qalsadi unavailable: %s", qalsadi_import_error)
        qalsadi_analyzer = None
        return
    try:
        import qalsadi.lemmatizer as qalsadi_lem

        qalsadi_analyzer = qalsadi_lem.Lemmatizer()
        logger.info("Qalsadi loaded")
    except Exception as exc:
        logger.warning("Qalsadi failed: %s", exc)
        qalsadi_analyzer = None


def _normalize_arabic(t: str) -> str:
    t = re.sub(r"[أإآ]", "ا", str(t or ""))
    t = re.sub(r"ـ", "", t)
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)
    return t.strip()


def qalsadi_analyze(text: str) -> Dict[str, Any]:
    global qalsadi_analyzer
    if qalsadi_analyzer is None:
        load_qalsadi()

    if qalsadi_analyzer is None:
        return unavailable_result("qalsadi", qalsadi_import_error or "Qalsadi package is not available.", text)

    t0 = time.time()
    try:
        normalized = _normalize_arabic(text)
        if not normalized:
            return {"tool": "qalsadi", "status": "error", "reason": "Empty text after normalization.", "tokens": []}

        tokens_text = simple_word_tokenize(normalized) if simple_word_tokenize else _fallback_tokenize(normalized)
        lemmas = qalsadi_analyzer.lemmatize_text(normalized)
        tokens: List[Dict[str, Any]] = []
        for i, word in enumerate(tokens_text):
            tokens.append({"surface": word, "lemma": lemmas[i] if i < len(lemmas) else word, "pos": None, "stem": None})

        log_time("qalsadi", text, time.time() - t0)
        return {
            "tool": "qalsadi",
            "status": "ok",
            "approach": "rule-based lemmatization",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
        }
    except Exception as exc:
        logger.warning("[QALSADI] error: %s", exc)
        return {"tool": "qalsadi", "status": "error", "reason": str(exc), "input": text, "word_count": 0, "tokens": []}


class QalsadiTool(BaseTool):
    tool_name = "qalsadi"

    def is_loaded(self) -> bool:
        return qalsadi_analyzer is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return qalsadi_analyze(text)
