from __future__ import annotations

import time
import threading
from typing import Any, Dict, List

from camel_tools.tokenizers.word import simple_word_tokenize

from app.tools.base_tool import BaseTool
from app.utils.logger import logger, log_time


qalsadi_analyzer = None
qalsadi_thread_local = threading.local()



def load_qalsadi() -> None:
    global qalsadi_analyzer
    try:
        import qalsadi.lemmatizer as qalsadi_lem

        # INSTANCE not module
        qalsadi_analyzer = qalsadi_lem.Lemmatizer()
        logger.info("✅ Qalsadi loaded")
    except Exception as e:
        logger.error(f"❌ Qalsadi failed: {e}")
        qalsadi_analyzer = None


def _normalize_arabic(t: str) -> str:
    import re

    t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"ـ", "", t)
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)
    return t.strip()


def qalsadi_analyze(text: str) -> Dict[str, Any]:
    global qalsadi_analyzer

    # Lazy-load so the analyzer instance is guaranteed to exist in this module.
    if qalsadi_analyzer is None:
        load_qalsadi()

    if qalsadi_analyzer is None:
        return {"tool": "qalsadi", "status": "failed", "tokens": []}

    t0 = time.time()
    try:
        normalized = _normalize_arabic(text or "")
        if not normalized:
            return {
                "tool": "qalsadi",
                "status": "error",
                "error": "Empty text after normalization",
                "tokens": [],
            }

        tokens_text = simple_word_tokenize(normalized)

        # IMPORTANT VERIFIED: lemmatize_text returns flat list[str]
        lemmas = qalsadi_analyzer.lemmatize_text(normalized)

        tokens: List[Dict[str, Any]] = []
        for i, word in enumerate(tokens_text):
            lemma = lemmas[i] if i < len(lemmas) else word
            tokens.append({
                "surface": word,
                "lemma": lemma,
                "pos": None,
                "stem": None,
            })

        log_time("qalsadi", text, time.time() - t0)
        return {
            "tool": "qalsadi",
            "status": "ok",
            "approach": "rule-based lemmatization",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
        }
    except Exception as e:
        logger.error(f"[QALSADI] error: {e}")
        return {"tool": "qalsadi", "status": "error", "error": str(e), "tokens": []}


class QalsadiTool(BaseTool):
    tool_name = "qalsadi"

    def is_loaded(self) -> bool:
        return qalsadi_analyzer is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return qalsadi_analyze(text)

