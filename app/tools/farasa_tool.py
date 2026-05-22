from __future__ import annotations

import time
from typing import Any, Dict, List

from app.core.tool_registry import unavailable_result
from app.tools.base_tool import BaseTool
from app.utils.logger import logger, log_time


FarasaSegmenter = None
simple_word_tokenize = None
farasa_segmenter = None
farasa_import_error = None


def _fallback_tokenize(text: str) -> List[str]:
    return [part for part in str(text or "").split() if part]


def _ensure_imports() -> bool:
    global FarasaSegmenter, simple_word_tokenize, farasa_import_error
    if FarasaSegmenter and simple_word_tokenize:
        return True
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize as _simple_word_tokenize
        from farasa.segmenter import FarasaSegmenter as _FarasaSegmenter

        FarasaSegmenter = _FarasaSegmenter
        simple_word_tokenize = _simple_word_tokenize
        farasa_import_error = None
        return True
    except Exception as exc:
        farasa_import_error = str(exc)
        return False


def load_farasa() -> None:
    global farasa_segmenter
    if not _ensure_imports():
        logger.warning("Farasa unavailable: %s", farasa_import_error)
        farasa_segmenter = None
        return
    try:
        farasa_segmenter = FarasaSegmenter(interactive=False)
        logger.info("Farasa loaded")
    except Exception as exc:
        logger.warning("Farasa failed: %s", exc)
        farasa_segmenter = None


def farasa_analyze(text: str) -> Dict[str, Any]:
    global farasa_segmenter
    if not farasa_segmenter:
        load_farasa()

    if not farasa_segmenter:
        return unavailable_result("farasa", farasa_import_error or "Farasa package, Java, or JAR files are not available.", text)

    t0 = time.time()
    try:
        segmented = farasa_segmenter.segment(text)
        raw_tokens = simple_word_tokenize(text) if simple_word_tokenize else _fallback_tokenize(text)
        raw_segs = segmented.split()

        token_outputs: List[Dict[str, Any]] = []
        for i, token in enumerate(raw_tokens):
            seg = raw_segs[i] if i < len(raw_segs) else token
            parts = [p for p in seg.split("+") if p]
            token_outputs.append({"surface": token, "analyses": [], "segmentation": parts})

        log_time("farasa", text, time.time() - t0)
        return {
            "tool": "farasa",
            "status": "ok",
            "input": text,
            "word_count": len(token_outputs),
            "segmented_text": segmented,
            "tokens": token_outputs,
        }
    except Exception as exc:
        logger.warning("[FARASA] error: %s", exc)
        return {"tool": "farasa", "status": "error", "reason": str(exc), "input": text, "word_count": 0, "tokens": []}


class FarasaTool(BaseTool):
    tool_name = "farasa"

    def is_loaded(self) -> bool:
        return farasa_segmenter is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return farasa_analyze(text)
