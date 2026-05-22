from __future__ import annotations

import time
from typing import Any, Dict, List

from farasa.segmenter import FarasaSegmenter
from camel_tools.tokenizers.word import simple_word_tokenize

from app.tools.base_tool import BaseTool
from app.utils.logger import logger, log_time


farasa_segmenter = None


def load_farasa() -> None:
    global farasa_segmenter
    try:
        farasa_segmenter = FarasaSegmenter(interactive=False)
        logger.info("✅ Farasa loaded")
    except Exception as e:
        logger.error(f"❌ Farasa failed: {e}")
        farasa_segmenter = None


def farasa_analyze(text: str) -> Dict[str, Any]:
    global farasa_segmenter
    if not farasa_segmenter:
        load_farasa()

    if not farasa_segmenter:
        return {"tool": "farasa", "status": "failed", "error": "Farasa not loaded", "tokens": []}

    t0 = time.time()
    try:
        segmented = farasa_segmenter.segment(text)
        raw_tokens = simple_word_tokenize(text)
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
    except Exception as e:
        logger.error(f"[FARASA] error: {e}")
        return {"tool": "farasa", "status": "error", "error": str(e), "tokens": []}


class FarasaTool(BaseTool):
    tool_name = "farasa"

    def is_loaded(self) -> bool:
        return farasa_segmenter is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return farasa_analyze(text)

