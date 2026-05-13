from __future__ import annotations

import time
from typing import Any, Dict, List

from farasa.segmenter import FarasaSegmenter

from backend.analyzers.base import Analyzer
from backend.utils.text_norm import normalize_whitespace


class FarasaTool(Analyzer):
    tool_name = "farasa"

    def __init__(self):
        self._segmenter = FarasaSegmenter(interactive=False)

    def analyze(self, text: str) -> Dict[str, Any]:
        t0 = time.time()
        segmented = self._segmenter.segment(normalize_whitespace(text))

        raw_tokens = []
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize

            raw_tokens = simple_word_tokenize(text)
        except Exception:
            raw_tokens = text.split()

        raw_segs = segmented.split()
        token_outputs: List[Dict[str, Any]] = []

        for i, token in enumerate(raw_tokens):
            seg = raw_segs[i] if i < len(raw_segs) else token
            parts = [p for p in seg.split("+") if p]
            token_outputs.append({"surface": token, "segmentation": parts, "analyses": []})

        return {
            "tool": "farasa",
            "status": "ok",
            "input": text,
            "word_count": len(token_outputs),
            "tokens": token_outputs,
            "segmented_text": segmented,
            "elapsed": time.time() - t0,
        }

