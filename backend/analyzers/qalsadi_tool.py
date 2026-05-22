from __future__ import annotations

import time
from typing import Any, Dict, List

from camel_tools.tokenizers.word import simple_word_tokenize

from backend.analyzers.base import Analyzer
from backend.utils.text_norm import normalize_whitespace


QALSADI_POS_MAP = {
    "فعل": "VERB",
    "اسم": "NOUN",
    "صفة": "ADJ",
    "حرف": "PART",
    "ضمير": "PRON",
    "ظرف": "ADV",
    "STOPWORD": "STOP",
    # Some qalsadi builds might emit these generic types
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ": "ADJ",
}



class QalsadiTool(Analyzer):
    tool_name = "qalsadi"

    def __init__(self):
        import qalsadi.lemmatizer as qalsadi_lem

        # Qalsadi lemmatizer instance
        self._analyzer = qalsadi_lem.Lemmatizer()

    def analyze(self, text: str) -> Dict[str, Any]:
        t0 = time.time()
        tokens_text = simple_word_tokenize(normalize_whitespace(text))
        tokens: List[Dict[str, Any]] = []

        for word in tokens_text:
            result = self._analyzer.lemmatize_text(word)

            pos_ar: str = ""
            lemma: str = word
            root: str | None = None
            freq: float | None = None

            if result:
                lemma_obj = result[0]
                if isinstance(lemma_obj, str):
                    lemma = lemma_obj or word
                else:
                    pos_ar = getattr(lemma_obj, "type", "") or ""
                    lemma = getattr(lemma_obj, "lemma", word) or word
                    root = getattr(lemma_obj, "root", None)
                    freq_val = getattr(lemma_obj, "freq", None)
                    if freq_val is not None:
                        try:
                            freq = float(freq_val)
                        except Exception:
                            freq = None

            pos = QALSADI_POS_MAP.get(pos_ar, pos_ar or "UNKNOWN")

            tokens.append(
                {
                    "surface": word,
                    "lemma": lemma,
                    "root": root,
                    "pos": pos,
                    "freq": freq,
                    # keep extra fields for compatibility
                    "status": "ok",
                }
            )

        return {
            "tool": "qalsadi",
            "status": "ok",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
            "approach": "qalsadi.lemmatizer",
            "elapsed": time.time() - t0,
        }

