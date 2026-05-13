from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from camel_tools.tokenizers.word import simple_word_tokenize

from backend.analyzers.base import Analyzer
from backend.utils.text_norm import normalize_whitespace


QALSADI_POS_MAP = {
    "اسم": "NOUN",
    "فعل": "VERB",
    "صفة": "ADJECTIVE",
    "ظرف": "ADVERB",
    "حرف": "PARTICLE",
    "ضمير": "PRONOUN",
    "اسم علم": "NOUN",
    "مصدر": "NOUN",
    "اسم فاعل": "ADJECTIVE",
    "اسم مفعول": "ADJECTIVE",
    "صفة مشبهة": "ADJECTIVE",
}


class QalsadiTool(Analyzer):
    tool_name = "qalsadi"

    def __init__(self):
        import qalsadi.lemmatizer as qalsadi_lem

        # Qalsadi lemmatizer is not necessarily thread-safe; keep per-instance
        self._analyzer = qalsadi_lem.Lemmatizer()

    def analyze(self, text: str) -> Dict[str, Any]:
        t0 = time.time()
        tokens_text = simple_word_tokenize(normalize_whitespace(text))
        tokens: List[Dict[str, Any]] = []

        for word in tokens_text:
            result = self._analyzer.lemmatize_text(word)

            pos_ar: str = ""
            lemma: str = word
            stem: str = ""
            unvocalized: str = word

            if result:
                lemma_obj = result[0]
                if isinstance(lemma_obj, str):
                    lemma = lemma_obj or word
                else:
                    pos_ar = getattr(lemma_obj, "type", "") or ""
                    lemma = getattr(lemma_obj, "lemma", word) or word
                    stem = getattr(lemma_obj, "stem", "") or ""
                    unvocalized = getattr(lemma_obj, "unvocalized", word) or word

            tokens.append(
                {
                    "surface": word,
                    "lemma": lemma,
                    "stem": stem,
                    "unvocalized": unvocalized,
                    "pos_ar": pos_ar,
                    "pos": QALSADI_POS_MAP.get(pos_ar, pos_ar or "UNKNOWN"),
                }
            )

        return {
            "tool": "qalsadi",
            "status": "ok",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
            "approach": "rule-based",
            "elapsed": time.time() - t0,
        }

