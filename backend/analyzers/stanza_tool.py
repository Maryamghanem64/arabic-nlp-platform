from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import stanza

from backend.analyzers.base import Analyzer
from backend.utils.text_norm import normalize_whitespace


def parse_feats(feats: Optional[str]) -> Dict[str, str]:
    if not feats:
        return {}
    result: Dict[str, str] = {}
    for pair in feats.split("|"):
        if "=" in pair:
            key, val = pair.split("=", 1)
            key = key.lower()
            val_lower = val.lower()
            if "gender" in key:
                val = "masc" if "masc" in val_lower else "fem" if "fem" in val_lower else val_lower
            elif "number" in key:
                val = "sing" if "sing" in val_lower else "dual" if "dual" in val_lower else "plur" if "plur" in val_lower else val_lower
            elif "aspect" in key:
                val = "perf" if "perf" in val_lower else "impf" if "impf" in val_lower else val_lower
                result["tense"] = val
            elif "voice" in key:
                val = "act" if "act" in val_lower else "pass" if "pass" in val_lower else val_lower
            elif "case" in key:
                val = val_lower[:3]
            elif "definite" in key:
                val = "yes" if val_lower in ("def", "yes") else "no"
            else:
                val = val_lower
            result[key] = val
    return result


def normalize_pos(upos: Optional[str]) -> Optional[str]:
    if not upos:
        return None
    return upos.upper()


class StanzaTool(Analyzer):
    tool_name = "stanza"

    def __init__(self):
        self._pipeline = stanza.Pipeline(
            "ar",
            processors="tokenize,mwt,pos,lemma,depparse",
            verbose=False,
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        t0 = time.time()
        doc = self._pipeline(normalize_whitespace(text))

        tokens: List[Dict[str, Any]] = []
        for sentence in doc.sentences:
            for word in sentence.words:
                feats = parse_feats(word.feats)

                head = int(word.head) if word.head and str(word.head) != "0" else None
                head_text = None
                if head and 1 <= head <= len(sentence.words):
                    head_text = sentence.words[head - 1].text
                elif str(word.head) == "0":
                    head_text = "root"

                tokens.append(
                    {
                        "surface": word.text,
                        "lemma": word.lemma,
                        "pos": normalize_pos(word.upos),
                        "upos": word.upos,
                        "xpos": word.xpos,
                        "gender": feats.get("gender"),
                        "number": feats.get("number"),
                        "tense": feats.get("tense"),
                        "case": feats.get("case"),
                        "definite": feats.get("definite"),
                        "aspect": feats.get("aspect"),
                        "dependency": {"head": head, "head_text": head_text, "deprel": word.deprel},
                    }
                )

        return {
            "tool": "stanza",
            "status": "ok",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
            "elapsed": time.time() - t0,
        }

