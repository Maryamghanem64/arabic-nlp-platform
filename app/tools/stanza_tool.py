from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    import stanza
except Exception:  # pragma: no cover - keeps API importable without stanza installed
    stanza = None

from app.tools.base_tool import BaseTool
from app.utils.helpers import parse_feats
from app.utils.logger import logger, log_time


stanza_pipeline = None


def load_stanza() -> None:
    global stanza_pipeline
    try:
        if stanza is None:
            raise RuntimeError("stanza is not installed")
        stanza_pipeline = stanza.Pipeline(
            "ar",
            processors="tokenize,mwt,pos,lemma,depparse",
            verbose=False,
        )
        logger.info("✅ Stanza loaded")
    except Exception as e:
        logger.error(f"❌ Stanza failed: {e}")
        stanza_pipeline = None


def stanza_analyze(text: str) -> Dict[str, Any]:
    global stanza_pipeline
    if not stanza_pipeline:
        load_stanza()

    if not stanza_pipeline:
        return {"tool": "stanza", "status": "failed", "error": "Stanza not loaded", "tokens": []}

    t0 = time.time()
    try:
        doc = stanza_pipeline(text)
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
                        "upos": word.upos,
                        "xpos": word.xpos,
                        "gender": feats.get("gender"),
                        "number": feats.get("number"),
                        "tense": feats.get("tense"),
                        "person": feats.get("person"),
                        "voice": feats.get("voice"),
                        "case": feats.get("case"),
                        "definite": feats.get("definite"),
                        "aspect": feats.get("aspect"),
                        "dependency": {
                            "head": head,
                            "head_text": head_text,
                            "deprel": word.deprel,
                        },
                    }
                )

        log_time("stanza", text, time.time() - t0)
        return {"tool": "stanza", "status": "ok", "input": text, "word_count": len(tokens), "tokens": tokens}
    except Exception as e:
        logger.error(f"[STANZA] error: {e}")
        return {"tool": "stanza", "status": "error", "error": str(e), "tokens": []}


class StanzaTool(BaseTool):
    tool_name = "stanza"

    def is_loaded(self) -> bool:
        return stanza_pipeline is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return stanza_analyze(text)

