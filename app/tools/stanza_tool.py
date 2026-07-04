from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

from app.core.tool_registry import unavailable_result
from app.tools.base_tool import BaseTool
from app.utils.helpers import parse_feats
from app.utils.logger import logger, log_time


stanza = None
stanza_pipeline = None
stanza_import_error = None

_NON_CONTENT_POS = {"CCONJ", "SCONJ", "PART", "PUNCT", "SYM"}


def _stanza_resources_dir() -> Path | None:
    candidates = [
        Path(os.environ["STANZA_RESOURCES_DIR"]).expanduser() if os.environ.get("STANZA_RESOURCES_DIR") else None,
        Path.home() / "stanza_resources",
        Path.home() / "AppData" / "Local" / "StanfordNLP" / "stanza" / "resources",
        Path.home() / "AppData" / "Local" / "StanfordNLP" / "stanza" / "Cache" / "1.12.0" / "resources",
    ]
    for candidate in candidates:
        if candidate and (candidate / "ar").exists():
            return candidate
    return None


def _norm_pos(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _select_representative_word(words: List[Any]) -> Any:
    if not words:
        return None

    for word in words:
        if _norm_pos(getattr(word, "upos", None)) not in _NON_CONTENT_POS:
            return word
    return words[0]


def _word_payload(word: Any) -> Dict[str, Any]:
    feats = parse_feats(getattr(word, "feats", None))
    return {
        "id": getattr(word, "id", None),
        "text": getattr(word, "text", None),
        "lemma": getattr(word, "lemma", None),
        "upos": getattr(word, "upos", None),
        "xpos": getattr(word, "xpos", None),
        "deprel": getattr(word, "deprel", None),
        "head": getattr(word, "head", None),
        "gender": feats.get("gender"),
        "number": feats.get("number"),
        "tense": feats.get("tense"),
        "person": feats.get("person"),
        "voice": feats.get("voice"),
        "case": feats.get("case"),
        "definite": feats.get("definite"),
        "aspect": feats.get("aspect"),
    }


def _ensure_imports() -> bool:
    global stanza, stanza_import_error
    if stanza is not None:
        return True
    try:
        import stanza as _stanza

        stanza = _stanza
        stanza_import_error = None
        return True
    except Exception as exc:
        stanza_import_error = str(exc)
        return False


def load_stanza() -> None:
    global stanza_pipeline
    if not _ensure_imports():
        logger.warning("Stanza unavailable: %s", stanza_import_error)
        stanza_pipeline = None
        return
    try:
        processors = os.environ.get("STANZA_PROCESSORS", "tokenize,mwt,pos,lemma")
        pipeline_kwargs = {
            "lang": "ar",
            "processors": processors,
            "verbose": False,
            "use_gpu": False,
        }
        resources_dir = _stanza_resources_dir()
        if resources_dir is not None:
            pipeline_kwargs["dir"] = str(resources_dir)
        download_method = getattr(stanza, "DownloadMethod", None)
        if download_method is not None and hasattr(download_method, "REUSE_RESOURCES"):
            pipeline_kwargs["download_method"] = download_method.REUSE_RESOURCES
        stanza_pipeline = stanza.Pipeline(**pipeline_kwargs)
        logger.info("Stanza loaded")
    except Exception as exc:
        logger.warning("Stanza failed: %s", exc)
        stanza_pipeline = None


def stanza_analyze(text: str) -> Dict[str, Any]:
    global stanza_pipeline
    if not stanza_pipeline:
        load_stanza()

    if not stanza_pipeline:
        return unavailable_result("stanza", stanza_import_error or "Stanza package or Arabic models are not available.", text)

    t0 = time.time()
    try:
        doc = stanza_pipeline(text)
        tokens: List[Dict[str, Any]] = []
        for sentence in doc.sentences:
            word_to_surface: Dict[int, str] = {}
            if getattr(sentence, "tokens", None):
                for token in sentence.tokens:
                    surface = getattr(token, "text", None)
                    if surface is None and getattr(token, "words", None):
                        surface = "".join(getattr(w, "text", "") or "" for w in token.words)
                    if not surface:
                        continue
                    for word in getattr(token, "words", []) or []:
                        try:
                            word_id = int(getattr(word, "id", 0))
                        except Exception:
                            word_id = 0
                        if word_id > 0:
                            word_to_surface[word_id] = str(surface)

            if not word_to_surface:
                for word in sentence.words:
                    try:
                        word_id = int(getattr(word, "id", 0))
                    except Exception:
                        word_id = 0
                    if word_id > 0:
                        word_to_surface[word_id] = getattr(word, "text", None) or ""

            token_iter = getattr(sentence, "tokens", None) or []
            if token_iter:
                for token in token_iter:
                    words = list(getattr(token, "words", []) or [])
                    surface = getattr(token, "text", None) or "".join(getattr(w, "text", "") or "" for w in words)
                    if not surface:
                        continue

                    rep_word = _select_representative_word(words) or (words[0] if words else None)
                    feats = parse_feats(getattr(rep_word, "feats", None)) if rep_word else {}
                    head = None
                    head_text = None
                    deprel = None
                    if rep_word is not None:
                        try:
                            head = int(rep_word.head) if rep_word.head and str(rep_word.head) != "0" else None
                        except Exception:
                            head = None
                        deprel = getattr(rep_word, "deprel", None)
                        if head and 1 <= head <= len(sentence.words):
                            head_text = word_to_surface.get(head) or sentence.words[head - 1].text
                        elif str(getattr(rep_word, "head", None)) == "0":
                            head_text = "root"

                    tokens.append(
                        {
                            "surface": surface,
                            "lemma": getattr(rep_word, "lemma", None) if rep_word else None,
                            "upos": getattr(rep_word, "upos", None) if rep_word else None,
                            "pos": getattr(rep_word, "upos", None) if rep_word else None,
                            "xpos": getattr(rep_word, "xpos", None) if rep_word else None,
                            "gender": feats.get("gender"),
                            "number": feats.get("number"),
                            "tense": feats.get("tense"),
                            "person": feats.get("person"),
                            "voice": feats.get("voice"),
                            "case": feats.get("case"),
                            "definite": feats.get("definite"),
                            "aspect": feats.get("aspect"),
                            "mwt": len(words) > 1,
                            "mwt_words": [_word_payload(w) for w in words],
                            "dependency": {"head": head, "head_text": head_text, "deprel": deprel},
                            "analyses": [
                                {
                                    "lemma": getattr(rep_word, "lemma", None) if rep_word else None,
                                    "root": None,
                                    "pos": getattr(rep_word, "upos", None) if rep_word else None,
                                    "gender": feats.get("gender"),
                                    "number": feats.get("number"),
                                    "tense": feats.get("tense"),
                                    "gloss": None,
                                }
                            ],
                        }
                    )
                continue

            for word in sentence.words:
                feats = parse_feats(word.feats)
                head = int(word.head) if word.head and str(word.head) != "0" else None
                head_text = None
                if head and 1 <= head <= len(sentence.words):
                    head_text = word_to_surface.get(head) or sentence.words[head - 1].text
                elif str(word.head) == "0":
                    head_text = "root"

                tokens.append(
                    {
                        "surface": word.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "pos": word.upos,
                        "xpos": word.xpos,
                        "gender": feats.get("gender"),
                        "number": feats.get("number"),
                        "tense": feats.get("tense"),
                        "person": feats.get("person"),
                        "voice": feats.get("voice"),
                        "case": feats.get("case"),
                        "definite": feats.get("definite"),
                        "aspect": feats.get("aspect"),
                        "mwt": False,
                        "mwt_words": [],
                        "dependency": {"head": head, "head_text": head_text, "deprel": word.deprel},
                        "analyses": [
                            {
                                "lemma": word.lemma,
                                "root": None,
                                "pos": word.upos,
                                "gender": feats.get("gender"),
                                "number": feats.get("number"),
                                "tense": feats.get("tense"),
                                "gloss": None,
                            }
                        ],
                    }
                )

        log_time("stanza", text, time.time() - t0)
        return {"tool": "stanza", "status": "ok", "input": text, "word_count": len(tokens), "tokens": tokens}
    except Exception as exc:
        logger.warning("[STANZA] error: %s", exc)
        return {"tool": "stanza", "status": "error", "reason": str(exc), "input": text, "word_count": 0, "tokens": []}


class StanzaTool(BaseTool):
    tool_name = "stanza"

    def is_loaded(self) -> bool:
        return stanza_pipeline is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return stanza_analyze(text)
