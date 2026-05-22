from __future__ import annotations

import time
from typing import Any, Dict, List

from app.core.tool_registry import unavailable_result
from app.tools.base_tool import BaseTool
from app.utils.constants import ASPECT_MAP, GENDER_MAP, NUMBER_MAP
from app.utils.helpers import augment_root, clean_root, confidence_bucket, correct_number, map_pos, simplify_gloss
from app.utils.logger import logger, log_time


MorphologyDB = None
MLEDisambiguator = None
simple_word_tokenize = None
camel_db = None
camel_disambiguator = None
camel_import_error = None


def _ensure_imports() -> bool:
    global MorphologyDB, MLEDisambiguator, simple_word_tokenize, camel_import_error
    if MorphologyDB and MLEDisambiguator and simple_word_tokenize:
        return True
    try:
        from camel_tools.disambig.mle import MLEDisambiguator as _MLEDisambiguator
        from camel_tools.morphology.database import MorphologyDB as _MorphologyDB
        from camel_tools.tokenizers.word import simple_word_tokenize as _simple_word_tokenize

        MorphologyDB = _MorphologyDB
        MLEDisambiguator = _MLEDisambiguator
        simple_word_tokenize = _simple_word_tokenize
        camel_import_error = None
        return True
    except Exception as exc:
        camel_import_error = str(exc)
        return False


def load_camel() -> None:
    global camel_db, camel_disambiguator
    if not _ensure_imports():
        logger.warning("CAMeL unavailable: %s", camel_import_error)
        camel_db = None
        camel_disambiguator = None
        return
    try:
        camel_db = MorphologyDB.builtin_db()
        camel_disambiguator = MLEDisambiguator.pretrained()
        logger.info("CAMeL loaded")
    except Exception as exc:
        logger.warning("CAMeL failed: %s", exc)
        camel_db = None
        camel_disambiguator = None


def camel_analyze(text: str) -> Dict[str, Any]:
    global camel_db, camel_disambiguator
    if not camel_disambiguator or not camel_db:
        load_camel()

    if not camel_disambiguator or not camel_db:
        return unavailable_result("camel", camel_import_error or "CAMeL package/model is not available.", text)

    t0 = time.time()
    try:
        tokens = simple_word_tokenize(text)
        results = camel_disambiguator.disambiguate(tokens)
        token_outputs: List[Dict[str, Any]] = []

        for token, disambig in zip(tokens, results):
            analyses: List[Dict[str, Any]] = []
            segs = [token]
            for item in disambig.analyses[:3]:
                features = item.analysis
                score = round(float(item.score), 4)
                raw_root = clean_root(features.get("root"))
                raw_pos = features.get("pos")
                raw_lemma = features.get("lex")
                raw_gloss = features.get("gloss")
                aug_root, root_type, part_gloss = augment_root(raw_root or "", raw_lemma or "", raw_pos or "", token)
                clean_gloss = part_gloss or simplify_gloss(raw_gloss)
                corrected_num, num_fixed = correct_number(token, NUMBER_MAP.get(features.get("num")), segs, map_pos(raw_pos))
                corrections: List[str] = []
                if aug_root != raw_root:
                    corrections.append("root")
                if clean_gloss != raw_gloss:
                    corrections.append("gloss")
                if num_fixed:
                    corrections.append("number")

                analyses.append(
                    {
                        "lemma": raw_lemma,
                        "root": aug_root,
                        "root_type": root_type,
                        "pos": map_pos(raw_pos),
                        "gender": GENDER_MAP.get(features.get("gen")),
                        "number": corrected_num,
                        "tense": ASPECT_MAP.get(features.get("asp")),
                        "gloss": clean_gloss,
                        "confidence": score,
                        "confidence_level": confidence_bucket(score),
                        "corrections": corrections,
                    }
                )
            token_outputs.append({"surface": token, "analyses": analyses, "segmentation": segs})

        log_time("camel", text, time.time() - t0)
        return {"tool": "camel", "status": "ok", "input": text, "word_count": len(token_outputs), "tokens": token_outputs}
    except Exception as exc:
        logger.warning("[CAMEL] error: %s", exc)
        return {"tool": "camel", "status": "error", "reason": str(exc), "input": text, "word_count": 0, "tokens": []}


class CamelTool(BaseTool):
    tool_name = "camel"

    def is_loaded(self) -> bool:
        return camel_disambiguator is not None and camel_db is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return camel_analyze(text)
