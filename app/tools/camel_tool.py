from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize

from app.tools.base_tool import BaseTool
from app.utils.helpers import (
    augment_root,
    confidence_bucket,
    correct_number,
    map_pos,
    normalize_pos_for_compare,
    simplify_gloss,
    strip_diacritics,
    clean_root,
)
from app.utils.constants import (
    ASPECT_MAP,
    GENDER_MAP,
    NUMBER_MAP,
    POS_MAP,
    POS_UNIFIED,
    WEAK_VERB_ROOTS,
    SINGLE_LETTER_PARTICLES,
    FUSION_WEIGHTS,
    GLOSS_NOISE,
    KNOWN_FIXES,
)

from app.utils.logger import logger, log_time


camel_db = None
camel_disambiguator = None


def load_camel() -> None:
    global camel_db, camel_disambiguator
    try:
        camel_db = MorphologyDB.builtin_db()
        camel_disambiguator = MLEDisambiguator.pretrained()
        logger.info("✅ CAMeL loaded")
    except Exception as e:
        logger.error(f"❌ CAMeL failed: {e}")
        camel_db = None
        camel_disambiguator = None


def camel_analyze(text: str) -> Dict[str, Any]:
    if not camel_disambiguator or not camel_db:
        return {"tool": "camel", "status": "failed", "error": "CAMeL not loaded", "tokens": []}

    t0 = time.time()
    try:
        tokens = simple_word_tokenize(text)
        results = camel_disambiguator.disambiguate(tokens)

        token_outputs: List[Dict[str, Any]] = []
        for token, disambig in zip(tokens, results):
            analyses: List[Dict[str, Any]] = []
            segs = [token]

            for a in disambig.analyses[:3]:
                features = a.analysis
                score = round(float(a.score), 4)

                raw_root = clean_root(features.get("root"))
                raw_pos = features.get("pos")
                raw_lemma = features.get("lex")
                raw_gloss = features.get("gloss")

                aug_root, root_type, part_gloss = augment_root(raw_root or "", raw_lemma or "", raw_pos or "", token)
                clean_gloss = part_gloss or simplify_gloss(raw_gloss)

                corrections: List[str] = []
                if aug_root != raw_root:
                    corrections.append("root")
                if clean_gloss != raw_gloss:
                    corrections.append("gloss")

                corrected_num, num_fixed = correct_number(token, NUMBER_MAP.get(features.get("num")), segs, map_pos(raw_pos))
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
    except Exception as e:
        logger.error(f"[CAMEL] error: {e}")
        return {"tool": "camel", "status": "error", "error": str(e), "tokens": []}


class CamelTool(BaseTool):
    tool_name = "camel"

    def is_loaded(self) -> bool:
        return camel_disambiguator is not None and camel_db is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return camel_analyze(text)

