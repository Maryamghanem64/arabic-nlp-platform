from __future__ import annotations

import time
from typing import Any, Dict, List

from app.core.tool_registry import unavailable_result
from app.tools.base_tool import BaseTool
from app.utils.constants import ASPECT_MAP, GENDER_MAP, NUMBER_MAP
from app.utils.helpers import augment_root, clean_root, confidence_bucket, correct_number, map_pos, simplify_gloss, normalize_lemma_for_compare
from app.tools.camel_root_patch import patch_camel_root
from app.utils.logger import logger, log_time


MorphologyDB = None
MLEDisambiguator = None
simple_word_tokenize = None
camel_db = None
camel_disambiguator = None
camel_import_error = None


GLOSS_OVERRIDE = {
    "فصل": {
        "gloss": "class/chapter/semester",
        "bad_glosses": {"discharge", "separate", "split", "division"},
    },
    "مكتبة": {
        "gloss": "library/bookstore",
        "bad_glosses": {"translation", "wrong translation", "book"},
    },
    "معلم": {
        "gloss": "teacher",
        "bad_glosses": {"learned man", "scholar", "disciple"},
    },
    "طالب": {
        "gloss": "student",
        "bad_glosses": {"seeker", "requester", "demanding"},
    },
    "مدرسة": {
        "gloss": "school",
        "bad_glosses": {"lesson", "teacher"},
    },
    "جامعة": {
        "gloss": "university",
        "bad_glosses": {"collection", "gathering"},
    },
    "كتاب": {
        "gloss": "book",
        "bad_glosses": {"writing", "decree", "letter"},
    },
}


def _override_gloss(lemma: str | None, gloss: str | None) -> str | None:
    lemma_key = normalize_lemma_for_compare(lemma)
    if not lemma_key or lemma_key not in GLOSS_OVERRIDE:
        return gloss

    override = GLOSS_OVERRIDE[lemma_key]
    override_gloss = override["gloss"]
    if not gloss:
        return override_gloss

    simplified = normalize_lemma_for_compare(gloss)
    bad_glosses = {normalize_lemma_for_compare(item) for item in override.get("bad_glosses", set())}
    if simplified in bad_glosses or simplified not in {normalize_lemma_for_compare(override_gloss)}:
        return override_gloss
    return gloss


def _ensure_imports() -> bool:
    global MorphologyDB, MLEDisambiguator, simple_word_tokenize, camel_import_error
    if MorphologyDB and MLEDisambiguator and simple_word_tokenize:
        return True
    try:
        # Ensure emoji compatibility before importing camel-tools.
        # Some camel-tools versions import emoji.EMOJI_DATA at import-time.
        from backend.utils.emoji_compat import ensure_emoji_emoji_data

        ensure_emoji_emoji_data()

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
                # Repair 3: hamza must NOT be stripped before root classification
                patched_root, patched_type = patch_camel_root(raw_root or "")
                raw_root = patched_root
                # Override root_type to match patched classification
                root_type = patched_type
                raw_pos = features.get("pos")
                raw_lemma = features.get("lex")
                raw_gloss = features.get("gloss")
                aug_root, root_type2, part_gloss = augment_root(raw_root or "", raw_lemma or "", raw_pos or "", token)
                # Repair 3: keep patched root_type instead of augment_root's type
                root_type = patched_type
                clean_gloss = part_gloss or simplify_gloss(raw_gloss)
                clean_gloss = _override_gloss(raw_lemma, clean_gloss)
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
