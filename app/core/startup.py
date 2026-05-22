from __future__ import annotations

import csv
import io
import json
import logging
import re
import threading
import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from farasa.segmenter import FarasaSegmenter

# stanza/torch can crash on import in some environments.
# Keep the app importable by guarding stanza import/initialization.
try:
    import stanza  # type: ignore
except Exception as _stanza_import_error:  # pragma: no cover
    stanza = None  # type: ignore


from app.services.cache_service import cached_analyze, clear_cache
from app.utils.constants import (
    ASPECT_MAP,
    FUSION_WEIGHTS,
    GENDER_MAP,
    GOLD_DATASET,
    NUMBER_MAP,
    QALSADI_POS_MAP,
    SINGLE_LETTER_PARTICLES,
    WEAK_VERB_ROOTS,
    POS_MAP,
    POS_UNIFIED,
    KNOWN_FIXES,
    GLOSS_NOISE,
)

from app.utils.helpers import (
    augment_root,
    classify_conflict,
    confidence_bucket,
    correct_number,
    map_pos,
    normalize_pos_for_compare,
    simplify_gloss,
    strip_diacritics,
    clean_root,
    parse_feats,
)

from app.tools.camel_tool import camel_analyze as camel_analyze  # type: ignore
from app.tools.farasa_tool import farasa_analyze as farasa_analyze  # type: ignore
from app.tools.stanza_tool import stanza_analyze as stanza_analyze  # type: ignore
from app.tools.qalsadi_tool import qalsadi_analyze as qalsadi_analyze  # type: ignore

from app.services.fusion_service import fusion_system as fusion_system  # type: ignore
from app.services.eval_service import evaluate_tools as evaluate_tools  # type: ignore

from app.utils.logger import logger


def log_time(tool: str, text: str, elapsed: float):
    logger.info(f"[{tool.upper()}] '{text[:30]}' -> {elapsed:.3f}s")


# ============================================================
# Load Resources (kept as module globals)
# ============================================================

logger.info("Loading NLP resources...")

camel_db = None
camel_disambiguator = None
try:
    camel_db = MorphologyDB.builtin_db()
    camel_disambiguator = MLEDisambiguator.pretrained()
    logger.info("CAMeL Tools loaded")
except Exception as e:
    logger.error(f"CAMeL failed: {e}")

farasa_segmenter = None
try:
    farasa_segmenter = FarasaSegmenter(interactive=False)
    logger.info("Farasa loaded")
except Exception as e:
    logger.error(f"Farasa failed: {e}")

stanza_pipeline = None
try:
    stanza_pipeline = stanza.Pipeline(
        "ar",
        processors="tokenize,mwt,pos,lemma,depparse",
        verbose=False,
    )
    logger.info("Stanza loaded")
except Exception as e:
    logger.error(f"Stanza failed: {e}")

qalsadi_analyzer = None
qalsadi_thread_local = threading.local()
try:
    import qalsadi.lemmatizer as qalsadi_lem

    qalsadi_analyzer = qalsadi_lem.Lemmatizer()
    logger.info("Qalsadi loaded")
except Exception as e:
    logger.error(f"Qalsadi failed: {e}")

logger.info("Resource loading complete")


# ============================================================
# Parallel Runner (identical behavior)
# ============================================================


def run_all_tools(text: str):
    """Run CAMeL + Farasa + Stanza + Qalsadi in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_camel = executor.submit(cached_analyze, camel_analyze, text)
        f_farasa = executor.submit(cached_analyze, farasa_analyze, text)
        f_stanza = executor.submit(cached_analyze, stanza_analyze, text)
        f_qalsadi = executor.submit(cached_analyze, qalsadi_analyze, text)
        camel_res = f_camel.result()
        farasa_res = f_farasa.result()
        stanza_res = f_stanza.result()
        qalsadi_res = f_qalsadi.result()
    return camel_res, farasa_res, stanza_res, qalsadi_res


# Re-export for routers
__all__ = [
    "camel_analyze",
    "farasa_analyze",
    "stanza_analyze",
    "qalsadi_analyze",
    "cached_analyze",
    "clear_cache",
    "run_all_tools",
    "fusion_system",
    "evaluate_tools",
]

