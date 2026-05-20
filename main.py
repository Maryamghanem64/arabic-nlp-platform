# ============================================================
# Arabic NLP Comparative Platform (v8.3 - Pro + Qalsadi)
# ============================================================
# Changes:
#   - CAMeL Tools, Farasa, and Stanza preserved
#   - Qalsadi added as 4th lightweight rule-based tool
#   - SinaTools kept only as Future Work due to model size/loading cost
#   - Parallel execution, caching, fusion, evaluation, and export preserved
# ============================================================
# pip install qalsadi

from fastapi import FastAPI, HTTPException, Query

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import logging
import re
import time
import csv
import io
import json
import concurrent.futures
import threading


from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from farasa.segmenter import FarasaSegmenter
import stanza

# ============================================================
# Logging - structured with timing
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_time(tool: str, text: str, elapsed: float):
    logger.info(f"[{tool.upper()}] '{text[:30]}' -> {elapsed:.3f}s")

try:
    from backend.analyzers import PARTNER_TOOLS, get_all_partner_statuses
    PARTNER_TOOLS_LOADED = True
    logger.info("✅ Partner tools registry loaded")
except ImportError as e:
    PARTNER_TOOLS = {}
    PARTNER_TOOLS_LOADED = False
    logger.warning(f"⚠️ Partner tools not found: {e}")

# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Arabic NLP Comparative Platform",
    version="8.3",
    description="Compare Arabic NLP tools: CAMeL Tools, Farasa, Stanza, Qalsadi"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- SinaTools -- FUTURE WORK ------------------------------
# SinaTools was evaluated but excluded due to large model size (~880MB)
# and slow loading time on Windows. Planned for future microservice deployment.
# SinaTools -> planned Future Work (large model ~880MB)

# ============================================================
# Load Resources
# ============================================================
# NOTE: Resources moved to modular backend analyzers; keep only minimal app wiring here.

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
        verbose=False
    )
    logger.info("Stanza loaded")
except Exception as e:
    logger.error(f"Stanza failed: {e}")

qalsadi_lem = None
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
# Maps
# ============================================================

ASPECT_MAP = {"p": "past", "i": "present", "c": "imperative", "na": None}
GENDER_MAP = {"m": "masculine", "f": "feminine", "na": None}
NUMBER_MAP = {"s": "singular", "d": "dual", "p": "plural", "na": None}

POS_MAP = {
    "noun": "NOUN", "verb": "VERB", "adj": "ADJECTIVE",
    "prep": "ADPOSITION", "pron": "PRONOUN", "adv": "ADVERB",
    "conj": "CONJUNCTION", "part": "PARTICLE", "punc": "PUNCTUATION"
}

POS_UNIFIED = {
    "NOUN": "NOUN", "VERB": "VERB", "ADJECTIVE": "ADJ",
    "ADPOSITION": "ADP", "PRONOUN": "PRON", "ADVERB": "ADV",
    "CONJUNCTION": "CCONJ", "PARTICLE": "PART", "PUNCTUATION": "PUNCT",
    "CONJ_SUB": "SCONJ",
}

QALSADI_POS_MAP = {
    "فعل": "VERB",
    "اسم": "NOUN",
    "صفة": "ADJ",
    "حرف": "PART",
    "ضمير": "PRON",
    "ظرف": "ADV",

    # Participles / verbal nouns (map to NOUN as requested)
    "اسم فاعل": "NOUN",
    "اسم مفعول": "NOUN",
    "مصدر": "NOUN",

    # Stopword
    "STOPWORD": "STOP",
}


WEAK_VERB_ROOTS = {
    "ق.ل": "ق.و.ل", "ب.ع": "ب.ي.ع", "ن.م": "ن.و.م", "ص.م": "ص.و.م",
    "خ.ف": "خ.و.ف", "ز.ر": "ز.و.ر", "ط.ر": "ط.ي.ر", "س.ر": "س.ي.ر",
    "ع.د": "ع.و.د", "ج.ء": "ج.ي.ء", "ش.ء": "ش.ي.ء", "ك.ل": "أ.ك.ل",
}

SINGLE_LETTER_PARTICLES = {
    "ب": {"root": "ب", "gloss": "with/by", "pos": "ADPOSITION"},
    "ل": {"root": "ل", "gloss": "to/for", "pos": "ADPOSITION"},
    "و": {"root": "و", "gloss": "and", "pos": "CONJUNCTION"},
    "ف": {"root": "ف", "gloss": "then/so", "pos": "CONJUNCTION"},
    "ك": {"root": "ك", "gloss": "like/as", "pos": "ADPOSITION"},
}

GLOSS_NOISE = {
    "my", "your", "his", "her", "its", "our", "their",
    "i", "me", "you", "he", "him", "she", "it", "us", "them", "we",
    "the", "a", "an", "of", "for", "with", "that", "which", "who", "whose", "what",
    "defgen", "defnom", "defacc", "indef", "def", "one", "two", "three",
    "fempl", "mascpl", "femsg", "mascsg", "masc", "fem",
}

# Adaptive fixes: known problematic words.
KNOWN_FIXES: Dict[str, Dict] = {
    # "في": {"pos": "ADP"},
}

FUSION_WEIGHTS = {
    "lemma": {"camel": 3, "stanza": 1},
    "pos": {"camel": 2, "stanza": 2},
    "morphology": {"camel": 3, "stanza": 1},
    "segmentation": {"farasa": 3},
    "syntax": {"stanza": 3},
}

# ============================================================
# In-Memory Cache
# ============================================================

_CACHE: Dict[str, Any] = {}


def cached_analyze(func, text: str) -> Dict[str, Any]:
    key = f"{func.__name__}::{text}"
    if key in _CACHE:
        logger.info(f"[CACHE] HIT - {func.__name__}")
        return _CACHE[key]
    result = func(text)
    _CACHE[key] = result
    return result


def clear_cache():
    _CACHE.clear()


# ============================================================
# Helpers
# ============================================================

def map_pos(pos: Optional[str]) -> Optional[str]:
    return POS_MAP.get(pos, pos.upper()) if pos else None


def clean_root(root: Optional[str]) -> Optional[str]:
    return root.replace("#.", "").replace(".#", "").strip() if root else None


def confidence_bucket(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def simplify_gloss(gloss: Optional[str]) -> Optional[str]:
    if not gloss:
        return None
    simplified = re.sub(r"[\[\]().;]", "", gloss.split(";")[0]).strip()
    simplified = simplified.replace("the+", "").replace("+", " ").replace("_", " ")
    words = simplified.split()
    clean = [w for w in words if w.lower() not in GLOSS_NOISE]
    result = " ".join(clean).strip()
    return result if result else None


def strip_diacritics(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[\u064B-\u065F\u0670]", "", text)


def augment_root(root: str, lemma: str, pos: str, surface: str = "") -> tuple:
    if not root:
        return root, "unknown", None
    if surface in SINGLE_LETTER_PARTICLES:
        p = SINGLE_LETTER_PARTICLES[surface]
        return p["root"], "monoliteral", p["gloss"]
    parts = root.split(".")
    if len(parts) >= 3:
        return root, "triliteral", None
    if len(parts) == 2 and pos == "verb" and root in WEAK_VERB_ROOTS:
        aug = WEAK_VERB_ROOTS[root]
        logger.info(f"[ROOT] augmented: {root} -> {aug}")
        return aug, "triliteral_weak", None
    if len(parts) == 2:
        return root, "biliteral", None
    if len(parts) == 1:
        return root, "monoliteral", None
    return root, "unknown", None


def correct_number(surface: str, number: str, segmentation: List[str], pos: str) -> tuple:
    if not number or pos != "NOUN":
        return number, False
    if (
        number == "dual"
        and surface.endswith("تي")
        and len(segmentation) >= 2
        and segmentation[-2] == "ت"
        and segmentation[-1] == "ي"
    ):
        logger.info(f"[NUMBER] corrected: {surface} dual -> singular")
        return "singular", True
    return number, False


def parse_feats(feats: Optional[str]) -> Dict[str, str]:
    if not feats:
        return {}
    result = {}
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


def normalize_pos_for_compare(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    return POS_UNIFIED.get(pos.upper(), pos.upper())


def classify_conflict(feature: str, val_a: Any, val_b: Any) -> Dict[str, str]:
    severity_map = {
        "pos": "high",
        "lemma": "medium",
        "root": "medium",
        "tense": "low",
        "gender": "low",
        "number": "low",
    }
    return {
        "feature":  feature,
        "tool_a":   str(val_a),
        "tool_b":   str(val_b),
        "severity": severity_map.get(feature, "low"),
        "type":     f"{feature}_mismatch",
    }





# ============================================================
# Tool Functions
# ============================================================

def camel_analyze(text: str) -> Dict[str, Any]:
    if not camel_disambiguator or not camel_db:
        return {"tool": "camel", "status": "failed", "error": "CAMeL not loaded", "tokens": []}
    t0 = time.time()
    try:
        tokens = simple_word_tokenize(text)
        results = camel_disambiguator.disambiguate(tokens)
        token_outputs = []

        for token, disambig in zip(tokens, results):
            analyses = []
            segs = [token]

            for a in disambig.analyses[:3]:
                features = a.analysis
                score = round(a.score, 4)
                raw_root = clean_root(features.get("root"))
                raw_pos = features.get("pos")
                raw_lemma = features.get("lex")
                raw_gloss = features.get("gloss")
                aug_root, root_type, part_gloss = augment_root(
                    raw_root or "", raw_lemma or "", raw_pos or "", token
                )
                clean_gloss = part_gloss or simplify_gloss(raw_gloss)
                corrections = []
                if aug_root != raw_root:
                    corrections.append("root")
                if clean_gloss != raw_gloss:
                    corrections.append("gloss")
                corrected_num, num_fixed = correct_number(
                    token, NUMBER_MAP.get(features.get("num")), segs, map_pos(raw_pos)
                )
                if num_fixed:
                    corrections.append("number")

                analyses.append({
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
                })

            token_outputs.append({
                "surface": token,
                "analyses": analyses,
                "segmentation": segs,
            })

        log_time("camel", text, time.time() - t0)
        return {
            "tool": "camel",
            "status": "ok",
            "input": text,
            "word_count": len(token_outputs),
            "tokens": token_outputs
        }
    except Exception as e:
        logger.error(f"[CAMEL] error: {e}")
        return {"tool": "camel", "status": "error", "error": str(e), "tokens": []}


def farasa_analyze(text: str) -> Dict[str, Any]:
    if not farasa_segmenter:
        return {"tool": "farasa", "status": "failed", "error": "Farasa not loaded", "tokens": []}
    t0 = time.time()
    try:
        segmented = farasa_segmenter.segment(text)
        raw_tokens = simple_word_tokenize(text)
        raw_segs = segmented.split()
        token_outputs = []

        for i, token in enumerate(raw_tokens):
            seg = raw_segs[i] if i < len(raw_segs) else token
            parts = [p for p in seg.split("+") if p]
            token_outputs.append({
                "surface": token,
                "analyses": [],
                "segmentation": parts,
            })

        log_time("farasa", text, time.time() - t0)
        return {
            "tool": "farasa",
            "status": "ok",
            "input": text,
            "word_count": len(token_outputs),
            "segmented_text": segmented,
            "tokens": token_outputs
        }
    except Exception as e:
        logger.error(f"[FARASA] error: {e}")
        return {"tool": "farasa", "status": "error", "error": str(e), "tokens": []}


def stanza_analyze(text: str) -> Dict[str, Any]:
    if not stanza_pipeline:
        return {"tool": "stanza", "status": "failed", "error": "Stanza not loaded", "tokens": []}
    t0 = time.time()
    try:
        doc = stanza_pipeline(text)
        tokens = []

        for sentence in doc.sentences:
            for word in sentence.words:
                feats = parse_feats(word.feats)
                head = int(word.head) if word.head and str(word.head) != "0" else None
                head_text = None
                if head and 1 <= head <= len(sentence.words):
                    head_text = sentence.words[head - 1].text
                elif str(word.head) == "0":
                    head_text = "root"

                tokens.append({
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
                    }
                })

        log_time("stanza", text, time.time() - t0)
        return {
            "tool": "stanza",
            "status": "ok",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens
        }
    except Exception as e:
        logger.error(f"[STANZA] error: {e}")
        return {"tool": "stanza", "status": "error", "error": str(e), "tokens": []}


def qalsadi_analyze(text: str) -> Dict[str, Any]:
    """Qalsadi rule-based Arabic lemmatizer.
    lemmatize_text(text) returns flat list of strings.
    Verified: l.lemmatize_text('كتب الطالب') → ['تب', 'طالب']
    Note: Qalsadi provides lemma only — no POS tagging.
    """
    if not qalsadi_analyzer:
        return {
            "tool": "qalsadi",
            "status": "failed",
            "error": "Qalsadi not loaded",
            "tokens": []
        }
    t0 = time.time()
    try:
        tokens_text = simple_word_tokenize(text)
        lemmas = qalsadi_analyzer.lemmatize_text(text)
        tokens = []
        for i, word in enumerate(tokens_text):
            lemma = lemmas[i] if i < len(lemmas) else word
            tokens.append({
                "surface": word,
                "lemma":   lemma,
                "pos":     None,
                "stem":    None,
                "pos_ar":  None,
            })
        log_time("qalsadi", text, time.time() - t0)
        return {
            "tool":       "qalsadi",
            "status":     "ok",
            "approach":   "rule-based lemmatization",
            "input":      text,
            "word_count": len(tokens),
            "tokens":     tokens
        }
    except Exception as e:
        logger.error(f"[QALSADI] error: {e}")
        return {
            "tool":   "qalsadi",
            "status": "error",
            "error":  str(e),
            "tokens": []
        }



# ============================================================
# Parallel Runner
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


# ============================================================
# Fusion - Scoring-Based
# ============================================================

def score_pos(camel_pos_raw, stanza_pos_raw) -> tuple:
    """Returns (final_pos, source, score, notes)."""
    camel_pos = normalize_pos_for_compare(camel_pos_raw)
    stanza_pos = stanza_pos_raw.upper() if stanza_pos_raw else None
    notes = []

    if camel_pos == stanza_pos and camel_pos:
        return camel_pos, "agreement", 4, notes

    if camel_pos and stanza_pos:
        notes.append(f"POS conflict: camel={camel_pos} stanza={stanza_pos}")

        camel_score = FUSION_WEIGHTS["pos"].get("camel", 0)
        stanza_score = FUSION_WEIGHTS["pos"].get("stanza", 0)

        if camel_pos in ["ADP", "SCONJ", "CCONJ", "PART"]:
            camel_score += 1

        if stanza_pos == "X":
            stanza_score -= 2

        if camel_score >= stanza_score:
            return camel_pos, "camel_scored", camel_score, notes
        return stanza_pos, "stanza_scored", stanza_score, notes

    if camel_pos:
        return camel_pos, "camel_only", 1, notes
    if stanza_pos:
        return stanza_pos, "stanza_only", 1, notes
    return None, "none", 0, notes


def fuse_confidence(camel_score: float, pos_source: str) -> tuple:
    """Unified confidence metric."""
    bonus = 0.1 if pos_source == "agreement" else (-0.1 if pos_source in ("camel_only", "stanza_only") else 0.0)
    final = round(min(1.0, camel_score + bonus), 3)
    return final, confidence_bucket(final)


def fuse_token(word, camel_tok=None, stanza_tok=None, farasa_tok=None):
    fused = {
        "word": word,
        "final": {},
        "sources": {},
        "confidence": "medium",
        "notes": [],
        "conflicts": [],
    }

    fix = KNOWN_FIXES.get(word, {})

    if farasa_tok and farasa_tok.get("segmentation"):
        fused["final"]["segmentation"] = farasa_tok["segmentation"]
        fused["sources"]["segmentation"] = "farasa"
    else:
        fused["final"]["segmentation"] = [word]
        fused["sources"]["segmentation"] = "fallback"

    camel_analyses = camel_tok.get("analyses", []) if camel_tok else []

    if camel_analyses:
        fused["final"]["lemma"] = camel_analyses[0].get("lemma")
        fused["sources"]["lemma"] = "camel"
    elif stanza_tok:
        fused["final"]["lemma"] = stanza_tok.get("lemma")
        fused["sources"]["lemma"] = "stanza"

    if camel_analyses:
        fused["final"]["root"] = camel_analyses[0].get("root")
        fused["final"]["root_type"] = camel_analyses[0].get("root_type")
        fused["sources"]["root"] = "camel"

    if camel_analyses:
        fused["final"]["gloss"] = camel_analyses[0].get("gloss")
        fused["sources"]["gloss"] = "camel"

    camel_pos_raw = fix.get("pos") or (camel_analyses[0].get("pos") if camel_analyses else None)
    stanza_pos_raw = stanza_tok.get("upos") if stanza_tok else None

    final_pos, pos_source, _, pos_notes = score_pos(camel_pos_raw, stanza_pos_raw)
    fused["final"]["pos"] = final_pos
    fused["sources"]["pos"] = pos_source
    fused["notes"].extend(pos_notes)

    if pos_notes:
        fused["conflicts"].append(classify_conflict("pos", camel_pos_raw, stanza_pos_raw))

    if camel_analyses:
        fused["final"]["gender"] = camel_analyses[0].get("gender")
        fused["final"]["number"] = camel_analyses[0].get("number")
        fused["final"]["tense"] = camel_analyses[0].get("tense")
        fused["sources"]["morphology"] = "camel"

    if stanza_tok:
        fused["final"]["case"] = stanza_tok.get("case")
        fused["final"]["definite"] = stanza_tok.get("definite")
        fused["sources"]["case"] = "stanza"

    if stanza_tok and stanza_tok.get("dependency"):
        fused["final"]["dependency"] = stanza_tok["dependency"]
        fused["sources"]["dependency"] = "stanza"

    raw_conf = camel_analyses[0].get("confidence", 0.5) if camel_analyses else 0.5
    conf_score, conf_level = fuse_confidence(raw_conf, pos_source)
    fused["final"]["confidence_score"] = conf_score
    fused["final"]["confidence_level"] = conf_level
    fused["confidence"] = conf_level

    if fix:
        fused["notes"].append(f"applied known_fix for '{word}'")

    return fused


def fusion_system(text, camel_res, stanza_res, farasa_res):
    farasa_tokens = farasa_res.get("tokens", [])
    camel_tokens = camel_res.get("tokens", [])
    stanza_tokens = stanza_res.get("tokens", [])
    fused_output = []
    stanza_index = 0

    for i, farasa_tok in enumerate(farasa_tokens):
        word = farasa_tok["surface"]
        camel_tok = camel_tokens[i] if i < len(camel_tokens) else None

        collected = []
        while stanza_index < len(stanza_tokens):
            collected.append(stanza_tokens[stanza_index])
            stanza_index += 1
            if "".join([t["surface"] for t in collected]).replace(" ", "") == word:
                break

        merged_stanza = None
        if collected:
            main = collected[-1].copy()
            main["merged_tokens"] = [t["surface"] for t in collected]
            merged_stanza = main

        fused_output.append(fuse_token(word, camel_tok, merged_stanza, farasa_tok))

    return {"text": text, "fusion": fused_output}


# ============================================================
# Evaluation - Advanced Metrics (Precision / Recall / F1)
# ============================================================

def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def evaluate_tools(text, camel_res, stanza_res, farasa_res):
    """Evaluate using normalized tokens only (migration-safe).

    If inputs are not normalized yet, fall back to legacy behavior.
    """
    # Try normalized flow first
    try:
        from backend.services.normalizer import (
            normalize_tool_output,
        )

        camel_n = normalize_tool_output("camel", camel_res)
        stanza_n = normalize_tool_output("stanza", stanza_res)
        farasa_n = normalize_tool_output("farasa", farasa_res)

        words = [t.get("surface") for t in (farasa_n.get("tokens", []) or [])]
        total = len(words)
        camel_tokens = camel_n.get("tokens", [])
        stanza_tokens = stanza_n.get("tokens", [])
        farasa_tokens = farasa_n.get("tokens", [])

        pos_tp = pos_fp = pos_fn = 0
        lemma_match = 0
        seg_coverage = 0
        conflicts = []
        all_conflicts = []

        from backend.services.comparison_service import _lemma_equal, _pos_equal

        for i in range(total):
            c_tok = camel_tokens[i] if i < len(camel_tokens) else None
            s_tok = stanza_tokens[i] if i < len(stanza_tokens) else None
            f_tok = farasa_tokens[i] if i < len(farasa_tokens) else None

            if c_tok and s_tok:
                if _pos_equal(c_tok.get("pos"), s_tok.get("pos")):
                    pos_tp += 1
                else:
                    pos_fp += 1
                    pos_fn += 1
                    if words[i]:
                        conflicts.append({
                            "word": words[i],
                            "camel_pos": c_tok.get("pos"),
                            "stanza_pos": s_tok.get("pos"),
                        })

            if c_tok and s_tok:
                if _lemma_equal(c_tok.get("lemma"), s_tok.get("lemma")):
                    lemma_match += 1

            if f_tok and f_tok.get("segmentation"):
                seg_coverage += 1

        pos_agreement = pos_tp / total if total else 0
        pos_prf = compute_prf(pos_tp, pos_fp, pos_fn)

        return {
            "total_words": total,
            "pos_agreement": round(pos_agreement, 2),
            "pos_agreement_pct": f"{round(pos_agreement * 100, 1)}%",
            "pos_precision": pos_prf["precision"],
            "pos_recall": pos_prf["recall"],
            "pos_f1": pos_prf["f1"],
            "lemma_match": round(lemma_match / total, 2) if total else 0,
            "lemma_match_pct": f"{round(lemma_match / total * 100, 1)}%" if total else "0%",
            "segmentation_coverage": round(seg_coverage / total, 2) if total else 0,
            "pos_conflicts": conflicts,
            "all_conflicts": all_conflicts,
        }
    except Exception:
        # Legacy fallback (keeps backward compatibility)
        pass

    words = [t["surface"] for t in farasa_res.get("tokens", [])]
    camel_tokens = camel_res.get("tokens", [])
    stanza_tokens = stanza_res.get("tokens", [])
    farasa_tokens = farasa_res.get("tokens", [])
    total = len(words)


    pos_tp = pos_fp = pos_fn = 0
    lemma_match = 0
    seg_coverage = 0
    conflicts = []
    all_conflicts = []

    for i in range(total):
        camel_ana = camel_tokens[i]["analyses"][0] if i < len(camel_tokens) and camel_tokens[i].get("analyses") else None
        stanza_tok = stanza_tokens[i] if i < len(stanza_tokens) else None
        farasa_tok = farasa_tokens[i] if i < len(farasa_tokens) else None


        if camel_ana and stanza_tok:
            camel_pos = normalize_pos_for_compare(camel_ana.get("pos"))
            stanza_pos = stanza_tok.get("upos", "").upper()

            if camel_pos and stanza_pos:
                if camel_pos == stanza_pos:
                    pos_tp += 1
                else:
                    pos_fp += 1
                    pos_fn += 1
                    conflicts.append({
                        "word": words[i],
                        "camel_pos": camel_pos,
                        "stanza_pos": stanza_pos,
                    })
                    all_conflicts.append(classify_conflict("pos", camel_pos, stanza_pos))

            c_lemma = strip_diacritics(camel_ana.get("lemma"))
            s_lemma = strip_diacritics(stanza_tok.get("lemma"))
            if c_lemma and s_lemma:
                if c_lemma == s_lemma:
                    lemma_match += 1
                else:
                    all_conflicts.append(classify_conflict("lemma", c_lemma, s_lemma))

        if farasa_tok and farasa_tok.get("segmentation"):
            seg_coverage += 1

    pos_agreement = pos_tp / total if total else 0
    pos_prf = compute_prf(pos_tp, pos_fp, pos_fn)

    return {
        "total_words": total,
        "pos_agreement": round(pos_agreement, 2),
        "pos_agreement_pct": f"{round(pos_agreement * 100, 1)}%",
        "pos_precision": pos_prf["precision"],
        "pos_recall": pos_prf["recall"],
        "pos_f1": pos_prf["f1"],
        "lemma_match": round(lemma_match / total, 2) if total else 0,
        "lemma_match_pct": f"{round(lemma_match / total * 100, 1)}%" if total else "0%",
        "segmentation_coverage": round(seg_coverage / total, 2) if total else 0,
        "pos_conflicts": conflicts,
        "all_conflicts": all_conflicts,
    }


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    partner_status = get_all_partner_statuses() if PARTNER_TOOLS_LOADED else {
        "udpipe":   {"status": "not_loaded"},
        "alkhalil": {"status": "not_loaded"},
        "arabert":  {"status": "not_loaded"},
        "madamira": {"status": "not_loaded"},
    }
    return {
        "platform": "Arabic NLP Comparative Platform",
        "version":  "8.3",
        "tools": {
            "camel":     {"status": "ok" if camel_disambiguator else "failed"},
            "farasa":    {"status": "ok" if farasa_segmenter    else "failed"},
            "stanza":    {"status": "ok" if stanza_pipeline     else "failed"},
            "qalsadi":   {"status": "ok" if qalsadi_analyzer    else "failed"},
            "sinatools": {"status": "future_work",
                         "note": "Excluded — 880MB model, planned microservice"},
            **{k: {"status": v["status"]} for k, v in partner_status.items()},
        },
    }



@app.get("/analyze/camel")
def analyze_camel(text: str):
    """Return unified normalized schema for CAMeL."""
    if not text.strip():
        raise HTTPException(400, "Empty text")
    raw = cached_analyze(camel_analyze, text)
    try:
        from backend.services.normalizer import normalize_tool_output
        return normalize_tool_output("camel", raw)
    except Exception as e:
        return {"tool": "camel", "status": "error", "input": text, "word_count": 0, "tokens": [], "error": str(e)}



@app.get("/analyze/farasa")
def analyze_farasa(text: str):
    """Return unified normalized schema for Farasa."""
    if not text.strip():
        raise HTTPException(400, "Empty text")
    raw = cached_analyze(farasa_analyze, text)
    try:
        from backend.services.normalizer import normalize_tool_output
        return normalize_tool_output("farasa", raw)
    except Exception as e:
        return {"tool": "farasa", "status": "error", "input": text, "word_count": 0, "tokens": [], "error": str(e)}



@app.get("/analyze/stanza")
def analyze_stanza(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    return cached_analyze(stanza_analyze, text)


@app.get("/analyze/qalsadi")
def analyze_qalsadi(text: str):
    # Qalsadi uses analex rule-based checks. Output is structured per token.
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    return cached_analyze(qalsadi_analyze, text)



@app.get("/analyze/{tool}")
def analyze_by_tool(tool: str, text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    tool = tool.strip().lower()

    if   tool == "camel":   return cached_analyze(camel_analyze,   text)
    elif tool == "farasa":  return cached_analyze(farasa_analyze,  text)
    elif tool == "stanza":  return cached_analyze(stanza_analyze,  text)
    elif tool == "qalsadi": return cached_analyze(qalsadi_analyze, text)
    elif tool == "sinatools":
        return {"tool":"sinatools","status":"future_work",
                "message":"Planned microservice — excluded due to 880MB model","tokens":[]}
    elif PARTNER_TOOLS_LOADED and tool in PARTNER_TOOLS:
        return cached_analyze(PARTNER_TOOLS[tool].analyze, text)
    else:
        available = ["camel","farasa","stanza","qalsadi"] + list(PARTNER_TOOLS.keys())
        raise HTTPException(404, f"Tool '{tool}' not found. Available: {available}")

@app.get("/analyze-combined")
def analyze_combined(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)
    return {
        "input": text,
        "camel": camel_res,
        "farasa": farasa_res,
        "stanza": stanza_res,
        "qalsadi": qalsadi_res,
    }


@app.get("/compare")
def compare(text: str, tools: str = Query("camel,farasa,stanza,qalsadi")):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    tool_list = [t.strip().lower() for t in tools.split(",")]
    results = {}
    if "camel" in tool_list:
        results["camel"] = cached_analyze(camel_analyze, text)
    if "farasa" in tool_list:
        results["farasa"] = cached_analyze(farasa_analyze, text)
    if "stanza" in tool_list:
        results["stanza"] = cached_analyze(stanza_analyze, text)
    if "qalsadi" in tool_list:
        results["qalsadi"] = cached_analyze(qalsadi_analyze, text)
    return {"input": text, "tools": tool_list, "results": results}


@app.get("/fusion")
def fusion_endpoint(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)
    fused = fusion_system(text, camel_res, stanza_res, farasa_res)
    return {
        "input":        text,
        "qalsadi":      qalsadi_res,
        "fusion_result": fused,
    }



@app.get("/evaluate")
def evaluate(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    camel_res, farasa_res, stanza_res, _ = run_all_tools(text)
    return {
        "input": text,
        "evaluation": evaluate_tools(text, camel_res, stanza_res, farasa_res)
    }


GOLD_DATASET = [
    {"text": "كتب الطالب الدرس",
     "gold": [
         {"word":"كتب",    "pos":"VERB","lemma":"كتب"},
         {"word":"الطالب", "pos":"NOUN","lemma":"طالب"},
         {"word":"الدرس",  "pos":"NOUN","lemma":"درس"},
     ]},
    {"text": "ذهب محمد إلى المدرسة",
     "gold": [
         {"word":"ذهب",    "pos":"VERB","lemma":"ذهب"},
         {"word":"محمد",   "pos":"NOUN","lemma":"محمد"},
         {"word":"إلى",    "pos":"ADP", "lemma":"إلى"},
         {"word":"المدرسة","pos":"NOUN","lemma":"مدرسة"},
     ]},
    {"text": "قرأ الطلاب الكتب في المكتبة",
     "gold": [
         {"word":"قرأ",     "pos":"VERB","lemma":"قرأ"},
         {"word":"الطلاب",  "pos":"NOUN","lemma":"طالب"},
         {"word":"الكتب",   "pos":"NOUN","lemma":"كتاب"},
         {"word":"في",      "pos":"ADP", "lemma":"في"},
         {"word":"المكتبة", "pos":"NOUN","lemma":"مكتبة"},
     ]},
    {"text": "سيعمل العمال بكفاءة",
     "gold": [
         {"word":"سيعمل", "pos":"VERB","lemma":"عمل"},
         {"word":"العمال","pos":"NOUN","lemma":"عامل"},
         {"word":"بكفاءة","pos":"NOUN","lemma":"كفاءة"},
     ]},
    {"text": "العين جميلة",
     "gold": [
         {"word":"العين", "pos":"NOUN","lemma":"عين"},
         {"word":"جميلة", "pos":"ADJ", "lemma":"جميل"},
     ]},
    {"text": "لم يذهب محمد إلى المدرسة أمس",
     "gold": [
         {"word":"لم",      "pos":"PART","lemma":"لم"},
         {"word":"يذهب",    "pos":"VERB","lemma":"ذهب"},
         {"word":"محمد",    "pos":"NOUN","lemma":"محمد"},
         {"word":"إلى",     "pos":"ADP", "lemma":"إلى"},
         {"word":"المدرسة", "pos":"NOUN","lemma":"مدرسة"},
         {"word":"أمس",     "pos":"ADV", "lemma":"أمس"},
     ]},
    {"text": "باع التاجر بضاعته بسعر مرتفع",
     "gold": [
         {"word":"باع",    "pos":"VERB","lemma":"باع"},
         {"word":"التاجر", "pos":"NOUN","lemma":"تاجر"},
         {"word":"بضاعته","pos":"NOUN","lemma":"بضاعة"},
         {"word":"بسعر",   "pos":"NOUN","lemma":"سعر"},
         {"word":"مرتفع",  "pos":"ADJ", "lemma":"مرتفع"},
     ]},
    {"text": "إن البنات يأكلن المثلجات",
     "gold": [
         {"word":"إن",       "pos":"PART","lemma":"إن"},
         {"word":"البنات",   "pos":"NOUN","lemma":"بنت"},
         {"word":"يأكلن",    "pos":"VERB","lemma":"أكل"},
         {"word":"المثلجات", "pos":"NOUN","lemma":"مثلجة"},
     ]},
    {"text": "الكتاب الذي قرأته مفيد جداً",
     "gold": [
         {"word":"الكتاب","pos":"NOUN","lemma":"كتاب"},
         {"word":"الذي",  "pos":"PRON","lemma":"الذي"},
         {"word":"قرأته", "pos":"VERB","lemma":"قرأ"},
         {"word":"مفيد",  "pos":"ADJ", "lemma":"مفيد"},
         {"word":"جداً",  "pos":"ADV", "lemma":"جداً"},
     ]},
    {"text": "وجدت المعلمة طالبة مجتهدة في الفصل",
     "gold": [
         {"word":"وجدت",    "pos":"VERB","lemma":"وجد"},
         {"word":"المعلمة", "pos":"NOUN","lemma":"معلمة"},
         {"word":"طالبة",   "pos":"NOUN","lemma":"طالب"},
         {"word":"مجتهدة",  "pos":"ADJ", "lemma":"مجتهد"},
         {"word":"في",      "pos":"ADP", "lemma":"في"},
         {"word":"الفصل",   "pos":"NOUN","lemma":"فصل"},
     ]},
]

@app.get("/evaluate/dataset")
def evaluate_dataset():
    """Evaluate tools against gold dataset — 10 sentences"""
    results = []
    for item in GOLD_DATASET:
        camel_res, farasa_res, stanza_res, _ = run_all_tools(item["text"])
        eval_result = evaluate_tools(item["text"], camel_res, stanza_res, farasa_res)
        results.append({
            "text":              item["text"],
            "pos_agreement_pct": eval_result["pos_agreement_pct"],
            "lemma_match_pct":   eval_result["lemma_match_pct"],
            "pos_f1":            eval_result["pos_f1"],
            "total_words":       eval_result["total_words"],
        })
    avg_f1 = sum(r["pos_f1"] for r in results) / len(results)
    return {
        "total_sentences": len(results),
        "average_f1":      round(avg_f1, 3),
        "average_f1_pct":  f"{round(avg_f1 * 100, 1)}%",
        "results":         results,
    }

@app.get("/export")
def export_results(text: str, format: str = Query("json", description="json or csv")):
    """Export full analysis as downloadable JSON or CSV."""
    if not text.strip():
        raise HTTPException(400, "Empty text")

    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)
    fused = fusion_system(text, camel_res, stanza_res, farasa_res)
    evaln = evaluate_tools(text, camel_res, stanza_res, farasa_res)

    if format == "json":
        payload = {
            "input": text,
            "combined": {
                "camel": camel_res,
                "farasa": farasa_res,
                "stanza": stanza_res,
                "qalsadi": qalsadi_res,
            },
            "fusion": fused,
            "evaluation": evaln,
        }
        return StreamingResponse(
            io.StringIO(json.dumps(payload, ensure_ascii=False, indent=2)),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=analysis.json"}
        )

    elif format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "word", "lemma", "root", "root_type", "pos", "pos_source",
            "gender", "number", "tense", "gloss", "case", "definite",
            "confidence_score", "confidence_level", "notes"
        ])
        for tok in fused.get("fusion", []):
            f = tok.get("final", {})
            writer.writerow([
                tok.get("word", ""),
                f.get("lemma", ""), f.get("root", ""), f.get("root_type", ""),
                f.get("pos", ""), tok.get("sources", {}).get("pos", ""),
                f.get("gender", ""), f.get("number", ""), f.get("tense", ""),
                f.get("gloss", ""), f.get("case", ""), f.get("definite", ""),
                f.get("confidence_score", ""), f.get("confidence_level", ""),
                "; ".join(tok.get("notes", []))
            ])
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=analysis.csv"}
        )

    else:
        raise HTTPException(400, "format must be 'json' or 'csv'")


@app.post("/cache/clear")
def cache_clear():
    clear_cache()
    return {"status": "cache cleared"}


# ============================================================
# UI-ready Endpoints (optimized payloads)
# ============================================================


def _tool_names_from_query(tools: str) -> List[str]:
    return [t.strip().lower() for t in tools.split(",") if t.strip()]


def _normalize_for_ui(tool_name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    # Use backend normalizer when possible; fallback to empty on failure.
    try:
        from backend.services.normalizer import normalize_tool_output

        return normalize_tool_output(tool_name, raw)
    except Exception:
        return {"tool": tool_name, "status": "error", "input": None, "word_count": 0, "tokens": []}


def _ui_token_from_aligned(*, base_token: Dict[str, Any], tool_token: Optional[Dict[str, Any]], tool_name: str) -> Dict[str, Any]:
    if tool_name == "farasa":
        seg = tool_token.get("segmentation") if tool_token else None
        return {"segments": seg if isinstance(seg, list) and seg else [base_token.get("surface")]}

    lemma = tool_token.get("lemma") if tool_token else None
    root = tool_token.get("root") if tool_token else None
    pos = tool_token.get("pos") if tool_token else None

    from backend.services.ui_contracts import placeholder, safe_pos, pos_badge

    pos_std = safe_pos(pos)

    return {
        "lemma": placeholder(lemma),
        "root": placeholder(root),
        "pos": pos_std,
        "badge": pos_badge(pos_std),
    }


def _agreement_for_row(*, aligned_row: Any) -> Dict[str, Any]:
    # aligned_row is backend.services.alignment_engine.AlignedToken-like
    tools = aligned_row.tools

    from backend.services.ui_contracts import compute_status_color

    camel = tools.get("camel")
    stanza = tools.get("stanza")
    qalsadi = tools.get("qalsadi")
    # reference: prefer camel, else stanza, else qalsadi
    ref = camel or stanza or qalsadi

    def eq(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        import re

        def stripd(x: Any) -> str:
            return re.sub(r"[\u064B-\u065F\u0670]", "", str(x)).strip()

        sa = stripd(a)
        sb = stripd(b)
        return sa != "" and sa == sb

    pos_ok = True
    lemma_ok = True
    root_ok = True

    # POS across non-empty tools
    pos_vals = [t.get("pos") for t in [camel, stanza, qalsadi] if t and t.get("pos")]
    if len(pos_vals) >= 2:
        pos_ok = all(v == pos_vals[0] for v in pos_vals)

    lemma_vals = [t.get("lemma") for t in [camel, stanza, qalsadi] if t and t.get("lemma")]
    if len(lemma_vals) >= 2:
        lemma_ok = all(eq(v, lemma_vals[0]) for v in lemma_vals)

    root_vals = [t.get("root") for t in [camel, stanza, qalsadi] if t and t.get("root")]
    if len(root_vals) >= 2:
        root_ok = all(eq(v, root_vals[0]) for v in root_vals)

    status, color = ("none", "red")
    if pos_ok and lemma_ok and root_ok:
        status, color = "full", "green"
    elif pos_ok or lemma_ok or root_ok:
        status, color = "partial", "yellow"

    return {
        "pos": pos_ok,
        "lemma": lemma_ok,
        "root": root_ok,
        "status": status,
        "status_color": color,
        "agreement_state": status,
    }


@app.get("/ui/analyze/{tool}")
def ui_analyze(tool: str, text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    tool_l = tool.strip().lower()
    if tool_l not in {"camel", "farasa", "stanza", "qalsadi"}:
        raise HTTPException(404, "Tool not supported for ui/analyze")

    # raw analyze
    raw = cached_analyze(
        {"camel": camel_analyze, "farasa": farasa_analyze, "stanza": stanza_analyze, "qalsadi": qalsadi_analyze}[tool_l],
        text,
    )

    normalized = _normalize_for_ui(tool_l, raw)
    tokens = normalized.get("tokens", []) or []

    # UI rows: flat list (no nested analyses)
    from backend.services.ui_contracts import placeholder, safe_pos

    rows = []
    for t in tokens:
        if tool_l == "farasa":
            seg = t.get("segmentation")
            segs = seg if isinstance(seg, list) and seg else [t.get("surface")]
            rows.append({"word": t.get("surface"), "segments": segs})
        else:
            rows.append(
                {
                    "word": t.get("surface"),
                    "lemma": placeholder(t.get("lemma")),
                    "root": placeholder(t.get("root")),
                    "pos": safe_pos(t.get("pos")),
                }
            )

    return {"tool": tool_l, "rows": rows}


@app.get("/ui/compare")
def ui_compare(
    text: str,
    tools: str = Query("camel,stanza,qalsadi,farasa"),
):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    tool_list = _tool_names_from_query(tools)

    # Run selected tools (raw), then normalize
    raw_results: Dict[str, Dict[str, Any]] = {}

    if "camel" in tool_list:
        raw_results["camel"] = cached_analyze(camel_analyze, text)
    if "farasa" in tool_list:
        raw_results["farasa"] = cached_analyze(farasa_analyze, text)
    if "stanza" in tool_list:
        raw_results["stanza"] = cached_analyze(stanza_analyze, text)
    if "qalsadi" in tool_list:
        raw_results["qalsadi"] = cached_analyze(qalsadi_analyze, text)

    norm = {t: _normalize_for_ui(t, raw_results[t]) for t in raw_results.keys()}

    # Canonical base: prefer farasa tokens if present, else camel/stanza/qalsadi first available
    base_tool = "farasa" if "farasa" in norm else ("camel" if "camel" in norm else "stanza" if "stanza" in norm else "qalsadi")
    base_tokens = norm.get(base_tool, {}).get("tokens", []) or []

    # Tools tokens for alignment (only those present)
    tools_tokens = {t: norm[t].get("tokens", []) or [] for t in norm.keys()}

    from backend.services.alignment_engine import align_tools, compute_agreements

    aligned, _meta = align_tools(base_tokens=base_tokens, tools_tokens=tools_tokens)
    agreements = compute_agreements(aligned_tokens=aligned)

    # Build rows
    rows = []
    for at in aligned:
        word = at.base.get("surface")

        row = {
            "word": word,
            "camel": None,
            "stanza": None,
            "qalsadi": None,
            "farasa": None,
            "agreement": None,
        }

        if "camel" in norm:
            row["camel"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("camel"), tool_name="camel")
        if "stanza" in norm:
            row["stanza"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("stanza"), tool_name="stanza")
        if "qalsadi" in norm:
            row["qalsadi"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("qalsadi"), tool_name="qalsadi")
        if "farasa" in norm:
            row["farasa"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("farasa"), tool_name="farasa")

        row["agreement"] = _agreement_for_row(aligned_row=at)
        rows.append(row)

    return {
        "summary": {
            "pos_agreement": agreements.get("pos_agreement", 0),
            "lemma_agreement": agreements.get("lemma_agreement", 0),
            "root_agreement": agreements.get("root_agreement", 0),
            "token_count": agreements.get("token_count", len(rows)),
        },
        "rows": rows,
    }


@app.get("/ui/fusion")
def ui_fusion(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    # Use raw tools, normalize, align by surface
    camel_raw = cached_analyze(camel_analyze, text)
    stanza_raw = cached_analyze(stanza_analyze, text)
    qalsadi_raw = cached_analyze(qalsadi_analyze, text)
    farasa_raw = cached_analyze(farasa_analyze, text)

    from backend.services.normalizer import normalize_tool_output

    camel_n = _normalize_for_ui("camel", camel_raw)
    stanza_n = _normalize_for_ui("stanza", stanza_raw)
    qalsadi_n = _normalize_for_ui("qalsadi", qalsadi_raw)
    farasa_n = _normalize_for_ui("farasa", farasa_raw)

    base_tokens = farasa_n.get("tokens", []) or []
    tools_tokens = {
        "camel": camel_n.get("tokens", []) or [],
        "stanza": stanza_n.get("tokens", []) or [],
        "qalsadi": qalsadi_n.get("tokens", []) or [],
        "farasa": farasa_n.get("tokens", []) or [],
    }

    from backend.services.alignment_engine import align_tools

    aligned, _ = align_tools(base_tokens=base_tokens, tools_tokens=tools_tokens)

    # UI fusion: pick from camel when present else stanza else qalsadi.
    rows = []
    for at in aligned:
        camel = at.tools.get("camel")
        stanza = at.tools.get("stanza")
        qalsadi = at.tools.get("qalsadi")

        # Simple agreement-based choice for pos/lemma/root
        agreement = _agreement_for_row(aligned_row=at)

        def pick(feature: str):
            for t, src in [(camel, "camel"), (stanza, "stanza"), (qalsadi, "qalsadi")]:
                if t and t.get(feature):
                    return t.get(feature), src
            return None, "-"

        lemma, lemma_src = pick("lemma")
        root, root_src = pick("root")
        pos, pos_src = pick("pos")

        from backend.services.ui_contracts import placeholder, safe_pos

        # Confidence: average of available normalized confidences (camel & stanza)
        confs = []
        for t in [camel, stanza, qalsadi]:
            if t and isinstance(t.get("confidence"), dict):
                confs.append(float(t["confidence"].get("score") or 0.0))
        confidence = round(sum(confs) / len(confs), 2) if confs else 0.0

        rows.append(
            {
                "word": at.base.get("surface"),
                "lemma": placeholder(lemma),
                "root": placeholder(root),
                "pos": safe_pos(pos),
                "gloss": placeholder(camel.get("gloss") if camel else None, default="-"),
                "confidence": confidence,
                "source": {
                    "lemma": lemma_src if lemma_src else "camel",
                    "root": root_src if root_src else "camel",
                    "pos": "agreement" if agreement.get("status") in ("full", "partial") else (pos_src if pos_src else "camel"),
                },
            }
        )

    return {"rows": rows}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
