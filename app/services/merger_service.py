from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Dict, List, Optional, Tuple

from camel_tools.tokenizers.word import simple_word_tokenize

from app.services.cache_service import cached_analyze
from app.utils.constants import FUSION_WEIGHTS, GOLD_DATASET, KNOWN_FIXES
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
)

from app.tools.camel_tool import camel_analyze
from app.tools.farasa_tool import farasa_analyze
from app.tools.stanza_tool import stanza_analyze
from app.tools.qalsadi_tool import qalsadi_analyze


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


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def score_pos(camel_pos_raw, stanza_pos_raw) -> tuple:
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
    """Fusion consuming normalized token schema.

    Token alignment is done by surface string (primary key), not by index.
    """
    from backend.services.normalizer import normalize_tool_output

    camel_n = normalize_tool_output("camel", camel_res)
    stanza_n = normalize_tool_output("stanza", stanza_res)
    farasa_n = normalize_tool_output("farasa", farasa_res)

    camel_map = {t.get("surface"): t for t in (camel_n.get("tokens") or []) if isinstance(t, dict)}
    stanza_map = {t.get("surface"): t for t in (stanza_n.get("tokens") or []) if isinstance(t, dict)}
    farasa_tokens = [t for t in (farasa_n.get("tokens") or []) if isinstance(t, dict)]

    fused_output = []
    for f_tok in farasa_tokens:
        word = f_tok.get("surface")
        c_tok = camel_map.get(word)
        s_tok = stanza_map.get(word)

        # Convert normalized tokens into the legacy inputs expected by fuse_token()
        # so frontend/response shape remains unchanged.
        camel_legacy = (
            {"surface": word, "analyses": [{"lemma": c_tok.get("lemma"), "root": c_tok.get("root"), "root_type": c_tok.get("meta", {}).get("root_type"), "pos": c_tok.get("pos"), "gender": c_tok.get("features", {}).get("gender"), "number": c_tok.get("features", {}).get("number"), "tense": c_tok.get("features", {}).get("tense"), "gloss": c_tok.get("gloss"), "confidence": c_tok.get("confidence", {}).get("score", 0.5)}]}
            if c_tok
            else None
        )
        stanza_legacy = (
            {
                "surface": word,
                "lemma": s_tok.get("lemma"),
                "upos": s_tok.get("pos"),
                "case": s_tok.get("features", {}).get("case"),
                "definite": s_tok.get("features", {}).get("definite"),
                "dependency": s_tok.get("dependency"),
            }
            if s_tok
            else None
        )

        farasa_legacy = {
            "surface": word,
            "segmentation": f_tok.get("segmentation"),
        }

        fused_output.append(fuse_token(word, camel_legacy, stanza_legacy, farasa_legacy))

    return {"text": text, "fusion": fused_output}



def evaluate_tools(text, camel_res, stanza_res, farasa_res):
    """Evaluation using normalized tokens only.

    Token matching is by surface string; no index alignment.
    """
    from backend.services.normalizer import normalize_tool_output
    from backend.services.comparison_service import _lemma_equal, _pos_equal

    camel_n = normalize_tool_output("camel", camel_res)
    stanza_n = normalize_tool_output("stanza", stanza_res)
    farasa_n = normalize_tool_output("farasa", farasa_res)

    words = [t.get("surface") for t in (farasa_n.get("tokens") or []) if isinstance(t, dict)]
    total = len(words)

    camel_map = {t.get("surface"): t for t in (camel_n.get("tokens") or []) if isinstance(t, dict)}
    stanza_map = {t.get("surface"): t for t in (stanza_n.get("tokens") or []) if isinstance(t, dict)}
    farasa_map = {t.get("surface"): t for t in (farasa_n.get("tokens") or []) if isinstance(t, dict)}

    pos_tp = pos_fp = pos_fn = 0
    lemma_match = 0
    seg_coverage = 0
    conflicts = []
    all_conflicts = []

    for w in words:
        c_tok = camel_map.get(w)
        s_tok = stanza_map.get(w)
        f_tok = farasa_map.get(w)

        if f_tok and f_tok.get("segmentation"):
            seg_coverage += 1

        if c_tok and s_tok:
            if _pos_equal(c_tok.get("pos"), s_tok.get("pos")):
                pos_tp += 1
            else:
                pos_fp += 1
                pos_fn += 1
                conflicts.append({"word": w, "camel_pos": c_tok.get("pos"), "stanza_pos": s_tok.get("pos")})
                all_conflicts.append(classify_conflict("pos", c_tok.get("pos"), s_tok.get("pos")))

            if _lemma_equal(c_tok.get("lemma"), s_tok.get("lemma")):
                lemma_match += 1
            else:
                # keep legacy conflict format helper
                if c_tok.get("lemma") or s_tok.get("lemma"):
                    all_conflicts.append(classify_conflict("lemma", c_tok.get("lemma"), s_tok.get("lemma")))

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


