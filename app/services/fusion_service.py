from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.utils.constants import FUSION_WEIGHTS, KNOWN_FIXES
from app.utils.helpers import (
    classify_conflict,
    confidence_bucket,
    normalize_pos_for_compare,
)


def score_pos(camel_pos_raw, stanza_pos_raw) -> tuple:
    """Returns (final_pos, source, score, notes)."""
    camel_pos = normalize_pos_for_compare(camel_pos_raw)
    stanza_pos = stanza_pos_raw.upper() if stanza_pos_raw else None
    notes: List[str] = []

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
    fused: Dict[str, Any] = {
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

    # Provenance attribution for POS must reflect the actual provider of the selected value.
    # "agreement" is a fusion label, not a real tool.
    if pos_source == "agreement":
        fused["sources"]["pos"] = "camel" if camel_pos_raw else "stanza"
    else:
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
    from backend.services.alignment_engine import align_tools

    farasa_tokens = farasa_res.get("tokens", [])
    camel_tokens = camel_res.get("tokens", [])
    stanza_tokens = stanza_res.get("tokens", [])

    aligned_tokens, _meta = align_tools(
        base_tokens=farasa_tokens,
        tools_tokens={"camel": camel_tokens, "stanza": stanza_tokens},
    )

    fused_output = []

    for atok in aligned_tokens:
        word = atok.base["surface"]
        camel_tok = atok.tools.get("camel")
        stanza_tok = atok.tools.get("stanza")
        farasa_tok = atok.base

        fused_output.append(fuse_token(word, camel_tok, stanza_tok, farasa_tok))

    return {"text": text, "fusion": fused_output}


