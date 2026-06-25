from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.utils.constants import FUSION_WEIGHTS, KNOWN_FIXES
from app.utils.helpers import (
    confidence_bucket,
    normalize_lemma_for_compare,
    normalize_pos_for_compare,
)
from backend.services.comparison_service import build_conflicts


_TOOL_RELIABILITY = {
    "camel": 0.35,
    "stanza": 0.35,
    "udpipe": 0.15,
    "qalsadi": 0.10,
    "alkhalil": 0.05,
}


def _alignment_match_type(tok: Optional[Dict[str, Any]]) -> Optional[str]:
    if not tok:
        return None
    alignment = tok.get("alignment")
    if isinstance(alignment, dict):
        match_type = alignment.get("match_type")
        if match_type:
            return str(match_type)
    return None


def _camel_root_type(camel_tok: Optional[Dict[str, Any]]) -> Optional[str]:
    if not camel_tok:
        return None
    meta = camel_tok.get("meta")
    if isinstance(meta, dict) and meta.get("root_type"):
        return meta.get("root_type")
    analyses = camel_tok.get("analyses") or []
    if analyses and isinstance(analyses[0], dict):
        root_type = analyses[0].get("root_type")
        if root_type:
            return root_type
    return camel_tok.get("root_type")


def score_pos(camel_pos_raw, stanza_pos_raw, *, partial_match: bool = False) -> tuple:
    """Returns (final_pos, source, score, notes)."""
    camel_pos = normalize_pos_for_compare(camel_pos_raw)
    stanza_pos = normalize_pos_for_compare(stanza_pos_raw)
    notes: List[str] = []

    if camel_pos == stanza_pos and camel_pos:
        return camel_pos, "partial_match" if partial_match else "agreement", 4, notes

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
        return camel_pos, "partial_match" if partial_match else "camel_only", 1, notes
    if stanza_pos:
        return stanza_pos, "partial_match" if partial_match else "stanza_only", 1, notes
    return None, "none", 0, notes



def _token_value(tok: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    if not tok:
        return None
    value = tok.get(key)
    if value is not None and str(value).strip():
        return str(value).strip()
    analyses = tok.get("analyses") or []
    if analyses and isinstance(analyses[0], dict):
        nested = analyses[0].get(key)
        if nested is not None and str(nested).strip():
            return str(nested).strip()
    return None


def _support_score(
    *,
    chosen_value: Optional[str],
    values_by_tool: Dict[str, Optional[str]],
    normalizer,
) -> float:
    if not chosen_value:
        return 0.0

    normalized_chosen = normalizer(chosen_value) if normalizer else str(chosen_value).strip()
    if not normalized_chosen or normalized_chosen == "X":
        return 0.0

    valid_weight = 0.0
    support_weight = 0.0

    for tool_name, raw_value in values_by_tool.items():
        if raw_value is None:
            continue
        normalized_value = normalizer(raw_value) if normalizer else str(raw_value).strip()
        if not normalized_value or normalized_value == "X":
            continue

        weight = float(_TOOL_RELIABILITY.get(tool_name, 0.0))
        if weight <= 0:
            continue

        valid_weight += weight
        if normalized_value == normalized_chosen:
            support_weight += weight

    if valid_weight <= 0:
        return 0.0

    total_possible_weight = sum(_TOOL_RELIABILITY.values())
    agreement_strength = support_weight / valid_weight
    reliability = valid_weight / total_possible_weight
    conflict_penalty = max(0.0, (valid_weight - support_weight) / total_possible_weight) * 0.35
    missing_penalty = max(0.0, total_possible_weight - valid_weight) / total_possible_weight * 0.25

    score = 0.1 + (0.55 * agreement_strength) + (0.35 * reliability) - conflict_penalty - missing_penalty
    return round(max(0.0, min(1.0, score)), 3)


def compute_token_confidence(fused: Dict[str, Any], *, camel_tok=None, stanza_tok=None, qalsadi_tok=None, alkhalil_tok=None, udpipe_tok=None) -> tuple[float, str]:
    pos_score = _support_score(
        chosen_value=fused.get("final", {}).get("pos"),
        values_by_tool={
            "camel": _token_value(camel_tok, "pos") or _token_value(camel_tok, "upos"),
            "stanza": _token_value(stanza_tok, "pos") or _token_value(stanza_tok, "upos"),
            "qalsadi": _token_value(qalsadi_tok, "pos") or _token_value(qalsadi_tok, "upos"),
            "alkhalil": _token_value(alkhalil_tok, "pos") or _token_value(alkhalil_tok, "upos"),
            "udpipe": _token_value(udpipe_tok, "pos") or _token_value(udpipe_tok, "upos"),
        },
        normalizer=normalize_pos_for_compare,
    )

    lemma_score = _support_score(
        chosen_value=fused.get("final", {}).get("lemma"),
        values_by_tool={
            "camel": _token_value(camel_tok, "lemma"),
            "stanza": _token_value(stanza_tok, "lemma"),
            "qalsadi": _token_value(qalsadi_tok, "lemma"),
            "alkhalil": _token_value(alkhalil_tok, "lemma"),
            "udpipe": _token_value(udpipe_tok, "lemma"),
        },
        normalizer=normalize_lemma_for_compare,
    )

    scores = [score for score in (pos_score, lemma_score) if score > 0]
    if not scores:
        return 0.0, confidence_bucket(0.0)

    final = round(sum(scores) / len(scores), 3)
    return final, confidence_bucket(final)


def fuse_token(word, camel_tok=None, stanza_tok=None, farasa_tok=None, qalsadi_tok=None, alkhalil_tok=None, udpipe_tok=None):
    fused: Dict[str, Any] = {
        "word": word,
        "final": {},
        "sources": {},
        "evidence": {},
        "confidence": "medium",
        "notes": [],
        "conflicts": [],
    }

    fix = KNOWN_FIXES.get(word, {})

    for name, tok in (
        ("camel", camel_tok),
        ("stanza", stanza_tok),
        ("farasa", farasa_tok),
        ("qalsadi", qalsadi_tok),
        ("alkhalil", alkhalil_tok),
        ("udpipe", udpipe_tok),
    ):
        if tok:
            fused["evidence"][name] = {
                "surface": tok.get("surface"),
                "lemma": tok.get("lemma"),
                "root": tok.get("root"),
                "pos": tok.get("pos") or tok.get("upos"),
                "segmentation": tok.get("segmentation"),
                "dependency": tok.get("dependency"),
            }

    if farasa_tok and farasa_tok.get("segmentation"):
        fused["final"]["segmentation"] = farasa_tok["segmentation"]
        fused["sources"]["segmentation"] = "farasa"
    else:
        fused["final"]["segmentation"] = [word]
        fused["sources"]["segmentation"] = "fallback"

    camel_analyses = camel_tok.get("analyses", []) if camel_tok else []
    qalsadi_analyses = qalsadi_tok.get("analyses", []) if qalsadi_tok else []
    alkhalil_analyses = alkhalil_tok.get("analyses", []) if alkhalil_tok else []
    udpipe_analyses = udpipe_tok.get("analyses", []) if udpipe_tok else []

    camel_lemma = camel_analyses[0].get("lemma") if camel_analyses else None
    stanza_lemma = stanza_tok.get("lemma") if stanza_tok else None
    qalsadi_lemma = qalsadi_analyses[0].get("lemma") if qalsadi_analyses else (qalsadi_tok.get("lemma") if qalsadi_tok else None)
    alkhalil_lemma = alkhalil_analyses[0].get("lemma") if alkhalil_analyses else None
    udpipe_lemma = udpipe_analyses[0].get("lemma") if udpipe_analyses else None

    if camel_lemma:
        fused["final"]["lemma"] = camel_lemma
        fused["sources"]["lemma"] = "camel"
    elif stanza_lemma:
        fused["final"]["lemma"] = stanza_lemma
        fused["sources"]["lemma"] = "stanza"
    elif qalsadi_lemma:
        fused["final"]["lemma"] = qalsadi_lemma
        fused["sources"]["lemma"] = "qalsadi"
    elif alkhalil_lemma:
        fused["final"]["lemma"] = alkhalil_lemma
        fused["sources"]["lemma"] = "alkhalil"
    elif udpipe_lemma:
        fused["final"]["lemma"] = udpipe_lemma
        fused["sources"]["lemma"] = "udpipe"

    if camel_analyses:
        fused["final"]["root"] = camel_analyses[0].get("root")
        fused["final"]["root_type"] = _camel_root_type(camel_tok)
        fused["sources"]["root"] = "camel"

    if camel_analyses:
        fused["final"]["gloss"] = camel_analyses[0].get("gloss")
        fused["sources"]["gloss"] = "camel"

    camel_pos_raw = fix.get("pos") or (camel_analyses[0].get("pos") if camel_analyses else None) or (qalsadi_analyses[0].get("pos") if qalsadi_analyses else None) or (alkhalil_analyses[0].get("pos") if alkhalil_analyses else None) or (udpipe_analyses[0].get("pos") if udpipe_analyses else None)
    stanza_pos_raw = stanza_tok.get("upos") if stanza_tok else None
    if not stanza_pos_raw and qalsadi_tok:
        stanza_pos_raw = qalsadi_tok.get("pos") or qalsadi_tok.get("upos")
    if not stanza_pos_raw and alkhalil_tok:
        stanza_pos_raw = alkhalil_tok.get("pos") or alkhalil_tok.get("upos")
    if not stanza_pos_raw and udpipe_tok:
        stanza_pos_raw = udpipe_tok.get("pos") or udpipe_tok.get("upos")

    partial_match = _alignment_match_type(stanza_tok) == "partial_match" or _alignment_match_type(camel_tok) == "partial_match"
    final_pos, pos_source, _, pos_notes = score_pos(camel_pos_raw, stanza_pos_raw, partial_match=partial_match)
    fused["final"]["pos"] = final_pos

    # Provenance attribution for POS must reflect the actual provider of the selected value.
    # "agreement" is a fusion label, not a real tool.
    if pos_source == "agreement":
        fused["sources"]["pos"] = "camel" if camel_pos_raw else "stanza"
    else:
        fused["sources"]["pos"] = pos_source

    fused["notes"].extend(pos_notes)


    fused["conflicts"].extend(
        build_conflicts(
            camel_tok=camel_tok,
            stanza_tok=stanza_tok,
            qalsadi_tok=qalsadi_tok,
            alkhalil_tok=alkhalil_tok,
            udpipe_tok=udpipe_tok,
        )
    )

    if camel_analyses:
        fused["final"]["gender"] = camel_analyses[0].get("gender")
        fused["final"]["number"] = camel_analyses[0].get("number")
        fused["final"]["tense"] = camel_analyses[0].get("tense")
        fused["sources"]["morphology"] = "camel"

    stanza_case = stanza_tok.get("case") if stanza_tok else None
    udpipe_case = udpipe_tok.get("case") if udpipe_tok else None
    if stanza_case:
        fused["final"]["case"] = stanza_case
        fused["final"]["definite"] = stanza_tok.get("definite")
        fused["sources"]["case"] = "stanza"
    elif udpipe_case:
        fused["final"]["case"] = udpipe_case
        fused["sources"]["case"] = "udpipe"

    if stanza_tok and stanza_tok.get("dependency"):
        fused["final"]["dependency"] = stanza_tok["dependency"]
        fused["sources"]["dependency"] = "stanza"
    elif qalsadi_tok and qalsadi_tok.get("dependency"):
        fused["final"]["dependency"] = qalsadi_tok["dependency"]
        fused["sources"]["dependency"] = "qalsadi"
    elif alkhalil_tok and alkhalil_tok.get("dependency"):
        fused["final"]["dependency"] = alkhalil_tok["dependency"]
        fused["sources"]["dependency"] = "alkhalil"
    elif udpipe_tok and udpipe_tok.get("dependency"):
        fused["final"]["dependency"] = udpipe_tok["dependency"]
        fused["sources"]["dependency"] = "udpipe"

    conf_score, conf_level = compute_token_confidence(
        fused,
        camel_tok=camel_tok,
        stanza_tok=stanza_tok,
        qalsadi_tok=qalsadi_tok,
        alkhalil_tok=alkhalil_tok,
        udpipe_tok=udpipe_tok,
    )
    udpipe_pos_raw = udpipe_tok.get("pos") or udpipe_tok.get("upos") if udpipe_tok else None
    if final_pos and udpipe_pos_raw and normalize_pos_for_compare(udpipe_pos_raw) == final_pos:
        fused["notes"].append("UDPipe confirms POS")
    if stanza_case and udpipe_case and str(stanza_case).strip().lower() == str(udpipe_case).strip().lower():
        fused["notes"].append("UDPipe confirms case")
    fused["final"]["confidence_score"] = conf_score
    fused["final"]["confidence_level"] = conf_level
    fused["confidence"] = conf_level

    if fix:
        fused["notes"].append(f"applied known_fix for '{word}'")

    return fused


def fusion_system(text, camel_res, stanza_res, farasa_res, qalsadi_res=None, all_tool_results=None):
    from backend.services.alignment_engine import align_tools
    from backend.services.normalizer import normalize_tool_output

    source_results = {
        "camel": camel_res,
        "stanza": stanza_res,
        "farasa": farasa_res,
        "qalsadi": qalsadi_res,
    }
    if isinstance(all_tool_results, dict):
        source_results.update(all_tool_results)

    normalized = {name: normalize_tool_output(name, payload or {}) for name, payload in source_results.items()}
    farasa_tokens = normalized.get("farasa", {}).get("tokens", []) or []
    tool_tokens = {
        "camel": normalized.get("camel", {}).get("tokens", []) or [],
        "stanza": normalized.get("stanza", {}).get("tokens", []) or [],
        "qalsadi": normalized.get("qalsadi", {}).get("tokens", []) or [],
        "alkhalil": normalized.get("alkhalil", {}).get("tokens", []) or [],
        "udpipe": normalized.get("udpipe", {}).get("tokens", []) or [],
    }

    aligned_tokens, _meta = align_tools(
        base_tokens=farasa_tokens,
        tools_tokens=tool_tokens,
    )

    fused_output = []

    for atok in aligned_tokens:
        word = atok.base["surface"]
        camel_tok = atok.tools.get("camel")
        stanza_tok = atok.tools.get("stanza")
        qalsadi_tok = atok.tools.get("qalsadi")
        alkhalil_tok = atok.tools.get("alkhalil")
        udpipe_tok = atok.tools.get("udpipe")
        farasa_tok = atok.base

        fused_output.append(fuse_token(word, camel_tok, stanza_tok, farasa_tok, qalsadi_tok, alkhalil_tok, udpipe_tok))

    return {"text": text, "fusion": fused_output}


