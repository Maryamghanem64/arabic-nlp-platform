from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.utils.constants import FUSION_WEIGHTS, KNOWN_FIXES
from app.utils.helpers import normalize_lemma_for_compare, normalize_pos_for_compare
from backend.services.comparison_service import build_conflicts
from backend.services.normalizer import extract_alkhalil_canonical_pos, normalize_alkhalil_pos
from app.services.expert_fusion_service import apply_expert_fusion


_TOOL_RELIABILITY = {
    "camel": 0.35,
    "stanza": 0.35,
    "udpipe": 0.15,
    "qalsadi": 0.10,
    "alkhalil": 0.05,
    "sinatools": 0.20,
}

TOOL_WEIGHTS = {
    "camel": 0.35,
    "stanza": 0.35,
    "udpipe": 0.15,
    "qalsadi": 0.10,
    "alkhalil": 0.05,
    "farasa": 0.05,
    "sinatools": 0.20,
}

_INVALID_VALUES = {None, "", "#", "X", "x", "UNK", "unknown", "None", "null"}


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


def _strip_arabic_diacritics(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = text.replace("\u0640", "")
    return text.strip()


def _normalize_root_for_fusion(value: Any) -> str:
    text = _strip_arabic_diacritics(value)
    text = re.sub(r"[.\s\-ـ]+", "", text)
    return text


def _normalize_lemma_for_fusion(value: Any) -> str:
    text = _strip_arabic_diacritics(value)
    text = re.sub(r"\d+$", "", text)
    text = text.replace("ٱ", "ا").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return text.strip()


def _normalize_pos_for_fusion(value: Any) -> Optional[str]:
    if value in _INVALID_VALUES:
        return None
    raw = str(value or "").strip()
    if not raw:
        return None

    alkhalil_mapped = normalize_alkhalil_pos(raw)
    if alkhalil_mapped:
        return alkhalil_mapped

    mapped = normalize_pos_for_compare(raw)
    if mapped and mapped not in _INVALID_VALUES:
        if mapped == "ADPOSITION":
            return "ADP"
        return mapped

    upper = raw.upper()
    if upper in {"NOUN", "VERB", "ADJ", "ADV", "ADP", "PRON", "PART", "DET", "CCONJ", "SCONJ", "PROPN", "NUM"}:
        return "NOUN" if upper == "PROPN" else upper
    return None


def _is_valid_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict)):
        return len(value) > 0
    text = str(value).strip()
    return text not in {"", "#", "X", "x", "UNK", "unknown", "None", "null"}


def score_pos(camel_pos_raw, stanza_pos_raw, *, partial_match: bool = False) -> tuple:
    camel_pos = _normalize_pos_for_fusion(camel_pos_raw)
    stanza_pos = _normalize_pos_for_fusion(stanza_pos_raw)
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
    if _is_valid_value(value):
        return str(value).strip()
    analyses = tok.get("analyses") or []
    if analyses and isinstance(analyses[0], dict):
        nested = analyses[0].get(key)
        if _is_valid_value(nested):
            return str(nested).strip()
    return None


def _has_dependency(tok: Optional[Dict[str, Any]]) -> bool:
    if not tok:
        return False
    dependency = tok.get("dependency")
    if not isinstance(dependency, dict):
        return False
    return any(dependency.get(key) not in (None, "", 0) for key in ("head", "head_text", "deprel"))


def _canonical_alkhalil_pos_for_fusion(
    alkhalil_tok: Optional[Dict[str, Any]],
    *,
    camel_tok=None,
    stanza_tok=None,
    udpipe_tok=None,
    sinatools_tok=None,
) -> tuple[Optional[str], Optional[str]]:
    votes: Dict[str, str] = {}
    for tool, tok in (
        ("camel", camel_tok),
        ("stanza", stanza_tok),
        ("udpipe", udpipe_tok),
        ("sinatools", sinatools_tok),
    ):
        raw = None
        if isinstance(tok, dict):
            raw = tok.get("pos") or tok.get("upos") or _token_value(tok, "pos") or _token_value(tok, "upos")
        norm = _normalize_pos_for_fusion(raw)
        if norm:
            votes[tool] = norm
    return extract_alkhalil_canonical_pos(alkhalil_tok, context_pos_votes=votes)


def _values_equal_for_feature(feature: str, a: Any, b: Any) -> bool:
    if not _is_valid_value(a) or not _is_valid_value(b):
        return False
    if feature == "pos":
        return _normalize_pos_for_fusion(a) == _normalize_pos_for_fusion(b)
    if feature == "lemma":
        return _normalize_lemma_for_fusion(a) == _normalize_lemma_for_fusion(b)
    if feature == "root":
        return _normalize_root_for_fusion(a) == _normalize_root_for_fusion(b)
    return str(a).strip() == str(b).strip()


def _clean_conflicts(conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove false conflicts and ensure displayed conflict values are normalized."""
    cleaned: List[Dict[str, Any]] = []
    for conflict in conflicts:
        feature = conflict.get("feature")

        if "tool_a_value" in conflict and "tool_b_value" in conflict:
            raw_a = conflict.get("raw_tool_a_value", conflict.get("tool_a_value"))
            raw_b = conflict.get("raw_tool_b_value", conflict.get("tool_b_value"))
            a = conflict.get("tool_a_value")
            b = conflict.get("tool_b_value")
            if not _is_valid_value(a) or not _is_valid_value(b):
                continue

            if feature == "pos":
                norm_a = _normalize_pos_for_fusion(a)
                norm_b = _normalize_pos_for_fusion(b)
                if not norm_a or not norm_b or norm_a == norm_b:
                    continue
                conflict = {**conflict, "tool_a_value": norm_a, "tool_b_value": norm_b, "raw_tool_a_value": raw_a, "raw_tool_b_value": raw_b}
            elif feature == "lemma":
                if _normalize_lemma_for_fusion(a) == _normalize_lemma_for_fusion(b):
                    continue
            elif feature == "root":
                if _normalize_root_for_fusion(a) == _normalize_root_for_fusion(b):
                    continue
            cleaned.append(conflict)
            continue

        values = conflict.get("values")
        if isinstance(values, dict):
            valid_values = {tool: value for tool, value in values.items() if _is_valid_value(value)}
            if len(valid_values) <= 1:
                continue
            if feature == "pos":
                normalized = {tool: _normalize_pos_for_fusion(value) for tool, value in valid_values.items() if _normalize_pos_for_fusion(value)}
                if len(set(normalized.values())) <= 1:
                    continue
            if feature == "lemma":
                normalized = {tool: _normalize_lemma_for_fusion(value) for tool, value in valid_values.items() if _normalize_lemma_for_fusion(value)}
                if len(set(normalized.values())) <= 1:
                    continue
            if feature == "root":
                normalized = {tool: _normalize_root_for_fusion(value) for tool, value in valid_values.items() if _normalize_root_for_fusion(value)}
                if len(set(normalized.values())) <= 1:
                    continue
        cleaned.append(conflict)
    return cleaned


def _decision_trace(*, final: Dict[str, Any], sources: Dict[str, Any], evidence: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    for feature in ("lemma", "root", "pos", "segmentation", "dependency"):
        value = final.get(feature)
        source = sources.get(feature)
        supporting: List[str] = []
        for tool, payload in evidence.items():
            if not isinstance(payload, dict):
                continue
            tool_value = payload.get(feature)
            if not _is_valid_value(tool_value):
                continue
            if feature == "pos":
                same = _normalize_pos_for_fusion(tool_value) == _normalize_pos_for_fusion(value)
            elif feature == "lemma":
                same = _normalize_lemma_for_fusion(tool_value) == _normalize_lemma_for_fusion(value)
            elif feature == "root":
                same = _normalize_root_for_fusion(tool_value) == _normalize_root_for_fusion(value)
            else:
                same = tool_value == value
            if same:
                supporting.append(tool)
        trace.append({"feature": feature, "chosen_value": value, "source": source, "supporting_tools": supporting, "conflict_count": len([c for c in conflicts if c.get("feature") == feature])})
    return trace


def score_to_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def compute_evidence_confidence(feature: str, selected_tool: str, selected_value, all_tool_values: dict, conflicts: list) -> float:
    _ = selected_tool
    if not _is_valid_value(selected_value):
        return 0.0
    valid = {t: v for t, v in all_tool_values.items() if _is_valid_value(v)}
    if not valid:
        return 0.25
    total_weight = sum(TOOL_WEIGHTS.get(t, 0.05) for t in valid)
    support_weight = sum(TOOL_WEIGHTS.get(t, 0.05) for t, v in valid.items() if _values_equal_for_feature(feature, v, selected_value))
    agreement_ratio = support_weight / total_weight if total_weight else 0.0
    real_conflicts = [c for c in conflicts if c.get("severity") in {"high", "medium"} or c.get("level") in {"high", "medium"}]
    conflict_penalty = min(0.30, 0.06 * len(real_conflicts))
    raw_score = 0.35 + 0.50 * agreement_ratio - conflict_penalty
    return round(max(0.10, min(0.95, raw_score)), 3)


def compute_token_confidence(fused: Dict[str, Any], *, camel_tok=None, stanza_tok=None, qalsadi_tok=None, alkhalil_tok=None, udpipe_tok=None, sinatools_tok=None) -> tuple[float, str]:
    alkhalil_pos, _ = _canonical_alkhalil_pos_for_fusion(alkhalil_tok, camel_tok=camel_tok, stanza_tok=stanza_tok, udpipe_tok=udpipe_tok, sinatools_tok=sinatools_tok)
    feature_payloads = [
        ("pos", fused.get("final", {}).get("pos"), fused.get("sources", {}).get("pos", ""), {
            "camel": _token_value(camel_tok, "pos") or _token_value(camel_tok, "upos"),
            "stanza": _token_value(stanza_tok, "pos") or _token_value(stanza_tok, "upos"),
            "qalsadi": _token_value(qalsadi_tok, "pos") or _token_value(qalsadi_tok, "upos"),
            "alkhalil": alkhalil_pos,
            "udpipe": _token_value(udpipe_tok, "pos") or _token_value(udpipe_tok, "upos"),
            "sinatools": _token_value(sinatools_tok, "pos") or _token_value(sinatools_tok, "upos"),
        }),
        ("lemma", fused.get("final", {}).get("lemma"), fused.get("sources", {}).get("lemma", ""), {
            "camel": _token_value(camel_tok, "lemma"),
            "stanza": _token_value(stanza_tok, "lemma"),
            "qalsadi": _token_value(qalsadi_tok, "lemma"),
            "alkhalil": _token_value(alkhalil_tok, "lemma"),
            "udpipe": _token_value(udpipe_tok, "lemma"),
            "sinatools": _token_value(sinatools_tok, "lemma"),
        }),
        ("root", fused.get("final", {}).get("root"), fused.get("sources", {}).get("root", ""), {
            "camel": _token_value(camel_tok, "root"),
            "qalsadi": _token_value(qalsadi_tok, "root"),
            "alkhalil": _token_value(alkhalil_tok, "root"),
            "udpipe": _token_value(udpipe_tok, "root"),
            "sinatools": _token_value(sinatools_tok, "root"),
        }),
    ]
    scores: List[float] = []
    for feature, selected_value, selected_tool, values_by_tool in feature_payloads:
        conflicts = [c for c in fused.get("conflicts", []) if c.get("feature") == feature]
        score = compute_evidence_confidence(feature, selected_tool, selected_value, values_by_tool, conflicts)
        if score > 0:
            scores.append(score)
    if not scores:
        return 0.0, score_to_level(0.0)
    final = round(sum(scores) / len(scores), 3)
    return final, score_to_level(final)


def fuse_token(word, camel_tok=None, stanza_tok=None, farasa_tok=None, qalsadi_tok=None, alkhalil_tok=None, udpipe_tok=None, sinatools_tok=None):
    fused: Dict[str, Any] = {"word": word, "final": {}, "sources": {}, "evidence": {}, "confidence": "medium", "notes": [], "conflicts": []}
    fix = KNOWN_FIXES.get(word, {})

    alkhalil_pos, alkhalil_raw_pos = _canonical_alkhalil_pos_for_fusion(alkhalil_tok, camel_tok=camel_tok, stanza_tok=stanza_tok, udpipe_tok=udpipe_tok, sinatools_tok=sinatools_tok)

    for name, tok in (("camel", camel_tok), ("stanza", stanza_tok), ("farasa", farasa_tok), ("qalsadi", qalsadi_tok), ("alkhalil", alkhalil_tok), ("udpipe", udpipe_tok), ("sinatools", sinatools_tok)):
        if tok:
            pos_value = alkhalil_pos if name == "alkhalil" else (tok.get("pos") or tok.get("upos"))
            fused["evidence"][name] = {"surface": tok.get("surface"), "lemma": tok.get("lemma"), "root": tok.get("root"), "pos": pos_value, "segmentation": tok.get("segmentation"), "dependency": tok.get("dependency")}

    if farasa_tok and farasa_tok.get("segmentation") and farasa_tok.get("status", "ok") == "ok":
        fused["final"]["segmentation"] = farasa_tok["segmentation"]
        fused["sources"]["segmentation"] = "farasa"
    else:
        fused["final"]["segmentation"] = [word]
        fused["sources"]["segmentation"] = "surface_fallback"
        fused["notes"].append("Farasa unavailable; used surface fallback segmentation.")

    camel_analyses = camel_tok.get("analyses", []) if camel_tok else []
    qalsadi_analyses = qalsadi_tok.get("analyses", []) if qalsadi_tok else []
    alkhalil_analyses = alkhalil_tok.get("analyses", []) if alkhalil_tok else []
    udpipe_analyses = udpipe_tok.get("analyses", []) if udpipe_tok else []
    sinatools_analyses = sinatools_tok.get("analyses", []) if sinatools_tok else []

    camel_lemma = camel_analyses[0].get("lemma") if camel_analyses else (_token_value(camel_tok, "lemma") if camel_tok else None)
    stanza_lemma = stanza_tok.get("lemma") if stanza_tok else None
    qalsadi_lemma = qalsadi_analyses[0].get("lemma") if qalsadi_analyses else (qalsadi_tok.get("lemma") if qalsadi_tok else None)
    alkhalil_lemma = alkhalil_analyses[0].get("lemma") if alkhalil_analyses else (_token_value(alkhalil_tok, "lemma") if alkhalil_tok else None)
    udpipe_lemma = udpipe_analyses[0].get("lemma") if udpipe_analyses else (_token_value(udpipe_tok, "lemma") if udpipe_tok else None)
    sinatools_lemma = sinatools_analyses[0].get("lemma") if sinatools_analyses else (_token_value(sinatools_tok, "lemma") if sinatools_tok else None)

    if camel_lemma:
        fused["final"]["lemma"] = camel_lemma; fused["sources"]["lemma"] = "camel"
    elif sinatools_lemma:
        fused["final"]["lemma"] = sinatools_lemma; fused["sources"]["lemma"] = "sinatools"
    elif stanza_lemma:
        fused["final"]["lemma"] = stanza_lemma; fused["sources"]["lemma"] = "stanza"
    elif qalsadi_lemma:
        fused["final"]["lemma"] = qalsadi_lemma; fused["sources"]["lemma"] = "qalsadi"
    elif alkhalil_lemma:
        fused["final"]["lemma"] = alkhalil_lemma; fused["sources"]["lemma"] = "alkhalil"
    elif udpipe_lemma:
        fused["final"]["lemma"] = udpipe_lemma; fused["sources"]["lemma"] = "udpipe"

    camel_root = _token_value(camel_tok, "root")
    alkhalil_root = _token_value(alkhalil_tok, "root")
    sinatools_root = _token_value(sinatools_tok, "root")
    qalsadi_root = _token_value(qalsadi_tok, "root")
    udpipe_root = _token_value(udpipe_tok, "root")

    if camel_root:
        fused["final"]["root"] = camel_root; fused["final"]["root_type"] = _camel_root_type(camel_tok); fused["sources"]["root"] = "camel"
    elif alkhalil_root:
        fused["final"]["root"] = alkhalil_root; fused["sources"]["root"] = "alkhalil"
    elif sinatools_root:
        fused["final"]["root"] = sinatools_root; fused["sources"]["root"] = "sinatools"
    elif qalsadi_root:
        fused["final"]["root"] = qalsadi_root; fused["sources"]["root"] = "qalsadi"
    elif udpipe_root:
        fused["final"]["root"] = udpipe_root; fused["sources"]["root"] = "udpipe"

    if camel_analyses:
        fused["final"]["gloss"] = camel_analyses[0].get("gloss")
        fused["sources"]["gloss"] = "camel"

    camel_pos_raw = fix.get("pos") or (camel_analyses[0].get("pos") if camel_analyses else None) or _token_value(camel_tok, "pos")
    stanza_pos_raw = stanza_tok.get("upos") if stanza_tok else None
    if not stanza_pos_raw and udpipe_tok:
        stanza_pos_raw = udpipe_tok.get("pos") or udpipe_tok.get("upos")
    if not stanza_pos_raw and sinatools_tok:
        stanza_pos_raw = sinatools_tok.get("pos") or sinatools_tok.get("upos")
    if not stanza_pos_raw and qalsadi_tok:
        stanza_pos_raw = qalsadi_tok.get("pos") or qalsadi_tok.get("upos")
    if not stanza_pos_raw and alkhalil_pos:
        stanza_pos_raw = alkhalil_pos

    partial_match = _alignment_match_type(stanza_tok) == "partial_match" or _alignment_match_type(camel_tok) == "partial_match"
    final_pos, pos_source, _, pos_notes = score_pos(camel_pos_raw, stanza_pos_raw, partial_match=partial_match)
    fused["final"]["pos"] = final_pos
    fused["sources"]["pos"] = "camel" if pos_source == "agreement" and camel_pos_raw else pos_source
    fused["notes"].extend(pos_notes)

    raw_conflicts = build_conflicts(camel_tok=camel_tok, stanza_tok=stanza_tok, qalsadi_tok=qalsadi_tok, alkhalil_tok=alkhalil_tok, udpipe_tok=udpipe_tok)
    for c in raw_conflicts:
        if c.get("feature") != "pos":
            continue
        if c.get("tool_a") == "alkhalil":
            c["raw_tool_a_value"] = alkhalil_raw_pos or c.get("tool_a_value")
            c["tool_a_value"] = alkhalil_pos or _normalize_pos_for_fusion(c.get("tool_a_value")) or c.get("tool_a_value")
        if c.get("tool_b") == "alkhalil":
            c["raw_tool_b_value"] = alkhalil_raw_pos or c.get("tool_b_value")
            c["tool_b_value"] = alkhalil_pos or _normalize_pos_for_fusion(c.get("tool_b_value")) or c.get("tool_b_value")
    fused["conflicts"].extend(_clean_conflicts(raw_conflicts))

    if camel_analyses:
        fused["final"]["gender"] = camel_analyses[0].get("gender")
        fused["final"]["number"] = camel_analyses[0].get("number")
        fused["final"]["tense"] = camel_analyses[0].get("tense")
        fused["sources"]["morphology"] = "camel"

    stanza_case = stanza_tok.get("case") if stanza_tok else None
    udpipe_case = udpipe_tok.get("case") if udpipe_tok else None
    if stanza_case:
        fused["final"]["case"] = stanza_case; fused["final"]["definite"] = stanza_tok.get("definite"); fused["sources"]["case"] = "stanza"
    elif udpipe_case:
        fused["final"]["case"] = udpipe_case; fused["sources"]["case"] = "udpipe"

    if _has_dependency(stanza_tok):
        fused["final"]["dependency"] = stanza_tok["dependency"]; fused["sources"]["dependency"] = "stanza"
    elif _has_dependency(udpipe_tok):
        fused["final"]["dependency"] = udpipe_tok["dependency"]; fused["sources"]["dependency"] = "udpipe"
    elif _has_dependency(qalsadi_tok):
        fused["final"]["dependency"] = qalsadi_tok["dependency"]; fused["sources"]["dependency"] = "qalsadi"
    elif _has_dependency(alkhalil_tok):
        fused["final"]["dependency"] = alkhalil_tok["dependency"]; fused["sources"]["dependency"] = "alkhalil"

    conf_score, conf_level = compute_token_confidence(fused, camel_tok=camel_tok, stanza_tok=stanza_tok, qalsadi_tok=qalsadi_tok, alkhalil_tok=alkhalil_tok, udpipe_tok=udpipe_tok, sinatools_tok=sinatools_tok)

    udpipe_pos_raw = (udpipe_tok.get("pos") or udpipe_tok.get("upos")) if udpipe_tok else None
    if final_pos and udpipe_pos_raw and _normalize_pos_for_fusion(udpipe_pos_raw) == _normalize_pos_for_fusion(final_pos):
        fused["notes"].append("UDPipe confirms POS")
    sinatools_pos_raw = (sinatools_tok.get("pos") or sinatools_tok.get("upos")) if sinatools_tok else None
    if final_pos and sinatools_pos_raw and _normalize_pos_for_fusion(sinatools_pos_raw) == _normalize_pos_for_fusion(final_pos):
        fused["notes"].append("SinaTools supports POS")
    if alkhalil_pos and final_pos and _normalize_pos_for_fusion(alkhalil_pos) == _normalize_pos_for_fusion(final_pos):
        fused["notes"].append("AlKhalil supports POS")

    if sinatools_lemma and fused["final"].get("lemma") and _normalize_lemma_for_fusion(sinatools_lemma) == _normalize_lemma_for_fusion(fused["final"].get("lemma")):
        fused["notes"].append("SinaTools supports lemma")
    if sinatools_root and fused["final"].get("root") and _normalize_root_for_fusion(sinatools_root) == _normalize_root_for_fusion(fused["final"].get("root")):
        fused["notes"].append("SinaTools supports root")
    if stanza_case and udpipe_case and str(stanza_case).strip().lower() == str(udpipe_case).strip().lower():
        fused["notes"].append("UDPipe confirms case")

    fused["final"]["confidence_score"] = conf_score
    fused["final"]["confidence_level"] = conf_level
    fused["confidence"] = conf_level
    if fix:
        fused["notes"].append(f"applied known_fix for '{word}'")
    fused["decision_trace"] = _decision_trace(final=fused["final"], sources=fused["sources"], evidence=fused["evidence"], conflicts=fused["conflicts"])
    return fused


def fusion_system(text, camel_res, stanza_res, farasa_res, qalsadi_res=None, all_tool_results=None):
    from backend.services.alignment_engine import align_tools
    from backend.services.normalizer import normalize_tool_output

    source_results = {"camel": camel_res, "stanza": stanza_res, "farasa": farasa_res, "qalsadi": qalsadi_res}
    if isinstance(all_tool_results, dict):
        source_results.update(all_tool_results)
    normalized = {name: normalize_tool_output(name, payload or {}) for name, payload in source_results.items()}

    base_tool = "farasa"
    farasa_status = normalized.get("farasa", {}).get("status")
    farasa_tokens = normalized.get("farasa", {}).get("tokens", []) or []
    tool_tokens = {
        "camel": normalized.get("camel", {}).get("tokens", []) or [],
        "stanza": normalized.get("stanza", {}).get("tokens", []) or [],
        "qalsadi": normalized.get("qalsadi", {}).get("tokens", []) or [],
        "alkhalil": normalized.get("alkhalil", {}).get("tokens", []) or [],
        "udpipe": normalized.get("udpipe", {}).get("tokens", []) or [],
        "sinatools": normalized.get("sinatools", {}).get("tokens", []) or [],
    }
    base_tokens = farasa_tokens if farasa_status == "ok" else []
    if not base_tokens:
        for candidate in ("camel", "stanza", "udpipe", "qalsadi", "alkhalil", "sinatools"):
            candidate_tokens = tool_tokens.get(candidate) or []
            if candidate_tokens:
                base_tool = candidate
                base_tokens = candidate_tokens
                break

    aligned_tokens, _meta = align_tools(base_tokens=base_tokens, tools_tokens=tool_tokens)
    fused_output = []
    for atok in aligned_tokens:
        word = atok.base["surface"]
        classic_fused = fuse_token(
            word,
            camel_tok=atok.tools.get("camel"),
            stanza_tok=atok.tools.get("stanza"),
            farasa_tok=atok.base if base_tool == "farasa" else atok.tools.get("farasa"),
            qalsadi_tok=atok.tools.get("qalsadi"),
            alkhalil_tok=atok.tools.get("alkhalil"),
            udpipe_tok=atok.tools.get("udpipe"),
            sinatools_tok=atok.tools.get("sinatools"),
        )

        expert_fused = apply_expert_fusion(
            classic_fused=classic_fused,
            tools={
                "camel": atok.tools.get("camel"),
                "stanza": atok.tools.get("stanza"),
                "farasa": atok.base if base_tool == "farasa" else atok.tools.get("farasa"),
                "qalsadi": atok.tools.get("qalsadi"),
                "alkhalil": atok.tools.get("alkhalil"),
                "udpipe": atok.tools.get("udpipe"),
                "sinatools": atok.tools.get("sinatools"),
            },
        )

        fused_output.append(expert_fused)

    return {"text": text, "fusion": fused_output, "meta": {"base_tool": base_tool, "base_token_count": len(base_tokens), "farasa_status": farasa_status, "sinatools_tokens": len(tool_tokens.get("sinatools") or [])}}
