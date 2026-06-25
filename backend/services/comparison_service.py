from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.schemas.analysis_schemas import MatchStatus
from app.utils.helpers import (
    classify_conflict,
    is_gender_convention_pair,
    is_mwt_clitic_artifact,
    normalize_lemma_for_compare,
    normalize_pos_for_compare,
)

_NON_CONTENT_POS = {"CCONJ", "SCONJ", "PART", "PUNCT", "SYM"}


def _lemma_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    if is_mwt_clitic_artifact(a) or is_mwt_clitic_artifact(b):
        return True
    if is_gender_convention_pair(a, b):
        return True
    return normalize_lemma_for_compare(a) == normalize_lemma_for_compare(b)


def _pos_equal(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return str(a).upper() == str(b).upper()


def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_pos(tok: Optional[Dict[str, Any]]) -> Optional[str]:
    if not tok:
        return None

    candidates: List[str] = []
    analyses = tok.get("analyses")
    if isinstance(analyses, list):
        for analysis in analyses:
            if not isinstance(analysis, dict):
                continue
            candidate = _first_non_empty(analysis.get("pos"), analysis.get("upos"))
            if candidate:
                candidates.append(candidate)

    candidate = _first_non_empty(tok.get("pos"), tok.get("upos"))
    if candidate:
        candidates.append(candidate)

    mwt_words = tok.get("mwt_words")
    if isinstance(mwt_words, list):
        for word in mwt_words:
            if not isinstance(word, dict):
                continue
            candidate = _first_non_empty(word.get("upos"), word.get("pos"))
            if candidate:
                candidates.append(candidate)

    normalized = [normalize_pos_for_compare(value) for value in candidates if value]
    if not normalized:
        return None

    for value in normalized:
        if value not in _NON_CONTENT_POS and value != "X":
            return value
    return normalized[0]


def _extract_lemma(tok: Optional[Dict[str, Any]]) -> Optional[str]:
    if not tok:
        return None
    analyses = tok.get("analyses")
    if isinstance(analyses, list) and analyses:
        first = analyses[0] if isinstance(analyses[0], dict) else {}
        lemma = _first_non_empty(first.get("lemma"))
        if lemma:
            return lemma
    return _first_non_empty(tok.get("lemma"))


def compare_match_status(vals: List[Optional[Any]], kind: str) -> MatchStatus:
    non_null = [v for v in vals if v is not None and str(v) != ""]
    if len(non_null) < 2:
        return MatchStatus.PARTIAL_MATCH

    first = non_null[0]
    for v in non_null[1:]:
        if kind == "pos":
            if not _pos_equal(normalize_pos_for_compare(first), normalize_pos_for_compare(v)):
                return MatchStatus.CONFLICT
        else:
            if not _lemma_equal(first, v):
                return MatchStatus.CONFLICT

    return MatchStatus.FULL_MATCH


@dataclass(frozen=True)
class TokenComparisonOutput:
    pos_status: MatchStatus
    lemma_status: MatchStatus
    segmentation_status: MatchStatus
    details: Dict[str, Any]


def build_conflicts(
    *,
    camel_tok: Optional[Dict[str, Any]],
    stanza_tok: Optional[Dict[str, Any]],
    qalsadi_tok: Optional[Dict[str, Any]],
    alkhalil_tok: Optional[Dict[str, Any]] = None,
    udpipe_tok: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    tool_map = {
        "camel": camel_tok,
        "stanza": stanza_tok,
        "qalsadi": qalsadi_tok,
        "alkhalil": alkhalil_tok,
        "udpipe": udpipe_tok,
    }

    pos_by_tool: Dict[str, Optional[str]] = {tool: _extract_pos(tok) for tool, tok in tool_map.items()}
    conflicts: List[Dict[str, Any]] = []

    ordered_tools = [tool for tool, tok in tool_map.items() if tok and _extract_pos(tok)]
    for i, tool_a in enumerate(ordered_tools):
        pos_a = pos_by_tool.get(tool_a)
        if not pos_a:
            continue
        for tool_b in ordered_tools[i + 1 :]:
            pos_b = pos_by_tool.get(tool_b)
            if not pos_b or pos_a == pos_b:
                continue
            conflicts.append(
                {
                    "feature": "pos",
                    "tool_a": tool_a,
                    "tool_b": tool_b,
                    "tool_a_value": pos_a,
                    "tool_b_value": pos_b,
                    "severity": classify_conflict("pos", pos_a, pos_b).get("severity", "high"),
                }
            )

    return conflicts


def build_comparison(
    *,
    camel_tok: Optional[Dict[str, Any]],
    stanza_tok: Optional[Dict[str, Any]],
    qalsadi_tok: Optional[Dict[str, Any]],
    farasa_tok: Optional[Dict[str, Any]],
) -> TokenComparisonOutput:
    pos_vals = [
        _extract_pos(camel_tok),
        _extract_pos(stanza_tok),
        _extract_pos(qalsadi_tok),
    ]
    # lemma candidates
    lemma_vals = [
        _extract_lemma(camel_tok),
        _extract_lemma(stanza_tok),
        _extract_lemma(qalsadi_tok),
    ]

    pos_status = compare_match_status(pos_vals, kind="pos")
    lemma_status = compare_match_status(lemma_vals, kind="lemma")

    seg_status = MatchStatus.CONFLICT
    if farasa_tok and farasa_tok.get("segmentation"):
        # If other tools don't provide segmentation, we label as PARTIAL to reflect
        # comparative value without overclaiming.
        seg_status = MatchStatus.PARTIAL_MATCH

    details = {
        "pos_vals": pos_vals,
        "lemma_vals": lemma_vals,
        "segmentation_farasa": farasa_tok.get("segmentation") if farasa_tok else None,
        "conflicts": build_conflicts(
            camel_tok=camel_tok,
            stanza_tok=stanza_tok,
            qalsadi_tok=qalsadi_tok,
        ),
    }

    return TokenComparisonOutput(
        pos_status=pos_status,
        lemma_status=lemma_status,
        segmentation_status=seg_status,
        details=details,
    )

