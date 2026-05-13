from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.schemas.analysis_schemas import MatchStatus
from backend.utils.text_norm import strip_diacritics


def _lemma_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return strip_diacritics(str(a)) == strip_diacritics(str(b))


def _pos_equal(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return str(a).upper() == str(b).upper()


def compare_match_status(vals: List[Optional[Any]], kind: str) -> MatchStatus:
    non_null = [v for v in vals if v is not None and str(v) != ""]
    if len(non_null) < 2:
        return MatchStatus.PARTIAL_MATCH

    first = non_null[0]
    for v in non_null[1:]:
        if kind == "pos":
            if not _pos_equal(first, v):
                return MatchStatus.CONFLICT
        else:
            if strip_diacritics(str(first)) != strip_diacritics(str(v)):
                return MatchStatus.CONFLICT

    return MatchStatus.FULL_MATCH


@dataclass(frozen=True)
class TokenComparisonOutput:
    pos_status: MatchStatus
    lemma_status: MatchStatus
    segmentation_status: MatchStatus
    details: Dict[str, Any]


def build_comparison(
    *,
    camel_tok: Optional[Dict[str, Any]],
    stanza_tok: Optional[Dict[str, Any]],
    qalsadi_tok: Optional[Dict[str, Any]],
    farasa_tok: Optional[Dict[str, Any]],
) -> TokenComparisonOutput:
    pos_vals = [
        camel_tok.get("pos") if camel_tok else None,
        stanza_tok.get("pos") if stanza_tok else None,
        qalsadi_tok.get("pos") if qalsadi_tok else None,
    ]
    # lemma candidates
    lemma_vals = [
        camel_tok.get("lemma") if camel_tok else None,
        stanza_tok.get("lemma") if stanza_tok else None,
        qalsadi_tok.get("lemma") if qalsadi_tok else None,
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
    }

    return TokenComparisonOutput(
        pos_status=pos_status,
        lemma_status=lemma_status,
        segmentation_status=seg_status,
        details=details,
    )

