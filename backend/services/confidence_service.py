from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.config.settings import Settings
from backend.services.suspicious_service import detect_suspicious_for_token


CONF_LEVELS = [(0.9, "high"), (0.6, "medium"), (0.0, "low")]


def confidence_bucket(score: float) -> str:
    for thr, level in CONF_LEVELS:
        if score >= thr:
            return level
    return "low"


def _is_unknown_pos(pos: Optional[str]) -> bool:
    return not pos or str(pos).upper() in {"UNKNOWN", "X"}


def compute_tool_confidence(
    *,
    tool_name: str,
    token_surface: str,
    pos: Optional[str],
    lemma: Optional[str],
    segmentation: Optional[List[str]],
    meta: Dict[str, Any],
    suspicious_flags: List[str],
) -> float:
    base = float(meta.get("confidence_weight", 0.1))

    score = base

    # POS
    if _is_unknown_pos(pos):
        score -= 0.25
    else:
        score += 0.25

    # lemma validity
    if lemma is None or str(lemma).strip() == "":
        score -= 0.25
    else:
        # very short lemmas are suspicious
        if len(str(lemma).strip()) < 2:
            score -= 0.2
        else:
            score += 0.1

    # segmentation availability
    if tool_name == "farasa":
        if not segmentation:
            score -= 0.25
        else:
            score += 0.25

    # suspicious flags reduce
    if suspicious_flags:
        score -= min(0.35, 0.08 * len(suspicious_flags))

    # clamp
    score = max(0.0, min(1.0, score))
    return round(score, 3)


def compute_fusion_confidence(
    *,
    pos_conf_by_tool: Dict[str, float],
    lemma_conf_by_tool: Dict[str, float],
    chosen_sources: Dict[str, str],
) -> float:
    # average the confidence of chosen sources
    pos_src = chosen_sources.get("pos")
    lemma_src = chosen_sources.get("lemma")

    pos_c = pos_conf_by_tool.get(pos_src, 0.0) if pos_src else 0.0
    lemma_c = lemma_conf_by_tool.get(lemma_src, 0.0) if lemma_src else 0.0

    final = 0.5 * (pos_c + lemma_c)
    final = max(0.0, min(1.0, final))
    return round(final, 3)

