from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.schemas.analysis_schemas import TokenComparison, TokenFusion
from backend.services.confidence_service import compute_fusion_confidence
from backend.services.suspicious_service import detect_suspicious_for_token


# unify POS labels into a coarse tagset
POS_UNIFIED = {
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "ADJECTIVE": "ADJ",
    "ADPOSITION": "ADP",
    "ADP": "ADP",
    "PRONOUN": "PRON",
    "PRON": "PRON",
    "ADVERB": "ADV",
    "ADV": "ADV",
    "CONJUNCTION": "CCONJ",
    "CCONJ": "CCONJ",
    "PARTICLE": "PART",
    "PART": "PART",
    "PUNCTUATION": "PUNCT",
    "PUNCT": "PUNCT",
}


def _unify_pos(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    key = str(pos).strip().upper()
    return POS_UNIFIED.get(key, key)


def _choose_by_confidence(
    candidates: Dict[str, Dict[str, Any]],
    feature: str,
    chosen_values: Dict[str, Any],
) -> tuple[Optional[Any], str, Dict[str, float]]:
    """candidates: {tool_name: {feature_value, confidence, suspicious_flags}}"""
    conf_by_tool: Dict[str, float] = {}
    best_tool = ""
    best_val = None
    best_conf = -1.0

    for tool_name, payload in candidates.items():
        val = payload.get(feature)
        conf = float(payload.get("confidence", 0.0))
        conf_by_tool[tool_name] = conf

        if val is None or val == "":
            continue

        # downweight qalsadi when suspicious
        if tool_name == "qalsadi" and payload.get("suspicious_flags"):
            conf -= 0.25 * min(3, len(payload.get("suspicious_flags")))

        if conf > best_conf:
            best_conf = conf
            best_tool = tool_name
            best_val = val

    if best_tool:
        chosen_values[feature] = best_val
        return best_val, best_tool, conf_by_tool

    return None, "none", conf_by_tool


def fuse_token(
    token_surface: str,
    tool_results: Dict[str, Dict[str, Any]],
    tool_confs: Dict[str, float],
    comparisons: TokenComparison,
) -> TokenFusion:
    # per-tool token analyses are already aligned by index upstream
    tools_payload: Dict[str, Any] = {}
    for tool_name, tok in tool_results.items():
        if not tok:
            continue
        tools_payload[tool_name] = tok

    # suspicious flags for each tool/token
    susp_by_tool: Dict[str, List[str]] = {}
    for tool_name, tok in tools_payload.items():
        lemma = tok.get("lemma")
        pos = tok.get("pos")
        susp_by_tool[tool_name] = detect_suspicious_for_token(
            surface=token_surface,
            lemma=lemma,
            pos=pos,
            tool_name=tool_name,
        )

    pos_candidates: Dict[str, Dict[str, Any]] = {}
    lemma_candidates: Dict[str, Dict[str, Any]] = {}
    seg_candidates: Dict[str, Dict[str, Any]] = {}

    for tool_name, tok in tools_payload.items():
        pos_val = _unify_pos(tok.get("pos") or tok.get("upos"))
        lemma_val = tok.get("lemma")
        seg_val = tok.get("segmentation")

        payload = {
            "pos": pos_val,
            "lemma": lemma_val,
            "segmentation": seg_val,
            "confidence": tool_confs.get(tool_name, 0.0),
            "suspicious_flags": susp_by_tool.get(tool_name, []),
        }
        pos_candidates[tool_name] = payload
        lemma_candidates[tool_name] = payload
        seg_candidates[tool_name] = payload

    chosen_sources: Dict[str, str] = {}
    final_pos, pos_src, _ = _choose_by_confidence(pos_candidates, "pos", chosen_sources)
    if pos_src and pos_src != "none" and final_pos is not None:
        chosen_sources["pos"] = pos_src

    final_lemma, lemma_src, _ = _choose_by_confidence(lemma_candidates, "lemma", chosen_sources)
    if lemma_src and lemma_src != "none" and final_lemma is not None:
        chosen_sources["lemma"] = lemma_src

    # segmentation: prefer farasa if available and reasonable
    final_seg = None
    seg_src = ""
    if "farasa" in seg_candidates and seg_candidates["farasa"].get("segmentation"):
        final_seg = seg_candidates["farasa"].get("segmentation")
        seg_src = "farasa"
    else:
        final_seg = [token_surface]
        seg_src = "fallback"
    chosen_sources["segmentation"] = seg_src

    # confidence breakdown for fusion
    pos_conf_by_tool = {src: tool_confs.get(src, 0.0) for src in pos_candidates.keys()}
    lemma_conf_by_tool = {src: tool_confs.get(src, 0.0) for src in lemma_candidates.keys()}

    fusion_conf = compute_fusion_confidence(
        pos_conf_by_tool=pos_conf_by_tool,
        lemma_conf_by_tool=lemma_conf_by_tool,
        chosen_sources=chosen_sources,
    )

    # chosen flags aggregation: if qalsadi suspicious, include its flags
    flags: List[str] = []
    if "qalsadi" in susp_by_tool:
        flags.extend([f for f in susp_by_tool["qalsadi"]])

    return TokenFusion(
        token=token_surface,
        tools={
            tool_name: {
                "surface": token_surface,
                "lemma": tok.get("lemma"),
                "pos": _unify_pos(tok.get("pos") or tok.get("upos")),
                "segmentation": tok.get("segmentation"),
                "raw": tok,
            }
            for tool_name, tok in tools_payload.items()
        },
        comparison=comparisons,
        fusion={
            "final_pos": final_pos,
            "final_lemma": final_lemma,
            "final_segmentation": final_seg,
            "confidence": float(fusion_conf),
            "confidence_level": "high" if fusion_conf >= 0.9 else "medium" if fusion_conf >= 0.6 else "low",
            "chosen_sources": chosen_sources,
        },
        flags=flags,
    )

