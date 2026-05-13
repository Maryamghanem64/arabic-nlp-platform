from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.analyzers.base import Analyzer
from backend.services.comparison_service import build_comparison
from backend.services.confidence_service import compute_tool_confidence
from backend.services.fusion_service import fuse_token
from backend.services.tool_runner import ToolRunner
from backend.services.comparison_service import TokenComparisonOutput
from backend.services.suspicious_service import detect_suspicious_for_token


def _get_tool_token_by_index(tool_res: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    tokens = tool_res.get("tokens") if tool_res else None
    if not tokens or idx >= len(tokens):
        return None
    return tokens[idx]


def align_tokens_by_index(
    *,
    tool_results: Dict[str, Dict[str, Any]],
    reference_tool: str = "farasa",
) -> List[str]:
    # Current implementation uses farasa tokenization order as baseline.
    # Each tool output is assumed token-aligned by index.
    farasa_tokens = tool_results.get(reference_tool, {}).get("tokens", [])
    return [t.get("surface") for t in farasa_tokens]


def compute_conf_by_tool_and_token(
    *,
    token_surface: str,
    tool_name: str,
    tok: Optional[Dict[str, Any]],
    tool_meta: Dict[str, Any],
    suspicious_flags: List[str],
) -> float:
    pos = tok.get("pos") or tok.get("upos") if tok else None
    lemma = tok.get("lemma") if tok else None
    segmentation = tok.get("segmentation") if tok else None
    return compute_tool_confidence(
        tool_name=tool_name,
        token_surface=token_surface,
        pos=pos,
        lemma=lemma,
        segmentation=segmentation,
        meta=tool_meta,
        suspicious_flags=suspicious_flags,
    )


def fusion_for_text(
    *,
    text: str,
    runner: ToolRunner,
    tool_results: Dict[str, Dict[str, Any]],
    tool_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    # tools expected: camel, stanza, qalsadi, farasa
    token_surfaces = align_tokens_by_index(tool_results=tool_results, reference_tool="farasa")

    fusion_tokens: List[Any] = []

    # Precompute suspicious + confidence per token per tool
    for idx, surface in enumerate(token_surfaces):
        camel_tok = _get_tool_token_by_index(tool_results.get("camel"), idx)
        stanza_tok = _get_tool_token_by_index(tool_results.get("stanza"), idx)
        qalsadi_tok = _get_tool_token_by_index(tool_results.get("qalsadi"), idx)
        farasa_tok = _get_tool_token_by_index(tool_results.get("farasa"), idx)

        # Build comparisons
        comp: TokenComparisonOutput = build_comparison(
            camel_tok=camel_tok,
            stanza_tok=stanza_tok,
            qalsadi_tok=qalsadi_tok,
            farasa_tok=farasa_tok,
        )

        # suspicious flags per tool
        susp_by_tool: Dict[str, List[str]] = {}
        for name, tok in [
            ("camel", camel_tok),
            ("stanza", stanza_tok),
            ("qalsadi", qalsadi_tok),
            ("farasa", farasa_tok),
        ]:
            lemma = tok.get("lemma") if tok else None
            pos = tok.get("pos") or tok.get("upos") if tok else None
            susp_by_tool[name] = detect_suspicious_for_token(
                surface=surface,
                lemma=lemma,
                pos=pos,
                tool_name=name,
            )

        # confidence per tool
        tool_confs: Dict[str, float] = {}
        for name, tok in [
            ("camel", camel_tok),
            ("stanza", stanza_tok),
            ("qalsadi", qalsadi_tok),
            ("farasa", farasa_tok),
        ]:
            tool_confs[name] = compute_conf_by_tool_and_token(
                token_surface=surface,
                tool_name=name,
                tok=tok,
                tool_meta=tool_meta.get(name, {}),
                suspicious_flags=susp_by_tool[name],
            )

        comparisons = comp

        token_fusion = fuse_token(
            token_surface=surface,
            tool_results={
                "camel": camel_tok or {},
                "stanza": stanza_tok or {},
                "qalsadi": qalsadi_tok or {},
                "farasa": farasa_tok or {},
            },
            tool_confs=tool_confs,
            comparisons=comparisons,
        )

        # attach structured comparison statuses and flags
        fusion_tokens.append({
            "token": token_fusion.token,
            "tools": token_fusion.tools,
            "comparison": {
                "pos_status": token_fusion.comparison.pos_status,
                "lemma_status": token_fusion.comparison.lemma_status,
                "segmentation_status": token_fusion.comparison.segmentation_status,
                "details": token_fusion.comparison.details,
            },
            "fusion": {
                "final_pos": token_fusion.fusion.final_pos,
                "final_lemma": token_fusion.fusion.final_lemma,
                "final_segmentation": token_fusion.fusion.final_segmentation,
                "confidence": token_fusion.fusion.confidence,
                "confidence_level": token_fusion.fusion.confidence_level,
                "chosen_sources": token_fusion.fusion.chosen_sources,
            },
            "flags": token_fusion.flags,
        })

    return {"text": text, "fusion_result": fusion_tokens}

