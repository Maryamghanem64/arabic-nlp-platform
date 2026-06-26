from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.utils.helpers import classify_conflict, normalize_lemma_for_compare, normalize_pos_for_compare
from backend.services.alignment_engine import align_tools, compute_agreements


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel) for normalized comparison."""
    if not text:
        return text
    return re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670]", "", text)


def lemma_equivalent(a: str | None, b: str | None) -> bool:
    """Normalized lemma comparison: strip diacritics then compare."""
    if not a or not b:
        return False
    return strip_diacritics(a.strip()) == strip_diacritics(b.strip())



def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def evaluate_tools(
    text: str,
    camel_res,
    stanza_res,
    farasa_res,
    qalsadi_res=None,
    all_tool_results: Dict[str, Dict] | None = None,
):
    """Evaluate CAMeL vs Stanza using surface-string alignment.

    Also reports excluded_tools based on statuses of *all* tools.
    """
    excluded = {"error", "unavailable", "future_work", "lazy", "disabled", "timeout"}
    metric_tools = ("camel", "farasa", "stanza", "qalsadi", "udpipe")

    if all_tool_results is None:
        all_tool_results = {}

    all_statuses: Dict[str, str | None] = {}
    for name, res in all_tool_results.items():
        if isinstance(res, dict):
            all_statuses[name] = res.get("status")
        else:
            all_statuses[name] = None

    # Ensure core tools are also represented even if caller didn't pass them.
    all_statuses.setdefault("camel", camel_res.get("status") if isinstance(camel_res, dict) else None)
    all_statuses.setdefault("farasa", farasa_res.get("status") if isinstance(farasa_res, dict) else None)
    all_statuses.setdefault("stanza", stanza_res.get("status") if isinstance(stanza_res, dict) else None)
    all_statuses.setdefault("qalsadi", qalsadi_res.get("status") if isinstance(qalsadi_res, dict) else None)

    active_tools = [t for t, s in all_statuses.items() if s == "ok"]
    active_tool_count = len(active_tools)
    excluded_tools = sorted(
        {
            t
            for t, s in all_statuses.items()
            if s in excluded
        }
    )


    from backend.services.normalizer import normalize_tool_output

    normalized_results = {
        name: normalize_tool_output(name, res if isinstance(res, dict) else {})
        for name, res in all_tool_results.items()
    }
    normalized_results.setdefault("camel", normalize_tool_output("camel", camel_res if isinstance(camel_res, dict) else {}))
    normalized_results.setdefault("stanza", normalize_tool_output("stanza", stanza_res if isinstance(stanza_res, dict) else {}))
    normalized_results.setdefault("farasa", normalize_tool_output("farasa", farasa_res if isinstance(farasa_res, dict) else {}))
    normalized_results.setdefault("qalsadi", normalize_tool_output("qalsadi", qalsadi_res if isinstance(qalsadi_res, dict) else {}))

    farasa_tokens_filtered = [
        t
        for t in (normalized_results.get("farasa", {}).get("tokens", []) or [])
        if isinstance(t, dict) and t.get("surface")
    ]

    total = len([t.get("surface") for t in farasa_tokens_filtered if t.get("surface")])

    aligned_tokens, _meta = align_tools(
        base_tokens=farasa_tokens_filtered,
        tools_tokens={
            "camel": normalized_results.get("camel", {}).get("tokens", []) or [],
            "stanza": normalized_results.get("stanza", {}).get("tokens", []) or [],
            "qalsadi": normalized_results.get("qalsadi", {}).get("tokens", []) or [],
            "alkhalil": normalized_results.get("alkhalil", {}).get("tokens", []) or [],
            "udpipe": normalized_results.get("udpipe", {}).get("tokens", []) or [],
        },
    )

    agreements = compute_agreements(aligned_tokens=aligned_tokens)

    pos_agreement = (agreements.get("pos_agreement", 0) / 100) if total else 0
    lemma_exact = (agreements.get("lemma_exact_agreement", 0) / 100) if total else 0
    lemma_normalized = (agreements.get("lemma_normalized_agreement", agreements.get("lemma_agreement", 0)) / 100) if total else 0

    pos_precision = pos_recall = pos_f1 = round(pos_agreement, 3)

    seg_cov = 0
    for atok in aligned_tokens:
        base_seg = atok.base.get("segmentation")
        if isinstance(base_seg, list) and base_seg:
            seg_cov += 1

    conflicts = []
    all_conflicts = []

    for atok in aligned_tokens:
        camel_tok = atok.tools.get("camel")
        stanza_tok = atok.tools.get("stanza")

        camel_ana = None
        if camel_tok and isinstance(camel_tok, dict):
            analyses = camel_tok.get("analyses") or []
            if analyses and isinstance(analyses[0], dict):
                camel_ana = analyses[0]

        w = atok.base.get("surface")

        if camel_ana and stanza_tok:
            camel_pos = normalize_pos_for_compare(camel_ana.get("pos"))
            stanza_pos = normalize_pos_for_compare(stanza_tok.get("upos") or stanza_tok.get("pos"))

            if camel_pos and stanza_pos and camel_pos != "X" and stanza_pos != "X" and camel_pos != stanza_pos:
                conflicts.append({"word": w, "camel_pos": camel_pos, "stanza_pos": stanza_pos})
                all_conflicts.append(classify_conflict("pos", camel_pos, stanza_pos))

            c_lemma = normalize_lemma_for_compare(camel_ana.get("lemma"))
            s_lemma = normalize_lemma_for_compare(stanza_tok.get("lemma"))
            if c_lemma and s_lemma and not lemma_equivalent(c_lemma, s_lemma):
                all_conflicts.append(classify_conflict("lemma", c_lemma, s_lemma))

    result = {
        "total_words": total,
        "pos_agreement": round(pos_agreement, 2),
        "pos_agreement_pct": f"{round(pos_agreement * 100, 1)}%",
        "pos_precision": pos_precision,
        "pos_recall": pos_recall,
        "pos_f1": pos_f1,
        "lemma_match": round(lemma_normalized, 2) if total else 0,
        "lemma_match_pct": f"{round(lemma_normalized * 100, 1)}%" if total else "0%",
        "lemma_exact_match": round(lemma_exact, 2) if total else 0,
        "lemma_exact_match_pct": f"{round(lemma_exact * 100, 1)}%" if total else "0%",
        "lemma_normalized_match": round(lemma_normalized, 2) if total else 0,
        "lemma_normalized_match_pct": f"{round(lemma_normalized * 100, 1)}%" if total else "0%",
        "segmentation_coverage": round(seg_cov / total, 2) if total else 0,
        "pos_conflicts": conflicts,
        "all_conflicts": all_conflicts,
        "active_tools": active_tools,
        "excluded_tools": sorted(excluded_tools),
        "metrics_note": f"Scores reflect {active_tool_count} active tools only",
    }

    # If all tools are inactive, ensure metrics reflect 0 active participation.
    if active_tool_count == 0:
        result["total_words"] = 0
        result["pos_agreement"] = 0
        result["pos_agreement_pct"] = "0%"
        result["pos_precision"] = 0
        result["pos_recall"] = 0
        result["pos_f1"] = 0
        result["lemma_match"] = 0
        result["lemma_match_pct"] = "0%"
        result["lemma_exact_match"] = 0
        result["lemma_exact_match_pct"] = "0%"
        result["lemma_normalized_match"] = 0
        result["lemma_normalized_match_pct"] = "0%"
        result["segmentation_coverage"] = 0
        result["pos_conflicts"] = []
        result["all_conflicts"] = []

    return result



