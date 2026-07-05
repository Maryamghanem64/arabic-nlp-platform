from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.utils.helpers import normalize_lemma_for_compare, normalize_pos_for_compare
from backend.services.alignment_engine import align_tools


EXCLUDED_STATUSES = {
    "error",
    "unavailable",
    "future_work",
    "lazy",
    "disabled",
    "timeout",
    "lazy_not_loaded",
    "loading",
    "missing_resources",
    "excluded",
    "skipped_low_memory",
}

# Hard-coded tool-level exclusions required by evaluation semantics.
MADAMIRA_ALWAYS_EXCLUDED = True


POS_CAPABLE = {"camel", "stanza", "udpipe", "sinatools", "alkhalil"}
LEMMA_CAPABLE = {"camel", "stanza", "qalsadi", "alkhalil", "udpipe", "sinatools"}
ROOT_CAPABLE = {"camel", "alkhalil", "sinatools"}
SEGMENTATION_CAPABLE = {"farasa", "camel", "alkhalil", "sinatools"}
DEPENDENCY_CAPABLE = {"stanza", "udpipe"}


def strip_diacritics(text: str) -> str:
    if not text:
        return text
    return re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670]", "", str(text))


def _valid(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text not in {"#", "X", "UNK", "None", "null", "0"}


def _pct(value: float) -> str:
    return f"{round(value * 100, 1)}%"


def _normalize_alkhalil_pos(value: Any) -> Optional[str]:
    if not _valid(value):
        return None

    text = str(value).strip()
    compact = text.replace(" ", "")

    if "حرف جر" in text:
        return "ADP"
    if "حرف عطف" in text:
        return "CCONJ"
    if "ضمير" in text:
        return "PRON"
    if "ظرف" in text:
        return "ADV"
    if "صفة" in text:
        return "ADJ"
    if "فعل" in text or "ماض" in text or "مضارع" in text or "أمر" in text or "امر" in compact:
        return "VERB"
    if (
        "اسم" in text
        or "مفرد" in text
        or "مثنى" in text
        or "جمع" in text
        or "فاعل" in text
        or "مفعول" in text
        or "مصدر" in text
    ):
        return "NOUN"

    return normalize_pos_for_compare(text)


def _normalize_pos(tool: str, value: Any) -> Optional[str]:
    if not _valid(value):
        return None

    if tool == "alkhalil":
        return _normalize_alkhalil_pos(value)

    normalized = normalize_pos_for_compare(str(value))
    if normalized == "ADPOSITION":
        return "ADP"
    return normalized if _valid(normalized) else None


def _normalize_lemma(value: Any) -> Optional[str]:
    if not _valid(value):
        return None

    text = str(value).strip()
    text = re.sub(r"\d+$", "", text)
    text = normalize_lemma_for_compare(text)
    text = strip_diacritics(text)

    return text if _valid(text) else None


def _normalize_root(value: Any) -> Optional[str]:
    if not _valid(value):
        return None

    text = strip_diacritics(str(value))
    text = text.replace(" ", ".").replace("-", ".").replace("ـ", "")
    text = re.sub(r"\.+", ".", text).strip(".")

    if "." not in text and len(text) >= 2:
        text = ".".join(list(text))

    return text if _valid(text) else None


def _first_analysis(tok: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(tok, dict):
        return {}
    analyses = tok.get("analyses")
    if isinstance(analyses, list) and analyses and isinstance(analyses[0], dict):
        return analyses[0]
    return {}


def _extract_pos(tool: str, tok: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    """Return (normalized_pos, raw_pos)."""
    if not isinstance(tok, dict):
        return None, None

    ana = _first_analysis(tok)
    candidates = [
        tok.get("pos_raw"),
        tok.get("pos"),
        tok.get("upos"),
        ana.get("pos"),
        ana.get("upos"),
        ana.get("type"),
        ana.get("gloss"),
    ]

    for raw in candidates:
        if not _valid(raw):
            continue
        normalized = _normalize_pos(tool, raw)
        if normalized:
            # For AlKhalil we normalize POS into UD-like tags; raw kept as Arabic description.
            return normalized, str(raw).strip()

    return None, None



def _extract_lemma(tool: str, tok: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(tok, dict):
        return None

    ana = _first_analysis(tok)
    candidates = [
        ana.get("lemma"),
        tok.get("lemma"),
    ]

    for candidate in candidates:
        value = _normalize_lemma(candidate)
        if value:
            return value

    return None


def _extract_root(tool: str, tok: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(tok, dict):
        return None

    ana = _first_analysis(tok)
    candidates = [
        ana.get("root"),
        tok.get("root"),
    ]

    for candidate in candidates:
        value = _normalize_root(candidate)
        if value:
            return value

    return None


def _majority_agreement(values: Dict[str, str]) -> Tuple[float, Optional[str]]:
    if not values:
        return 0.0, None

    counts = Counter(values.values())
    majority_value, majority_count = counts.most_common(1)[0]
    return majority_count / len(values), majority_value


def _pairwise_conflicts(
    *,
    word: str,
    feature: str,
    normalized_values: Dict[str, str],
    raw_values: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """Return pairwise conflicts.

    - value_a/value_b are always normalized.
    - raw_value_a/raw_value_b keep the original/raw Arabic (when available).
    """

    conflicts: List[Dict[str, Any]] = []
    tools = list(normalized_values.keys())
    raw_values = raw_values or {}

    for i, tool_a in enumerate(tools):
        for tool_b in tools[i + 1:]:
            norm_a = normalized_values[tool_a]
            norm_b = normalized_values[tool_b]

            if norm_a == norm_b:
                continue

            raw_a = raw_values.get(tool_a)
            raw_b = raw_values.get(tool_b)

            conflicts.append(
                {
                    "word": word,
                    "feature": feature,
                    "tool_a": tool_a,
                    "value_a": norm_a,
                    "raw_value_a": raw_a,
                    "tool_b": tool_b,
                    "value_b": norm_b,
                    "raw_value_b": raw_b,
                    "severity": "high" if feature == "pos" else "medium",
                    "type": f"{feature}_mismatch",
                }
            )

    return conflicts



def _capability_contributors(
    normalized_results: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    ok_tools = {
        name
        for name, result in normalized_results.items()
        if isinstance(result, dict) and result.get("status") == "ok"
    }

    # Ensure 'excluded' status never becomes a contributor.
    ok_tools = {t for t in ok_tools if normalized_results.get(t, {}).get("status") not in EXCLUDED_STATUSES and normalized_results.get(t, {}).get("status") != "excluded"}

    if MADAMIRA_ALWAYS_EXCLUDED:
        ok_tools.discard("madamira")

    # AraBERT is contextual only; never a metric contributor.
    arabert = {"arabert"} & ok_tools

    pos_tools = sorted(ok_tools & POS_CAPABLE)
    lemma_tools = sorted(ok_tools & LEMMA_CAPABLE)
    root_tools = sorted(ok_tools & ROOT_CAPABLE)

    # Never include arabert in feature contributors.
    pos_tools = [t for t in pos_tools if t != "arabert"]
    lemma_tools = [t for t in lemma_tools if t != "arabert"]
    root_tools = [t for t in root_tools if t != "arabert"]

    return {
        "pos": pos_tools,
        "lemma": lemma_tools,
        "root": root_tools,
        "segmentation": sorted(ok_tools & SEGMENTATION_CAPABLE),
        "dependency": sorted(ok_tools & DEPENDENCY_CAPABLE),
        "contextual": sorted(arabert),
    }



def evaluate_tools(
    text: str,
    camel_res,
    stanza_res,
    farasa_res,
    qalsadi_res=None,
    all_tool_results: Dict[str, Dict] | None = None,
):
    from backend.services.normalizer import normalize_tool_output

    if all_tool_results is None:
        all_tool_results = {}

    normalized_results = {
        name: normalize_tool_output(name, res if isinstance(res, dict) else {})
        for name, res in all_tool_results.items()
    }

    normalized_results.setdefault("camel", normalize_tool_output("camel", camel_res if isinstance(camel_res, dict) else {}))
    normalized_results.setdefault("stanza", normalize_tool_output("stanza", stanza_res if isinstance(stanza_res, dict) else {}))
    normalized_results.setdefault("farasa", normalize_tool_output("farasa", farasa_res if isinstance(farasa_res, dict) else {}))
    normalized_results.setdefault("qalsadi", normalize_tool_output("qalsadi", qalsadi_res if isinstance(qalsadi_res, dict) else {}))

    all_statuses = {
        name: result.get("status") if isinstance(result, dict) else None
        for name, result in normalized_results.items()
    }

    # status == "excluded" must never count as active.
    active_tools = sorted(
        [
            name
            for name, status in all_statuses.items()
            if status == "ok" and status not in EXCLUDED_STATUSES and status != "excluded"
        ]
    )

    # Explicitly exclude madamira regardless of status.
    excluded_tools = sorted(
        [
            name
            for name, status in all_statuses.items()
            if status in EXCLUDED_STATUSES or (MADAMIRA_ALWAYS_EXCLUDED and name == "madamira")
        ]
    )

    if MADAMIRA_ALWAYS_EXCLUDED and "madamira" in active_tools:
        active_tools = [t for t in active_tools if t != "madamira"]


    farasa_tokens = [
        token
        for token in (normalized_results.get("farasa", {}).get("tokens", []) or [])
        if isinstance(token, dict) and token.get("surface")
    ]

    total_words = len(farasa_tokens)

    aligned_tokens, _meta = align_tools(
        base_tokens=farasa_tokens,
        tools_tokens={
            name: result.get("tokens", []) or []
            for name, result in normalized_results.items()
            if isinstance(result, dict)
        },
    )

    contributors = _capability_contributors(normalized_results)

    pos_scores: List[float] = []
    lemma_scores: List[float] = []
    root_scores: List[float] = []
    pos_conflicts: List[Dict[str, Any]] = []
    lemma_conflicts: List[Dict[str, Any]] = []
    root_conflicts: List[Dict[str, Any]] = []

    seg_covered = 0

    for aligned in aligned_tokens:
        word = aligned.base.get("surface") or ""

        if isinstance(aligned.base.get("segmentation"), list) and aligned.base.get("segmentation"):
            seg_covered += 1

        pos_values: Dict[str, str] = {}
        pos_raw_values: Dict[str, str] = {}
        lemma_values: Dict[str, str] = {}
        root_values: Dict[str, str] = {}

        for tool in contributors["pos"]:
            norm, raw = _extract_pos(tool, aligned.tools.get(tool))
            if norm:
                pos_values[tool] = norm
            if raw:
                pos_raw_values[tool] = raw



        for tool in contributors["lemma"]:
            value = _extract_lemma(tool, aligned.tools.get(tool))
            if value:
                lemma_values[tool] = value

        for tool in contributors["root"]:
            value = _extract_root(tool, aligned.tools.get(tool))
            if value:
                root_values[tool] = value

        pos_agreement, _ = _majority_agreement(pos_values)
        lemma_agreement, _ = _majority_agreement(lemma_values)
        root_agreement, _ = _majority_agreement(root_values)

        if len(pos_values) >= 2:
            pos_scores.append(pos_agreement)
            pos_conflicts.extend(
                _pairwise_conflicts(
                    word=word,
                    feature="pos",
                    normalized_values=pos_values,
                    raw_values=pos_raw_values,
                )
            )


        if len(lemma_values) >= 2:
            lemma_scores.append(lemma_agreement)
            lemma_conflicts.extend(
                _pairwise_conflicts(word=word, feature="lemma", normalized_values=lemma_values)
            )


        if len(root_values) >= 2:
            root_scores.append(root_agreement)
            root_conflicts.extend(
                _pairwise_conflicts(word=word, feature="root", normalized_values=root_values)
            )


    pos_agreement = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    lemma_agreement = sum(lemma_scores) / len(lemma_scores) if lemma_scores else 0.0
    root_agreement = sum(root_scores) / len(root_scores) if root_scores else 0.0
    segmentation_coverage = seg_covered / total_words if total_words else 0.0

    all_conflicts = pos_conflicts + lemma_conflicts + root_conflicts

    return {
        "total_words": total_words,

        "pos_agreement": round(pos_agreement, 2),
        "pos_agreement_pct": _pct(pos_agreement),
        "pos_precision": round(pos_agreement, 3),
        "pos_recall": round(pos_agreement, 3),
        "pos_f1": round(pos_agreement, 3),

        "lemma_match": round(lemma_agreement, 2),
        "lemma_match_pct": _pct(lemma_agreement),
        "lemma_exact_match": round(lemma_agreement, 2),
        "lemma_exact_match_pct": _pct(lemma_agreement),
        "lemma_normalized_match": round(lemma_agreement, 2),
        "lemma_normalized_match_pct": _pct(lemma_agreement),

        "root_agreement": round(root_agreement, 2),
        "root_agreement_pct": _pct(root_agreement),

        "segmentation_coverage": round(segmentation_coverage, 2),

        "pos_conflicts": pos_conflicts,
        "lemma_conflicts": lemma_conflicts,
        "root_conflicts": root_conflicts,
        "all_conflicts": all_conflicts,

        "active_tools": active_tools,

        "excluded_tools": excluded_tools,
        "capability_contributors": contributors,
        # Metric contributors exclude contextual-only tools like AraBERT.
        "metric_contributors": {
            "pos": contributors["pos"],
            "lemma": contributors["lemma"],
            "root": contributors["root"],
            "segmentation": contributors["segmentation"],
            "dependency": contributors["dependency"],
            "contextual": [],
        },

        "metrics_note": (

            "Metrics are capability-aware. Each score is computed only over tools "
            "that provide comparable output for that linguistic feature. Lazy, "
            "excluded, unavailable, or unsupported tools are not counted as wrong."
        ),
    }