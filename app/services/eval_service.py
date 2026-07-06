from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.utils.helpers import (
    normalize_lemma_for_compare,
    normalize_pos_for_compare,
)
from backend.services.alignment_engine import align_tools
from backend.services.normalizer import (
    extract_alkhalil_canonical_pos,
    normalize_tool_output,
)


# ============================================================
# CAPABILITY-AWARE EVALUATION CONFIGURATION
# ============================================================

EXCLUDED_STATUSES = {
    "timeout",
    "unavailable",
    "future_work",
    "lazy",
    "disabled",
    "lazy_not_loaded",
    "loading",
    "missing_resources",
    "excluded",
    "skipped_low_memory",
    # Note: we intentionally do NOT treat general "error" as excluded globally.
    # Fix-1 handles Farasa timeout/error degradation explicitly.
}

FARASA_DEGRADED_STATUSES = {
    "timeout",
    "error",
    "unavailable",
    "missing_resources",
    "lazy_not_loaded",
    "loading",
}


# Fix-1: For evaluation scoring we explicitly exclude Farasa failure states
# by treating them as degraded evidence rather than “wrong output”.
FARASA_DEGRADED_NOTE = (
    "Farasa segmentation unavailable for this run; evaluation continued with available tools."
)




POS_CAPABLE = {
    "camel",
    "stanza",
    "udpipe",
    "sinatools",
    "alkhalil",
}

LEMMA_CAPABLE = {
    "camel",
    "stanza",
    "qalsadi",
    "alkhalil",
    "udpipe",
    "sinatools",
}

ROOT_CAPABLE = {
    "camel",
    "alkhalil",
    "sinatools",
}

SEGMENTATION_CAPABLE = {
    "farasa",
    "camel",
    "alkhalil",
    "sinatools",
}

DEPENDENCY_CAPABLE = {
    "stanza",
    "udpipe",
}

FUNCTIONAL_ROOT_POS = {
    "ADP",
    "PART",
    "CCONJ",
    "SCONJ",
    "DET",
}

POS_DECISION_WEIGHTS = {
    "stanza": 1.00,
    "udpipe": 0.90,
    "camel": 0.82,
    "sinatools": 0.72,
    "alkhalil": 0.55,
}


# ============================================================
# DYNAMIC ALIGNMENT BASE PRIORITY
# ============================================================

ALIGNMENT_BASE_PRIORITY = (
    "farasa",
    "camel",
    "stanza",
    "udpipe",
    "sinatools",
    "alkhalil",
    "qalsadi",
)


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def strip_diacritics(text: str) -> str:
    if not text:
        return text

    return re.sub(
        r"[\u0610-\u061a\u064b-\u065f\u0670]",
        "",
        str(text),
    )


def _valid(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, (list, dict, tuple, set)):
        return False

    text = str(value).strip()

    return (
        bool(text)
        and text not in {
            "#",
            "X",
            "UNK",
            "None",
            "null",
            "0",
        }
    )


def _pct(value: float) -> str:
    return f"{round(value * 100, 1)}%"


def _normalize_pos(
    tool: str,
    value: Any,
) -> Optional[str]:

    if not _valid(value):
        return None

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

    text = (
        text
        .replace(" ", ".")
        .replace("-", ".")
        .replace("ـ", "")
    )

    text = re.sub(r"\.+", ".", text).strip(".")

    if "." not in text and len(text) >= 2:
        text = ".".join(list(text))

    return text if _valid(text) else None


# ============================================================
# TOKEN EXTRACTION HELPERS
# ============================================================

def _first_analysis(
    tok: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    if not isinstance(tok, dict):
        return {}

    analyses = tok.get("analyses")

    if (
        isinstance(analyses, list)
        and analyses
        and isinstance(analyses[0], dict)
    ):
        return analyses[0]

    return {}


def _extract_pos(
    tool: str,
    tok: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:

    if not isinstance(tok, dict):
        return None, None

    if tool == "alkhalil":
        return extract_alkhalil_canonical_pos(
            tok,
            context_pos_votes=None,
        )

    ana = _first_analysis(tok)

    candidates = (
        tok.get("pos_raw"),
        tok.get("pos"),
        tok.get("upos"),
        ana.get("pos"),
        ana.get("upos"),
        ana.get("type"),
    )

    for raw in candidates:
        if not _valid(raw):
            continue

        normalized = _normalize_pos(tool, raw)

        if normalized:
            return normalized, str(raw).strip()

    return None, None


def _extract_lemma(
    tool: str,
    tok: Optional[Dict[str, Any]],
) -> Optional[str]:

    if not isinstance(tok, dict):
        return None

    ana = _first_analysis(tok)

    for raw in (
        ana.get("lemma"),
        tok.get("lemma"),
    ):
        value = _normalize_lemma(raw)

        if value:
            return value

    return None


def _extract_root(
    tool: str,
    tok: Optional[Dict[str, Any]],
) -> Optional[str]:

    if not isinstance(tok, dict):
        return None

    ana = _first_analysis(tok)

    for raw in (
        ana.get("root"),
        tok.get("root"),
    ):
        value = _normalize_root(raw)

        if value:
            return value

    return None


# ============================================================
# TOOL RESULT HELPERS
# ============================================================

def _tool_tokens(
    result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not isinstance(result, dict):
        return []

    tokens = result.get("tokens")

    if not isinstance(tokens, list):
        return []

    return [
        token
        for token in tokens
        if isinstance(token, dict)
        and str(token.get("surface") or "").strip()
    ]


def _tool_is_available(
    result: Optional[Dict[str, Any]],
) -> bool:

    if not isinstance(result, dict):
        return False

    status = str(
        result.get("status") or ""
    ).strip().lower()

    if status in EXCLUDED_STATUSES:
        return False

    return status == "ok"


# ============================================================
# DYNAMIC ALIGNMENT BASE SELECTION
# ============================================================

def _select_alignment_base(
    normalized_results: Dict[str, Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:

    for tool_name in ALIGNMENT_BASE_PRIORITY:

        result = normalized_results.get(tool_name, {})

        if not _tool_is_available(result):
            continue

        tokens = _tool_tokens(result)

        if tokens:
            return tool_name, tokens

    return "none", []


# ============================================================
# AGREEMENT HELPERS
# ============================================================

def _majority_agreement(
    values: Dict[str, str],
) -> Tuple[float, Optional[str]]:

    if not values:
        return 0.0, None

    majority_value, majority_count = Counter(
        values.values()
    ).most_common(1)[0]

    return (
        majority_count / len(values),
        majority_value,
    )


def _clear_functional_pos_decision(
    values: Dict[str, str],
) -> Optional[str]:

    if not values:
        return None

    weighted: Dict[str, float] = {}
    counted: Counter[str] = Counter()

    for tool, value in values.items():
        normalized = (
            str(value).strip().upper()
            if value is not None
            else ""
        )

        if not normalized:
            continue

        counted[normalized] += 1
        weighted[normalized] = (
            weighted.get(normalized, 0.0)
            + POS_DECISION_WEIGHTS.get(tool, 0.50)
        )

    if not weighted:
        return None

    weighted_ranked = sorted(
        weighted.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    weighted_pos, weighted_score = weighted_ranked[0]
    weighted_second = (
        weighted_ranked[1][1]
        if len(weighted_ranked) > 1
        else 0.0
    )

    if (
        weighted_pos in FUNCTIONAL_ROOT_POS
        and weighted_score > weighted_second
    ):
        return weighted_pos

    counted_ranked = counted.most_common()

    if not counted_ranked:
        return None

    majority_pos, majority_count = counted_ranked[0]
    majority_second = (
        counted_ranked[1][1]
        if len(counted_ranked) > 1
        else 0
    )

    if (
        majority_pos in FUNCTIONAL_ROOT_POS
        and majority_count > majority_second
    ):
        return majority_pos

    return None


def _pairwise_conflicts(
    *,
    word: str,
    feature: str,
    normalized_values: Dict[str, str],
    raw_values: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:

    conflicts: List[Dict[str, Any]] = []

    raw_values = raw_values or {}

    tools = list(normalized_values.keys())

    for i, tool_a in enumerate(tools):

        for tool_b in tools[i + 1:]:

            value_a = normalized_values[tool_a]

            value_b = normalized_values[tool_b]

            if value_a == value_b:
                continue

            conflicts.append(
                {
                    "word": word,
                    "feature": feature,
                    "tool_a": tool_a,
                    "value_a": value_a,
                    "raw_value_a": raw_values.get(tool_a),
                    "tool_b": tool_b,
                    "value_b": value_b,
                    "raw_value_b": raw_values.get(tool_b),
                    "severity": (
                        "high"
                        if feature == "pos"
                        else "medium"
                    ),
                    "type": f"{feature}_mismatch",
                }
            )

    return conflicts


# ============================================================
# CAPABILITY CONTRIBUTORS
# ============================================================

def _capability_contributors(
    normalized_results: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:

    ok_tools = {
        name
        for name, result in normalized_results.items()
        if _tool_is_available(result)
    }

    ok_tools.discard("madamira")

    contextual = sorted(
        {"arabert"} & ok_tools
    )

    return {
        "pos": sorted(
            (ok_tools & POS_CAPABLE) - {"arabert"}
        ),
        "lemma": sorted(
            (ok_tools & LEMMA_CAPABLE) - {"arabert"}
        ),
        "root": sorted(
            (ok_tools & ROOT_CAPABLE) - {"arabert"}
        ),
        "segmentation": sorted(
            ok_tools & SEGMENTATION_CAPABLE
        ),
        "dependency": sorted(
            ok_tools & DEPENDENCY_CAPABLE
        ),
        "contextual": contextual,
    }


# ============================================================
# SEGMENTATION COVERAGE
# ============================================================

def _has_segmentation_evidence(
    aligned: Any,
    segmentation_tools: List[str],
) -> bool:

    for tool in segmentation_tools:

        tok = aligned.tools.get(tool)

        if not isinstance(tok, dict):
            continue

        segmentation = tok.get("segmentation")

        if (
            isinstance(segmentation, list)
            and segmentation
        ):
            return True

    return False


# ============================================================
# MAIN CAPABILITY-AWARE EVALUATION
# ============================================================

def evaluate_tools(
    text: str,
    camel_res,
    stanza_res,
    farasa_res,
    qalsadi_res=None,
    all_tool_results: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
):

    if all_tool_results is None:
        all_tool_results = {}

    # --------------------------------------------------------
    # Normalize every registered tool result
    # --------------------------------------------------------

    normalized_results: Dict[
        str,
        Dict[str, Any],
    ] = {
        name: normalize_tool_output(
            name,
            result if isinstance(result, dict) else {},
        )
        for name, result in all_tool_results.items()
    }

    normalized_results.setdefault(
        "camel",
        normalize_tool_output(
            "camel",
            camel_res
            if isinstance(camel_res, dict)
            else {},
        ),
    )

    normalized_results.setdefault(
        "stanza",
        normalize_tool_output(
            "stanza",
            stanza_res
            if isinstance(stanza_res, dict)
            else {},
        ),
    )

    normalized_results.setdefault(
        "farasa",
        normalize_tool_output(
            "farasa",
            farasa_res
            if isinstance(farasa_res, dict)
            else {},
        ),
    )

    normalized_results.setdefault(
        "qalsadi",
        normalize_tool_output(
            "qalsadi",
            qalsadi_res
            if isinstance(qalsadi_res, dict)
            else {},
        ),
    )

    # --------------------------------------------------------
    # Tool status classification
    # --------------------------------------------------------

    all_statuses = {
        name: (
            result.get("status")
            if isinstance(result, dict)
            else None
        )
        for name, result in normalized_results.items()
    }

    active_tools = sorted(
        [
            name
            for name, result in normalized_results.items()
            if _tool_is_available(result)
            and name != "madamira"
        ]
    )

    farasa_degraded = str(
        all_statuses.get("farasa", "") or ""
    ).strip().lower() in FARASA_DEGRADED_STATUSES

    # Fix-1: If Farasa is degraded (timeout/error/unavailable), treat it as
    # excluded evidence for scoring while keeping it visible in UI.
    # We ensure it does not enter active_tools and does not count towards coverage.
    farasa_note = None

    if farasa_degraded:
        if "farasa" in active_tools:
            active_tools = [t for t in active_tools if t != "farasa"]
        farasa_note = FARASA_DEGRADED_NOTE



    excluded_tools = sorted(
        [
            name
            for name, status in all_statuses.items()
            if (
                str(status or "").lower()
                in EXCLUDED_STATUSES
                or name == "madamira"
                or (name == "farasa" and farasa_degraded)
            )
        ]
    )



    # --------------------------------------------------------
    # Dynamic alignment base
    # --------------------------------------------------------

    alignment_base_tool, base_tokens = (
        _select_alignment_base(
            normalized_results
        )
    )

    total_words = len(base_tokens)

    tools_tokens = {
        name: _tool_tokens(result)
        for name, result in normalized_results.items()
        if isinstance(result, dict)
        and _tool_is_available(result)
    }

    if base_tokens:

        aligned_tokens, alignment_meta = align_tools(
            base_tokens=base_tokens,
            tools_tokens=tools_tokens,
        )

    else:

        aligned_tokens = []

        alignment_meta = {
            "matched_counts": {},
            "base_count": 0,
        }

    # --------------------------------------------------------
    # Capability contributors
    # --------------------------------------------------------

    contributors = _capability_contributors(
        normalized_results
    )

    # Fix-1: Do not use Farasa evidence for segmentation coverage when Farasa is degraded.
    if farasa_degraded:
        if "farasa" in contributors.get("segmentation", []):
            contributors["segmentation"] = [
                t for t in contributors["segmentation"] if t != "farasa"
            ]


    # --------------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------------

    pos_scores: List[float] = []

    lemma_scores: List[float] = []

    root_scores: List[float] = []

    pos_conflicts: List[Dict[str, Any]] = []

    lemma_conflicts: List[Dict[str, Any]] = []

    root_conflicts: List[Dict[str, Any]] = []

    seg_covered = 0

    pos_evaluated_tokens = 0

    lemma_evaluated_tokens = 0

    root_evaluated_tokens = 0

    segmentation_evaluated_tokens = 0

    metric_contributor_sets: Dict[str, set[str]] = {
        "pos": set(),
        "lemma": set(),
        "root": set(),
        "segmentation": set(),
        "dependency": set(),
        "contextual": set(),
    }

    # --------------------------------------------------------
    # Evaluate aligned tokens
    # --------------------------------------------------------

    for aligned in aligned_tokens:

        word = str(
            aligned.base.get("surface") or ""
        )

        # ----------------------------------------------------
        # Segmentation evidence
        # ----------------------------------------------------

        if contributors["segmentation"]:

            segmentation_evaluated_tokens += 1

            if _has_segmentation_evidence(
                aligned,
                contributors["segmentation"],
            ):
                seg_covered += 1

            for tool in contributors["segmentation"]:

                tok = aligned.tools.get(tool)

                if not isinstance(tok, dict):
                    continue

                segmentation = tok.get("segmentation")

                if (
                    isinstance(segmentation, list)
                    and segmentation
                ):
                    metric_contributor_sets[
                        "segmentation"
                    ].add(tool)

        # ----------------------------------------------------
        # POS values
        # ----------------------------------------------------

        pos_values: Dict[str, str] = {}

        pos_raw_values: Dict[str, str] = {}

        other_votes: Dict[str, str] = {}

        for tool in contributors["pos"]:

            if tool == "alkhalil":
                continue

            norm, _raw = _extract_pos(
                tool,
                aligned.tools.get(tool),
            )

            if norm:
                other_votes[tool] = norm

        for tool in contributors["pos"]:

            if tool == "alkhalil":

                norm, raw = (
                    extract_alkhalil_canonical_pos(
                        aligned.tools.get(tool),
                        context_pos_votes=other_votes,
                    )
                )

            else:

                norm, raw = _extract_pos(
                    tool,
                    aligned.tools.get(tool),
                )

            if norm:
                pos_values[tool] = norm

            if raw:
                pos_raw_values[tool] = raw

        # ----------------------------------------------------
        # Lemma values
        # ----------------------------------------------------

        lemma_values: Dict[str, str] = {}

        for tool in contributors["lemma"]:

            value = _extract_lemma(
                tool,
                aligned.tools.get(tool),
            )

            if value:
                lemma_values[tool] = value

        # ----------------------------------------------------
        # Root values
        # ----------------------------------------------------

        root_values: Dict[str, str] = {}

        for tool in contributors["root"]:

            value = _extract_root(
                tool,
                aligned.tools.get(tool),
            )

            if value:
                root_values[tool] = value

        # ----------------------------------------------------
        # POS agreement
        # ----------------------------------------------------

        if len(pos_values) >= 2:

            score, _majority = _majority_agreement(
                pos_values
            )

            pos_scores.append(score)

            pos_evaluated_tokens += 1

            metric_contributor_sets["pos"].update(
                pos_values.keys()
            )

            token_tools = set(pos_values.keys())
            sinatools_involved = "sinatools" in token_tools

            pair_conflicts = _pairwise_conflicts(
                word=word,
                feature="pos",
                normalized_values=pos_values,
                raw_values=pos_raw_values,
            )

            # Fix-2: SinaTools lexical POS disagreement should be downgraded
            # when majority consensus among high-capability tools agrees.
            if sinatools_involved and pair_conflicts:
                # Count how many other tools agree with the majority value.
                consensus_ok = (
                    _majority_agreement(pos_values)[0] >= 0.67
                )
                if consensus_ok:
                    for c in pair_conflicts:
                        if c.get("feature") == "pos" and (
                            c.get("tool_a") == "sinatools" or c.get("tool_b") == "sinatools"
                        ):
                            c["severity"] = "medium"
                            c["note"] = (
                                "Single-tool lexical POS disagreement; consensus selected by capability-aware fusion."
                            )

            pos_conflicts.extend(pair_conflicts)


        # ----------------------------------------------------
        # Lemma agreement
        # ----------------------------------------------------

        if len(lemma_values) >= 2:

            score, _majority = _majority_agreement(
                lemma_values
            )

            lemma_scores.append(score)

            lemma_evaluated_tokens += 1

            metric_contributor_sets["lemma"].update(
                lemma_values.keys()
            )

            lemma_conflicts.extend(
                _pairwise_conflicts(
                    word=word,
                    feature="lemma",
                    normalized_values=lemma_values,
                )
            )

        # ----------------------------------------------------
        # Root agreement
        # ----------------------------------------------------

        functional_pos = _clear_functional_pos_decision(
            pos_values
        )

        if (
            len(root_values) >= 2
            and functional_pos is None
        ):

            score, _majority = _majority_agreement(
                root_values
            )

            root_scores.append(score)

            root_evaluated_tokens += 1

            metric_contributor_sets["root"].update(
                root_values.keys()
            )

            root_conflicts.extend(
                _pairwise_conflicts(
                    word=word,
                    feature="root",
                    normalized_values=root_values,
                )
            )

    # --------------------------------------------------------
    # Final metric calculation
    # --------------------------------------------------------

    pos_agreement = (
        sum(pos_scores) / len(pos_scores)
        if pos_scores
        else 0.0
    )

    lemma_agreement = (
        sum(lemma_scores) / len(lemma_scores)
        if lemma_scores
        else 0.0
    )

    root_agreement = (
        sum(root_scores) / len(root_scores)
        if root_scores
        else 0.0
    )

    segmentation_coverage = (
        seg_covered / segmentation_evaluated_tokens
        if segmentation_evaluated_tokens
        else 0.0
    )

    all_conflicts = (
        pos_conflicts
        + lemma_conflicts
        + root_conflicts
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "total_words": total_words,

        "pos_agreement": round(
            pos_agreement,
            2,
        ),

        "pos_agreement_pct": _pct(
            pos_agreement
        ),

        # Backward-compatible fields.
        # These remain agreement-derived proxy values.
        "pos_precision": round(
            pos_agreement,
            3,
        ),

        "pos_recall": round(
            pos_agreement,
            3,
        ),

        "pos_f1": round(
            pos_agreement,
            3,
        ),

        "lemma_match": round(
            lemma_agreement,
            2,
        ),

        "lemma_match_pct": _pct(
            lemma_agreement
        ),

        "lemma_exact_match": round(
            lemma_agreement,
            2,
        ),

        "lemma_exact_match_pct": _pct(
            lemma_agreement
        ),

        "lemma_normalized_match": round(
            lemma_agreement,
            2,
        ),

        "lemma_normalized_match_pct": _pct(
            lemma_agreement
        ),

        "root_agreement": round(
            root_agreement,
            2,
        ),

        "root_agreement_pct": _pct(
            root_agreement
        ),

        "segmentation_coverage": round(
            segmentation_coverage,
            2,
        ),

        "pos_conflicts": pos_conflicts,

        "lemma_conflicts": lemma_conflicts,

        "root_conflicts": root_conflicts,

        "all_conflicts": all_conflicts,

        "active_tools": active_tools,

        "excluded_tools": excluded_tools,

        "alignment_base_tool": (
            alignment_base_tool
        ),

        "alignment_meta": alignment_meta,

        "evaluated_token_counts": {
            "pos": pos_evaluated_tokens,
            "lemma": lemma_evaluated_tokens,
            "root": root_evaluated_tokens,
            "segmentation": (
                segmentation_evaluated_tokens
            ),
        },

        "capability_contributors": contributors,

        "metric_contributors": {
            "pos": sorted(
                metric_contributor_sets["pos"]
            ),
            "lemma": sorted(
                metric_contributor_sets["lemma"]
            ),
            "root": sorted(
                metric_contributor_sets["root"]
            ),
            "segmentation": sorted(
                metric_contributor_sets[
                    "segmentation"
                ]
            ),
            "dependency": contributors[
                "dependency"
            ],
            "contextual": contributors[
                "contextual"
            ],
        },

        "metrics_note": (
            "Metrics are capability-aware. "
            "Each score is computed only over tools "
            "that support the evaluated linguistic "
            "feature and produced comparable values. "
            "Lazy, excluded, unavailable, timeout, "
            "or unsupported tools are not counted as "
            "wrong. Alignment uses a dynamic available "
            "token base and does not structurally depend "
            "on Farasa."
        ),
        "degraded_notes": (
            [farasa_note] if farasa_note else []
        ),
    }
