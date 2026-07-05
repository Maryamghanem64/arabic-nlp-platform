from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from backend.services.normalizer import (
    extract_alkhalil_canonical_pos,
    normalize_alkhalil_pos,
)
from app.utils.helpers import normalize_pos_for_compare


INVALID = {
    None,
    "",
    "#",
    "X",
    "x",
    "UNK",
    "unknown",
    "None",
    "null",
}


FUNCTIONAL_POS = {
    "ADP",
    "PART",
    "CCONJ",
    "SCONJ",
}


IMPORTANT_EXPERT_FEATURES = {
    "lemma",
    "pos",
    "dependency",
}


FEATURE_WEIGHTS = {
    "segmentation": {
        "farasa": 1.00,
        "camel": 0.35,
        "sinatools": 0.25,
        "alkhalil": 0.20,
    },
    "lemma": {
        "camel": 1.00,
        "sinatools": 0.85,
        "qalsadi": 0.75,
        "stanza": 0.55,
        "udpipe": 0.50,
        "alkhalil": 0.45,
    },
    "root": {
        "camel": 1.00,
        "sinatools": 0.85,
        "alkhalil": 0.75,
        "qalsadi": 0.35,
    },
    "pos": {
        "stanza": 1.00,
        "udpipe": 0.90,
        "camel": 0.82,
        "sinatools": 0.72,
        "alkhalil": 0.55,
    },
    "dependency": {
        "udpipe": 1.00,
        "stanza": 0.95,
    },
    "morphology": {
        "camel": 1.00,
        "alkhalil": 0.70,
        "sinatools": 0.55,
    },
}


def _valid(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, list):
        return len(value) > 0

    if isinstance(value, dict):
        for item in value.values():
            if item is None:
                continue

            if isinstance(item, (list, dict)):
                if _valid(item):
                    return True
                continue

            if str(item).strip() not in INVALID:
                return True

        return False

    return str(value).strip() not in INVALID


def _strip_diacritics(value: Any) -> str:
    text = str(value or "")
    text = re.sub(
        r"[\u0610-\u061a\u064b-\u065f\u0670]",
        "",
        text,
    )
    return text.replace("ـ", "").strip()


def _normalize_lemma(value: Any) -> Optional[str]:
    if not _valid(value):
        return None

    text = _strip_diacritics(value)
    text = re.sub(r"\d+$", "", text)

    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
    )

    return text if text else None


def _normalize_root(value: Any) -> Optional[str]:
    if not _valid(value):
        return None

    text = _strip_diacritics(value)
    text = re.sub(r"[.\s\-ـ]+", "", text)

    if not text:
        return None

    return ".".join(list(text))


def _normalize_pos(
    tool: str,
    value: Any,
) -> Optional[str]:
    if not _valid(value):
        return None

    if tool == "alkhalil":
        mapped = normalize_alkhalil_pos(value)
    else:
        mapped = normalize_pos_for_compare(str(value))

    if mapped == "ADPOSITION":
        mapped = "ADP"

    if mapped == "PROPN":
        mapped = "NOUN"

    return mapped if _valid(mapped) else None


def _first_analysis(
    token: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(token, dict):
        return {}

    analyses = token.get("analyses")

    if (
        isinstance(analyses, list)
        and analyses
        and isinstance(analyses[0], dict)
    ):
        return analyses[0]

    return {}


def _token_value(
    token: Optional[Dict[str, Any]],
    key: str,
) -> Any:
    if not isinstance(token, dict):
        return None

    value = token.get(key)

    if _valid(value):
        return value

    analysis = _first_analysis(token)

    return analysis.get(key)


def _dependency_value(
    token: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(token, dict):
        return None

    dependency = token.get("dependency")

    if not isinstance(dependency, dict):
        return None

    for key in ("head", "head_text", "deprel"):
        value = dependency.get(key)

        if value is not None and str(value).strip() not in INVALID:
            return dependency

    return None


def _case_value(
    token: Optional[Dict[str, Any]],
) -> Any:
    if not isinstance(token, dict):
        return None

    features = token.get("features")

    if not isinstance(features, dict):
        features = {}

    return token.get("case") or features.get("case")


def _collect_pos_votes(
    tools: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, str]:
    votes: Dict[str, str] = {}

    for tool in (
        "camel",
        "stanza",
        "udpipe",
        "sinatools",
        "alkhalil",
    ):
        token = tools.get(tool)

        raw = (
            _token_value(token, "pos")
            or _token_value(token, "upos")
        )

        normalized = _normalize_pos(tool, raw)

        if normalized:
            votes[tool] = normalized

    return votes


def _extract_feature_value(
    feature: str,
    tool: str,
    token: Optional[Dict[str, Any]],
    *,
    context_pos_votes: Optional[Dict[str, str]] = None,
) -> Any:
    if not isinstance(token, dict):
        return None

    if feature == "segmentation":
        segmentation = token.get("segmentation")

        return segmentation if _valid(segmentation) else None

    if feature == "lemma":
        return _token_value(token, "lemma")

    if feature == "root":
        return _token_value(token, "root")

    if feature == "pos":
        if tool == "alkhalil":
            canonical, _raw = extract_alkhalil_canonical_pos(
                token,
                context_pos_votes=context_pos_votes or {},
            )
            return canonical

        return (
            _token_value(token, "pos")
            or _token_value(token, "upos")
        )

    if feature == "dependency":
        return _dependency_value(token)

    if feature == "case":
        return _case_value(token)

    return None


def _normalize_feature_value(
    feature: str,
    tool: str,
    value: Any,
) -> Any:
    if feature == "lemma":
        return _normalize_lemma(value)

    if feature == "root":
        return _normalize_root(value)

    if feature == "pos":
        return _normalize_pos(tool, value)

    if feature == "segmentation":
        if isinstance(value, list) and value:
            return value
        return None

    if feature == "dependency":
        if isinstance(value, dict) and value:
            return value
        return None

    if feature == "case":
        if _valid(value):
            return str(value).strip().lower()
        return None

    return value


def _confidence_level(score: float) -> str:
    if score >= 0.75:
        return "high"

    if score >= 0.55:
        return "medium"

    return "low"


def _value_key(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(map(str, value))

    if isinstance(value, dict):
        return "|".join(
            f"{key}:{value}"
            for key, value in sorted(value.items())
        )

    return str(value)


def _candidate_payload(
    scored: List[
        Tuple[
            float,
            str,
            List[Dict[str, Any]],
        ]
    ],
) -> List[Dict[str, Any]]:
    return [
        {
            "value": items[0]["value"],
            "score": round(score, 3),
            "tools": [
                item["tool"]
                for item in items
            ],
        }
        for score, _key, items in scored
    ]


def _choose_by_weighted_consensus(
    *,
    feature: str,
    tools: Dict[str, Optional[Dict[str, Any]]],
    fallback_value: Any = None,
    fallback_source: Optional[str] = None,
) -> Dict[str, Any]:
    weights = FEATURE_WEIGHTS.get(feature, {})

    context_pos_votes = (
        _collect_pos_votes(tools)
        if feature == "pos"
        else {}
    )

    candidates: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for tool, weight in weights.items():
        token = tools.get(tool)

        raw_value = _extract_feature_value(
            feature,
            tool,
            token,
            context_pos_votes=context_pos_votes,
        )

        normalized_value = _normalize_feature_value(
            feature,
            tool,
            raw_value,
        )

        if not _valid(normalized_value):
            continue

        key = _value_key(normalized_value)

        candidates[key].append(
            {
                "tool": tool,
                "value": normalized_value,
                "raw_value": raw_value,
                "weight": weight,
            }
        )

    if not candidates and _valid(fallback_value):
        return {
            "value": fallback_value,
            "expert": f"{feature}_expert",
            "primary_source": (
                fallback_source
                or "current_fusion"
            ),
            "supporting_tools": (
                [fallback_source]
                if fallback_source
                else []
            ),
            "disagreeing_tools": [],
            "strategy": "fallback_to_current_fusion",
            "confidence_score": 0.45,
            "confidence_level": "low",
            "ambiguity": False,
            "score_margin": None,
            "candidates": [],
        }

    if not candidates:
        return {
            "value": None,
            "expert": f"{feature}_expert",
            "primary_source": None,
            "supporting_tools": [],
            "disagreeing_tools": [],
            "strategy": "no_capable_evidence",
            "confidence_score": 0.0,
            "confidence_level": "low",
            "ambiguity": False,
            "score_margin": None,
            "candidates": [],
        }

    scored: List[
        Tuple[
            float,
            str,
            List[Dict[str, Any]],
        ]
    ] = []

    for key, items in candidates.items():
        score = sum(
            item["weight"]
            for item in items
        )

        scored.append(
            (
                score,
                key,
                items,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, _best_key, best_items = scored[0]

    second_score = (
        scored[1][0]
        if len(scored) > 1
        else 0.0
    )

    total_score = sum(
        score
        for score, _, _ in scored
    )

    raw_confidence = (
        best_score / total_score
        if total_score
        else 0.0
    )

    score_margin = (
        (best_score - second_score) / total_score
        if total_score
        else 0.0
    )

    ambiguity = (
        len(scored) > 1
        and score_margin < 0.15
    )

    confidence = raw_confidence

    if ambiguity:
        confidence = min(
            confidence,
            0.54,
        )

    best_value = best_items[0]["value"]

    primary_source = sorted(
        best_items,
        key=lambda item: item["weight"],
        reverse=True,
    )[0]["tool"]

    supporting_tools = [
        item["tool"]
        for item in best_items
    ]

    disagreeing_tools: List[str] = []

    for _score, _key, items in scored[1:]:
        disagreeing_tools.extend(
            item["tool"]
            for item in items
        )

    strategy = "capability_weighted_consensus"

    if ambiguity:
        strategy = (
            "capability_weighted_consensus_with_ambiguity"
        )

    return {
        "value": best_value,
        "expert": f"{feature}_expert",
        "primary_source": primary_source,
        "supporting_tools": supporting_tools,
        "disagreeing_tools": disagreeing_tools,
        "strategy": strategy,
        "confidence_score": round(
            confidence,
            3,
        ),
        "confidence_level": _confidence_level(
            confidence,
        ),
        "ambiguity": ambiguity,
        "score_margin": round(
            score_margin,
            3,
        ),
        "candidates": _candidate_payload(scored),
    }


def _deemphasize_functional_root(
    *,
    root_decision: Dict[str, Any],
    pos_decision: Dict[str, Any],
) -> Dict[str, Any]:
    selected_pos = pos_decision.get("value")

    if selected_pos not in FUNCTIONAL_POS:
        return root_decision

    updated = dict(root_decision)

    updated["strategy"] = (
        "functional_word_root_deemphasized"
    )

    updated["confidence_score"] = min(
        float(
            updated.get(
                "confidence_score",
                0.0,
            )
            or 0.0
        ),
        0.35,
    )

    updated["confidence_level"] = "low"

    updated["deemphasized"] = True

    updated["note"] = (
        "Root evidence is not treated as a major "
        "expert feature for functional words."
    )

    return updated


def _build_morphology_decision(
    *,
    final: Dict[str, Any],
    tools: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    supporting_tools = []

    for tool in (
        "camel",
        "alkhalil",
        "sinatools",
    ):
        token = tools.get(tool)

        if isinstance(token, dict):
            supporting_tools.append(tool)

    score = final.get(
        "confidence_score",
        0.6,
    )

    if not isinstance(score, (int, float)):
        score = 0.6

    score = max(
        0.0,
        min(
            float(score),
            1.0,
        ),
    )

    return {
        "value": {
            "gender": final.get("gender"),
            "number": final.get("number"),
            "tense": final.get("tense"),
            "case": final.get("case"),
        },
        "expert": "morphology_expert",
        "primary_source": "camel",
        "supporting_tools": supporting_tools,
        "disagreeing_tools": [],
        "strategy": (
            "camel_primary_with_morphological_support"
        ),
        "confidence_score": score,
        "confidence_level": _confidence_level(score),
        "ambiguity": False,
    }


def _calculate_expert_confidence(
    *,
    expert_decisions: Dict[str, Dict[str, Any]],
    selected_pos: Optional[str],
) -> Tuple[float, str]:
    weighted_scores: List[
        Tuple[float, float]
    ] = []

    confidence_weights = {
        "segmentation": 0.80,
        "lemma": 1.20,
        "root": 0.65,
        "pos": 1.30,
        "dependency": 1.00,
        "morphology": 0.90,
    }

    for feature, decision in expert_decisions.items():
        if not isinstance(decision, dict):
            continue

        strategy = decision.get("strategy")

        if strategy == "no_capable_evidence":
            continue

        value = decision.get("value")

        if not _valid(value):
            continue

        score = decision.get(
            "confidence_score",
        )

        if not isinstance(
            score,
            (int, float),
        ):
            continue

        weight = confidence_weights.get(
            feature,
            1.0,
        )

        if (
            feature == "root"
            and selected_pos in FUNCTIONAL_POS
        ):
            weight = 0.10

        weighted_scores.append(
            (
                float(score),
                weight,
            )
        )

    if not weighted_scores:
        return 0.0, "low"

    numerator = sum(
        score * weight
        for score, weight in weighted_scores
    )

    denominator = sum(
        weight
        for _, weight in weighted_scores
    )

    expert_score = (
        numerator / denominator
        if denominator
        else 0.0
    )

    important_low = False

    for feature in IMPORTANT_EXPERT_FEATURES:
        decision = expert_decisions.get(
            feature,
            {},
        )

        if (
            decision.get("strategy")
            == "no_capable_evidence"
        ):
            continue

        if (
            decision.get("confidence_level")
            == "low"
        ):
            important_low = True
            break

    if important_low:
        expert_score = min(
            expert_score,
            0.74,
        )

    pos_decision = expert_decisions.get(
        "pos",
        {},
    )

    if pos_decision.get("ambiguity"):
        expert_score = min(
            expert_score,
            0.69,
        )

    expert_score = max(
        0.0,
        min(
            expert_score,
            1.0,
        ),
    )

    return (
        round(
            expert_score,
            3,
        ),
        _confidence_level(
            expert_score,
        ),
    )


def apply_expert_fusion(
    *,
    classic_fused: Dict[str, Any],
    tools: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    final = dict(
        classic_fused.get(
            "final",
            {},
        )
    )

    sources = dict(
        classic_fused.get(
            "sources",
            {},
        )
    )

    expert_decisions: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # POS is calculated first because root importance
    # depends on the final POS category.
    pos_decision = _choose_by_weighted_consensus(
        feature="pos",
        tools=tools,
        fallback_value=final.get("pos"),
        fallback_source=sources.get("pos"),
    )

    expert_decisions["pos"] = pos_decision

    if _valid(pos_decision.get("value")):
        final["pos"] = pos_decision["value"]

        sources["pos"] = (
            pos_decision.get("primary_source")
            or sources.get("pos")
        )

    for feature in (
        "segmentation",
        "lemma",
        "root",
        "dependency",
    ):
        decision = _choose_by_weighted_consensus(
            feature=feature,
            tools=tools,
            fallback_value=final.get(feature),
            fallback_source=sources.get(feature),
        )

        if feature == "root":
            decision = _deemphasize_functional_root(
                root_decision=decision,
                pos_decision=pos_decision,
            )

        expert_decisions[feature] = decision

        if _valid(decision.get("value")):
            final[feature] = decision["value"]

            sources[feature] = (
                decision.get("primary_source")
                or sources.get(feature)
            )

    morphology_decision = (
        _build_morphology_decision(
            final=final,
            tools=tools,
        )
    )

    expert_decisions[
        "morphology"
    ] = morphology_decision

    expert_score, expert_level = (
        _calculate_expert_confidence(
            expert_decisions=expert_decisions,
            selected_pos=pos_decision.get("value"),
        )
    )

    final[
        "expert_confidence_score"
    ] = expert_score

    final[
        "expert_confidence_level"
    ] = expert_level

    updated = {
        **classic_fused,
        "final": final,
        "sources": sources,
        "expert_decisions": expert_decisions,
        "fusion_mode": "expert_fusion",
        "expert_summary": {
            "strategy": (
                "feature_specific_capability_weighted_fusion"
            ),
            "confidence_score": expert_score,
            "confidence_level": expert_level,
            "pos_ambiguity": bool(
                pos_decision.get("ambiguity")
            ),
            "important_low_confidence": [
                feature
                for feature in IMPORTANT_EXPERT_FEATURES
                if (
                    expert_decisions.get(
                        feature,
                        {},
                    ).get(
                        "confidence_level"
                    )
                    == "low"
                )
            ],
        },
    }

    updated[
        "decision_trace"
    ] = _build_expert_decision_trace(
        expert_decisions
    )

    return updated


def _build_expert_decision_trace(
    expert_decisions: Dict[
        str,
        Dict[str, Any],
    ],
) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []

    for feature, decision in expert_decisions.items():
        trace.append(
            {
                "feature": feature,
                "chosen_value": decision.get(
                    "value"
                ),
                "expert": decision.get(
                    "expert"
                ),
                "primary_source": decision.get(
                    "primary_source"
                ),
                "supporting_tools": decision.get(
                    "supporting_tools",
                    [],
                ),
                "disagreeing_tools": decision.get(
                    "disagreeing_tools",
                    [],
                ),
                "strategy": decision.get(
                    "strategy"
                ),
                "confidence_score": decision.get(
                    "confidence_score"
                ),
                "confidence_level": decision.get(
                    "confidence_level"
                ),
                "ambiguity": bool(
                    decision.get(
                        "ambiguity",
                        False,
                    )
                ),
                "score_margin": decision.get(
                    "score_margin"
                ),
                "note": decision.get(
                    "note"
                ),
            }
        )

    return trace