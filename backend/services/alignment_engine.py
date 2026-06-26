from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.utils.helpers import (
    is_gender_convention_pair,
    is_mwt_clitic_artifact,
    normalize_lemma_for_compare,
    normalize_pos_for_compare,
    strip_diacritics,
)

_NON_CONTENT_POS = {"CCONJ", "SCONJ", "PART", "PUNCT", "SYM"}
TOOL_WEIGHTS = {
    "camel": 0.35,
    "stanza": 0.35,
    "udpipe": 0.15,
    "qalsadi": 0.10,
    "alkhalil": 0.05,
}


@dataclass(frozen=True)
class AlignedToken:
    base: Dict[str, Any]
    tools: Dict[str, Optional[Dict[str, Any]]]


def _norm_surface(s: Optional[str]) -> str:
    return "" if s is None else str(s).strip()


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _is_punct(surface: str) -> bool:
    if not surface:
        return False
    punct = set(".،؛؟!?؛:,\"'()[]{}-—…")
    return all(ch in punct for ch in surface)


def _tool_token_variants(tok: Dict[str, Any]) -> List[str]:
    variants: List[str] = []

    surface = _norm_surface(tok.get("surface"))
    if surface:
        variants.append(surface)

    original_surface = _norm_surface(tok.get("original_surface"))
    if original_surface and original_surface not in variants:
        variants.append(original_surface)

    seg = tok.get("segmentation")
    if isinstance(seg, list) and seg:
        seg_norm = [_norm_surface(x) for x in seg if _norm_surface(x)]
        if seg_norm:
            variants.append("".join(seg_norm))
            variants.extend(seg_norm)

    return [v for v in variants if v]


def _concat_surface(tokens: List[Dict[str, Any]], start: int, end: int) -> str:
    parts: List[str] = []
    for idx in range(start, end):
        part = _norm_surface(tokens[idx].get("surface"))
        if part:
            parts.append(part)
    return "".join(parts)


def _first_non_empty(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _token_pos_candidates(tok: Optional[Dict[str, Any]]) -> List[str]:
    if not tok:
        return []

    candidates: List[str] = []
    analyses = tok.get("analyses")
    if isinstance(analyses, list):
        for analysis in analyses:
            if not isinstance(analysis, dict):
                continue
            for key in ("pos", "upos"):
                val = _first_non_empty(analysis.get(key))
                if val:
                    candidates.append(val)

    for key in ("pos", "upos"):
        val = _first_non_empty(tok.get(key))
        if val:
            candidates.append(val)

    mwt_words = tok.get("mwt_words")
    if isinstance(mwt_words, list):
        for word in mwt_words:
            if not isinstance(word, dict):
                continue
            for key in ("upos", "pos"):
                val = _first_non_empty(word.get(key))
                if val:
                    candidates.append(val)

    return candidates


def _preferred_pos(tok: Optional[Dict[str, Any]]) -> Optional[str]:
    candidates = [normalize_pos_for_compare(v) for v in _token_pos_candidates(tok)]
    candidates = [c for c in candidates if c]
    if not candidates:
        return None

    for candidate in candidates:
        if candidate not in _NON_CONTENT_POS and candidate != "X":
            return candidate
    return candidates[0]


def _weighted_majority_ratio(values_by_tool: Dict[str, Optional[str]], *, normalize: Optional[Any] = None) -> tuple[float, float, int]:
    counts: Dict[str, float] = {}
    total_weight = 0.0

    for tool_name, value in values_by_tool.items():
        if value is None:
            continue

        text = str(value).strip()
        if not text:
            continue

        if normalize is not None:
            text = normalize(text)

        if not text or text == "X":
            continue

        weight = float(TOOL_WEIGHTS.get(tool_name, 0.0))
        if weight <= 0:
            continue

        counts[text] = counts.get(text, 0.0) + weight
        total_weight += weight

    if total_weight <= 0:
        return 0.0, 0.0, 0

    majority = max(counts.values()) if counts else 0.0
    return majority / total_weight, majority, 1


def weighted_pos_agreement(tool_pos_map: Dict[str, Optional[str]]) -> float:
    """
    Compute weighted POS agreement across tools.
    Excludes tools that returned None or 'X' from both numerator and denominator.
    Returns float 0.0-1.0.
    """
    votes: List[tuple[str, float]] = []
    for tool, pos in tool_pos_map.items():
        if not pos or pos == "X":
            continue
        weight = TOOL_WEIGHTS.get(tool, 0.05)
        votes.append((pos, weight))

    if not votes:
        return 0.0

    total_weight = sum(w for _, w in votes)
    pos_scores: Dict[str, float] = {}
    for pos, weight in votes:
        pos_scores[pos] = pos_scores.get(pos, 0.0) + weight

    winner_weight = max(pos_scores.values())
    return round(winner_weight / total_weight, 3)


def _preferred_lemma(tok: Optional[Dict[str, Any]]) -> Optional[str]:
    if not tok:
        return None
    analyses = tok.get("analyses")
    if isinstance(analyses, list) and analyses:
        first = analyses[0] if isinstance(analyses[0], dict) else {}
        lemma = _first_non_empty(first.get("lemma"))
        if lemma:
            return lemma
    return _first_non_empty(tok.get("lemma"))


def align_tools(
    *,
    base_tokens: List[Dict[str, Any]],
    tools_tokens: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[AlignedToken], Dict[str, Any]]:

    tool_variants: Dict[str, List[List[str]]] = {}
    tool_surfaces: Dict[str, List[str]] = {}

    for tool_name, toks in tools_tokens.items():
        tool_surfaces[tool_name] = [_norm_surface(t.get("surface")) for t in toks]
        tool_variants[tool_name] = [_tool_token_variants(t) for t in toks]

    aligned: List[AlignedToken] = []
    pointers: Dict[str, int] = {tool: 0 for tool in tools_tokens}
    matched_counts: Dict[str, int] = {tool: 0 for tool in tools_tokens}

    for base in base_tokens:
        base_surface = _norm_surface(base.get("surface"))
        tools_map: Dict[str, Optional[Dict[str, Any]]] = {
            tool: None for tool in tools_tokens
        }

        for tool_name, toks in tools_tokens.items():
            start = pointers.get(tool_name, 0)
            found_idx: Optional[int] = None
            found_span_len = 1

            if _is_punct(base_surface):
                for j in range(start, len(toks)):
                    if _norm_surface(toks[j].get("surface")) == base_surface:
                        found_idx = j
                        break
            else:
                for j in range(start, len(toks)):
                    tool_surface = _norm_surface(toks[j].get("surface"))
                    if tool_surface == base_surface:
                        found_idx = j
                        break

                    # MWT-aware matching: prefer an exact concatenated span match over
                    # fragment-level substring matching so "و" does not steal "وجدت".
                    max_span = min(len(toks), j + 4)
                    for end in range(j + 2, max_span + 1):
                        if _concat_surface(toks, j, end) == base_surface:
                            found_idx = j
                            found_span_len = end - j
                            break
                    if found_idx is not None:
                        break

                    # Single-token variant match is a safe fallback for segmented tools.
                    if base_surface and base_surface in tool_variants[tool_name][j]:
                        found_idx = j
                        break


            if found_idx is not None:
                matched_token = dict(toks[found_idx])
                match_type = "mwt_match" if found_span_len > 1 else "full_match"
                if _norm_surface(matched_token.get("surface")) != base_surface and found_span_len == 1:
                    if _norm_surface(matched_token.get("original_surface")) == base_surface:
                        match_type = "full_match"
                    else:
                        match_type = "partial_match"
                span_len = found_span_len

                matched_token["alignment"] = {
                    "match_type": match_type,
                    "span_len": span_len,
                    "reconstructed_surface": base_surface,
                }
                if found_span_len > 1:
                    matched_token["alignment"]["span_tokens"] = [dict(toks[idx]) for idx in range(found_idx, found_idx + found_span_len)]
                pointers[tool_name] = found_idx + span_len
                tools_map[tool_name] = matched_token
                matched_counts[tool_name] += 1

        aligned.append(AlignedToken(base=base, tools=tools_map))

    meta = {
        "matched_counts": matched_counts,
        "base_count": len(base_tokens),
    }

    return aligned, meta


def compute_agreements(
    *,
    aligned_tokens: List[AlignedToken],
) -> Dict[str, Any]:

    def _extract_lemma(t: Optional[Dict[str, Any]]) -> Optional[str]:
        if not t:
            return None
        a = t.get("analyses")
        if a and isinstance(a, list) and len(a) > 0:
            first = a[0] if a else {}
            if isinstance(first, dict):
                return first.get("lemma")
        return t.get("lemma")


    def extract_pos(t: Optional[Dict[str, Any]]) -> Optional[str]:
        return _preferred_pos(t)

    pos_agree_sum = 0.0
    lemma_exact_agree_sum = 0.0
    lemma_norm_agree_sum = 0.0
    root_agree = 0
    seg_agree = 0
    pos_total = 0
    lemma_exact_total = 0
    lemma_norm_total = 0
    total = 0

    for atok in aligned_tokens:
        total += 1

        camel = atok.tools.get("camel")
        stanza = atok.tools.get("stanza")
        qalsadi = atok.tools.get("qalsadi")
        alkhalil = atok.tools.get("alkhalil")
        udpipe = atok.tools.get("udpipe")
        farasa = atok.tools.get("farasa")

        ref = camel or stanza or qalsadi or alkhalil or udpipe
        if not ref:
            continue

        pos_ratio = weighted_pos_agreement(
            {
                "camel": extract_pos(camel),
                "stanza": extract_pos(stanza),
                "qalsadi": extract_pos(qalsadi),
                "alkhalil": extract_pos(alkhalil),
                "udpipe": extract_pos(udpipe),
            }
        )
        if pos_ratio > 0:
            pos_agree_sum += pos_ratio
            pos_total += 1

        lemma_exact_ratio, _lemma_exact_majority, lemma_exact_weight = _weighted_majority_ratio(
            {
                "camel": _extract_lemma(camel),
                "stanza": _extract_lemma(stanza),
                "qalsadi": _extract_lemma(qalsadi),
                "alkhalil": _extract_lemma(alkhalil),
                "udpipe": _extract_lemma(udpipe),
            },
        )
        lemma_norm_ratio, _lemma_norm_majority, lemma_norm_weight = _weighted_majority_ratio(
            {
                "camel": _extract_lemma(camel),
                "stanza": _extract_lemma(stanza),
                "qalsadi": _extract_lemma(qalsadi),
                "alkhalil": _extract_lemma(alkhalil),
                "udpipe": _extract_lemma(udpipe),
            },
            normalize=normalize_lemma_for_compare,
        )

        if lemma_exact_weight:
            lemma_exact_agree_sum += lemma_exact_ratio
            lemma_exact_total += 1
        if lemma_norm_weight:
            lemma_norm_agree_sum += lemma_norm_ratio
            lemma_norm_total += 1



        ref_root = ref.get("root")

        root_vals = [
            t.get("root")
            for t in (camel, stanza, qalsadi, alkhalil, udpipe)
            if t and t.get("root")
        ]
        if root_vals and ref_root and all(strip_diacritics(v) == strip_diacritics(ref_root) for v in root_vals):
            root_agree += 1

        if farasa and isinstance(farasa.get("segmentation"), list):
            seg = farasa.get("segmentation") or []
            base_seg = atok.base.get("segmentation") or []

            if "".join(map(str, seg)) == "".join(map(str, base_seg)):
                seg_agree += 1

    def pct(n: int) -> int:
        return int(round((n / total * 100) if total else 0))

    def weighted_pct(sum_ratio: float, count: int) -> int:
        return int(round((sum_ratio / count * 100) if count else 0))

    return {
        "token_count": total,
        "pos_agreement": weighted_pct(pos_agree_sum, pos_total),
        "lemma_agreement": weighted_pct(lemma_norm_agree_sum, lemma_norm_total),
        "lemma_exact_agreement": weighted_pct(lemma_exact_agree_sum, lemma_exact_total),
        "lemma_normalized_agreement": weighted_pct(lemma_norm_agree_sum, lemma_norm_total),
        "root_agreement": pct(root_agree),
        "segmentation_agreement": pct(seg_agree),
    }
