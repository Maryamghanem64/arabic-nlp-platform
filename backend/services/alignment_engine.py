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

    pos_agree = 0
    lemma_agree = 0
    root_agree = 0
    seg_agree = 0
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

        ref_pos = normalize_pos_for_compare(extract_pos(ref))

        tool_pos_vals = [
            normalize_pos_for_compare(extract_pos(t))
            for t in (camel, stanza, qalsadi, alkhalil, udpipe)
            if t and extract_pos(t)
        ]

        if tool_pos_vals and all(v == ref_pos for v in tool_pos_vals):
            pos_agree += 1

        ref_lemma = _extract_lemma(ref)
        lemma_vals = [
            _extract_lemma(t)
            for t in (camel, stanza, qalsadi, alkhalil, udpipe)
            if _extract_lemma(t)
        ]

        if lemma_vals and ref_lemma:
            norm_ref = normalize_lemma_for_compare(ref_lemma)
            if all(normalize_lemma_for_compare(v) == norm_ref for v in lemma_vals):
                lemma_agree += 1
            elif all(is_mwt_clitic_artifact(v) or is_gender_convention_pair(v, ref_lemma) for v in lemma_vals):
                lemma_agree += 1



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

    return {
        "token_count": total,
        "pos_agreement": pct(pos_agree),
        "lemma_agreement": pct(lemma_agree),
        "root_agreement": pct(root_agree),
        "segmentation_agreement": pct(seg_agree),
    }
