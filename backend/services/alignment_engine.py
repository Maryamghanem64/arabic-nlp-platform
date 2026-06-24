from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.utils.helpers import normalize_pos_for_compare


@dataclass(frozen=True)
class AlignedToken:
    base: Dict[str, Any]
    tools: Dict[str, Optional[Dict[str, Any]]]


def _norm_surface(s: Optional[str]) -> str:
    return "" if s is None else str(s).strip()


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _strip_diacritics_local(text: Optional[str]) -> str:
    if text is None:
        return ""
    import re
    return re.sub(r"[\u064B-\u065F\u0670]", "", str(text))


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

    seg = tok.get("segmentation")
    if isinstance(seg, list) and seg:
        seg_norm = [_norm_surface(x) for x in seg if _norm_surface(x)]
        if seg_norm:
            variants.append("".join(seg_norm))
            variants.extend(seg_norm)

    return [v for v in variants if v]


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

            if _is_punct(base_surface):
                for j in range(start, len(toks)):
                    if _norm_surface(toks[j].get("surface")) == base_surface:
                        found_idx = j
                        break
            else:
                for j in range(start, len(toks)):
                    if _norm_surface(toks[j].get("surface")) == base_surface:
                        found_idx = j
                        break

                    # Direct surface match failed. For MWT-like cases, allow:
                    # 1) base_surface appears in any token variants produced by the tool
                    # 2) tool surface equals the first part of base_surface (common when stanza splits MWT)
                    if base_surface and base_surface in tool_variants[tool_name][j]:
                        found_idx = j
                        break

                    if base_surface and toks[j].get("segmentation") and isinstance(toks[j].get("segmentation"), list):
                        parts = [str(p) for p in toks[j].get("segmentation") if p]
                        if parts:
                            # If the tool broke base into parts, align using the first part as a key.
                            if _norm_surface(parts[0]) == base_surface:
                                found_idx = j
                                break

                    # If tool token surface is a suffix/prefix-part of base_surface, allow that as a key.
                    tool_surf = _norm_surface(toks[j].get("surface"))
                    if tool_surf and (tool_surf == base_surface or tool_surf in base_surface or base_surface in tool_surf):
                        found_idx = j
                        break


            if found_idx is not None:
                pointers[tool_name] = found_idx + 1
                tools_map[tool_name] = toks[found_idx]
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


    def strip(a: Any) -> str:
        return _strip_diacritics_local(str(a)) if a is not None else ""

    def extract_pos(t: Optional[Dict[str, Any]]) -> Optional[str]:
        if not t:
            return None

        # camel / qalsadi style
        if "analyses" in t and isinstance(t["analyses"], list):
            first = t["analyses"][0] if t["analyses"] else {}
            return first.get("pos") or first.get("upos")

        # stanza style
        return t.get("pos") or t.get("upos")

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
        farasa = atok.tools.get("farasa")

        ref = camel or stanza or qalsadi
        if not ref:
            continue

        ref_pos = normalize_pos_for_compare(extract_pos(ref))

        tool_pos_vals = [
            normalize_pos_for_compare(extract_pos(t))
            for t in (camel, stanza, qalsadi)
            if t and extract_pos(t)
        ]

        if tool_pos_vals and all(v == ref_pos for v in tool_pos_vals):
            pos_agree += 1

        ref_lemma = _extract_lemma(ref)
        lemma_vals = [
            _extract_lemma(t)
            for t in (camel, stanza, qalsadi)
            if _extract_lemma(t)
        ]

        if lemma_vals and ref_lemma and all(strip(v) == strip(ref_lemma) for v in lemma_vals):
            lemma_agree += 1



        ref_root = ref.get("root")

        root_vals = [
            t.get("root")
            for t in (camel, stanza, qalsadi)
            if t and t.get("root")
        ]
        if root_vals and ref_root and all(strip(v) == strip(ref_root) for v in root_vals):
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