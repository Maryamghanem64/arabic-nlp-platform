from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AlignedToken:
    base: Dict[str, Any]
    tools: Dict[str, Optional[Dict[str, Any]]]


def _norm_surface(s: Optional[str]) -> str:
    return "" if s is None else str(s).strip()


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _strip_diacritics_local(text: Optional[str]) -> str:
    # Local fallback: avoid importing heavy utilities.
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
    """Return possible surface variants for matching.

    For Farasa: segmentation is list of parts; we may match base surface
    against joined segmentation or a segment part.
    """
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
    """Align each base token with at most one token per tool.

    Alignment strategy:
    - Greedy forward match by surface.
    - If punctuation: punctuation surfaces must match.
    - If segmentation differs: base surface may match a Farasa segment
      or joined segmentation.

    Returns:
      aligned: list length == len(base_tokens)
      meta: debug information (kept small)
    """

    # Precompute tool variants for each token index.
    tool_variants: Dict[str, List[List[str]]] = {}
    tool_surfaces: Dict[str, List[str]] = {}
    for tool_name, toks in tools_tokens.items():
        tool_surfaces[tool_name] = [_norm_surface(t.get("surface")) for t in toks]
        tool_variants[tool_name] = [_tool_token_variants(t) for t in toks]

    aligned: List[AlignedToken] = []

    # Greedy: maintain current pointer per tool.
    pointers: Dict[str, int] = {tool: 0 for tool in tools_tokens.keys()}

    matched_counts: Dict[str, int] = {tool: 0 for tool in tools_tokens.keys()}

    for base in base_tokens:
        base_surface = _norm_surface(base.get("surface"))

        tools_map: Dict[str, Optional[Dict[str, Any]]] = {tool: None for tool in tools_tokens.keys()}

        for tool_name, toks in tools_tokens.items():
            start = pointers.get(tool_name, 0)
            found_idx: Optional[int] = None

            # punctuation: strict
            if _is_punct(base_surface):
                for j in range(start, len(toks)):
                    if _norm_surface(toks[j].get("surface")) == base_surface:
                        found_idx = j
                        break
            else:
                for j in range(start, len(toks)):
                    # direct surface
                    if _norm_surface(toks[j].get("surface")) == base_surface:
                        found_idx = j
                        break
                    # segmentation/variant match
                    if base_surface:
                        if base_surface in tool_variants[tool_name][j]:
                            found_idx = j
                            break

            if found_idx is not None:
                pointers[tool_name] = found_idx + 1
                tools_map[tool_name] = toks[found_idx]
                matched_counts[tool_name] += 1

        aligned.append(AlignedToken(base=base, tools=tools_map))

    meta = {"matched_counts": matched_counts, "base_count": len(base_tokens)}
    return aligned, meta


def compute_agreements(
    *,
    aligned_tokens: List[AlignedToken],
) -> Dict[str, Any]:
    """Compute pos/lemma/root/segmentation agreements using aligned normalized tokens."""

    def eq(a: Optional[Any], b: Optional[Any]) -> bool:
        if a is None or b is None:
            return False
        sa = _strip_diacritics_local(str(a))
        sb = _strip_diacritics_local(str(b))
        return sa == sb and sa != ""

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

        # Agreement is computed across tools that have value; for compact UI we
        # use camel as reference when available, otherwise stanza/qalsadi.
        ref = camel or stanza or qalsadi

        if ref:
            # POS agreement: all present must match reference
            ref_pos = ref.get("pos")
            tool_pos_vals = [
                t.get("pos") for t in [camel, stanza, qalsadi] if t and t.get("pos")
            ]
            if tool_pos_vals and all(v == ref_pos for v in tool_pos_vals):
                pos_agree += 1

            # lemma/root
            ref_lemma = ref.get("lemma")
            lemma_vals = [t.get("lemma") for t in [camel, stanza, qalsadi] if t and t.get("lemma")]
            if lemma_vals and all(eq(v, ref_lemma) for v in lemma_vals):
                lemma_agree += 1

            ref_root = ref.get("root")
            root_vals = [t.get("root") for t in [camel, stanza, qalsadi] if t and t.get("root")]
            if root_vals and all(eq(v, ref_root) for v in root_vals):
                root_agree += 1

        # segmentation agreement: farasa segments compared to base segmentation
        if farasa and isinstance(farasa.get("segmentation"), list):
            seg = farasa.get("segmentation")
            base_seg = atok.base.get("segmentation")
            if isinstance(base_seg, list):
                # Compare joined versions
                if "".join([_safe_str(x) for x in seg]) == "".join([_safe_str(x) for x in base_seg]):
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

