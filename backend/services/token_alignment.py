from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AlignedToolToken:
    tool: str
    token: Optional[Dict[str, Any]]


def _strip(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _norm_surface(s: Optional[str]) -> str:
    # Minimal normalization for alignment; keep it deterministic and safe.
    return _strip(s)


def _is_punct(surface: str) -> bool:
    if not surface:
        return False
    # Arabic punctuation + common ASCII punctuations
    punct = set(".،؛؟!?؛:,\"'()[]{}-—…")
    return all(ch in punct for ch in surface)


def align_tokens_by_surface(
    *,
    base_tokens: List[Dict[str, Any]],
    tool_tokens: List[Dict[str, Any]],
    tool_name: str,
) -> List[Optional[Dict[str, Any]]]:
    """Align tool tokens to base token indices.

    Key idea:
    - base is the canonical sequence (typically the UI tokenization)
    - tools may have segmentation differences (e.g., Farasa)
    - alignment is based on surface matches and punctuation handling

    Returns a list of length len(base_tokens) with either a matched tool token or None.
    """

    # Defensive copies of surfaces
    base_surfaces = [_norm_surface(t.get("surface")) for t in base_tokens]
    tool_surfaces = [_norm_surface(t.get("surface")) for t in tool_tokens]

    # Precompute segmentation maps for tools that provide it.
    # If a tool token has segmentation like ["ال", "كتاب"], we allow matching that
    # segmentation against a base surface.
    tool_seg = []
    for t in tool_tokens:
        seg = t.get("segmentation")
        if isinstance(seg, list) and seg:
            tool_seg.append([_norm_surface(x) for x in seg if _norm_surface(x)])
        else:
            tool_seg.append([_norm_surface(t.get("surface"))])

    out: List[Optional[Dict[str, Any]]] = [None] * len(base_tokens)

    # Greedy forward matching with backtracking buffer limited by local ambiguity.
    ti = 0
    for bi, bs in enumerate(base_surfaces):
        if not bs:
            continue

        matched: Optional[int] = None

        # Try direct match first
        for k in range(ti, len(tool_tokens)):
            ts = tool_surfaces[k]
            if not ts:
                continue
            if _norm_surface(ts) == bs:
                matched = k
                break

            # If base is punctuation, match punctuation even if segmentation differs.
            if _is_punct(bs) and _is_punct(ts):
                if ts == bs:
                    matched = k
                    break

            # If tool uses segmentation and base surface equals a joined segmentation part
            # (useful for cases where token surfaces differ but segmentation contains the base)
            # Also try join of segmentation without spaces.
            segs = tool_seg[k]
            if segs:
                joined = "".join(segs)
                if joined == bs:
                    matched = k
                    break
                # Also allow exact segment contained matches
                if bs in segs:
                    matched = k
                    break

            # Stop early if we passed a plausible punctuation boundary
            if _is_punct(bs) and k > ti:
                # still keep searching only locally
                pass

        if matched is None:
            # No match; leave None
            continue

        out[bi] = tool_tokens[matched]
        ti = matched + 1

    return out


def align_all_tools(
    *,
    base_tokens: List[Dict[str, Any]],
    tools_tokens: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    """Align multiple tools to the same base token sequence."""

    aligned: Dict[str, List[Optional[Dict[str, Any]]]] = {}
    for tool_name, toks in tools_tokens.items():
        aligned[tool_name] = align_tokens_by_surface(
            base_tokens=base_tokens,
            tool_tokens=toks,
            tool_name=tool_name,
        )
    return aligned

