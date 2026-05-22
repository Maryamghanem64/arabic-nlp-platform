from __future__ import annotations

from typing import Any, Dict, List, Optional


def placeholder(value: Optional[str], *, default: str = "-") -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def pos_badge(pos: Optional[str]) -> str:
    # Simple deterministic badge colors for UI.
    # UI can also ignore this and compute its own, but backend will send it.
    if not pos:
        return "gray"
    p = str(pos).upper()
    return {
        "VERB": "blue",
        "NOUN": "green",
        "ADJ": "purple",
        "ADV": "orange",
        "PRON": "teal",
        "ADP": "brown",
        "PART": "pink",
        "CCONJ": "cyan",
        "SCONJ": "cyan",
        "PUNCT": "gray",
        "NUM": "gold",
        "X": "gray",
    }.get(p, "gray")


def status_from_agreement(pos_ok: bool, lemma_ok: bool, root_ok: bool) -> tuple[str, str]:
    if pos_ok and lemma_ok and root_ok:
        return "full", "green"
    if pos_ok or lemma_ok or root_ok:
        return "partial", "yellow"
    return "none", "red"


def compute_status_color(pos_ok: bool, lemma_ok: bool, root_ok: bool) -> str:
    return status_from_agreement(pos_ok, lemma_ok, root_ok)[1]


def safe_pos(pos: Optional[str]) -> str:
    if not pos:
        return "X"
    p = str(pos).strip().upper()
    return p if p else "X"


def safe_features_tense(tense: Any) -> Optional[str]:
    # Keep as-is; UI doesn’t require. This is a placeholder helper.
    if tense is None:
        return None
    s = str(tense).strip()
    return s if s else None

