from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.utils.text_norm import is_arabic_token


REPEATED_CHARS_RE = re.compile(r"(.)\1{3,}")


def detect_suspicious_for_token(
    surface: str,
    lemma: Optional[str],
    pos: Optional[str],
    tool_name: str,
    raw: Dict[str, Any] | None = None,
) -> List[str]:
    flags: List[str] = []

    if not surface or not is_arabic_token(surface):
        flags.append("invalid_output")
        return flags

    if lemma is None or lemma == "":
        flags.append("weak_analysis")
        return flags

    lemma_str = str(lemma)

    if not pos:
        flags.append("missing_pos")

    # lemma shorter than expected (heuristic)
    # if lemma is drastically shorter than surface, suspicious
    if len(lemma_str) < max(2, int(0.4 * len(surface))):
        flags.append("suspicious_lemma_length")

    # lemma equals a 1-2 char truncation or empty-ish
    if len(lemma_str) <= 2 and surface != lemma_str:
        flags.append("suspicious_lemma")

    # repeated characters pattern (bad normalization)
    if REPEATED_CHARS_RE.search(lemma_str):
        flags.append("repeated_characters")

    # specific Qalsadi suspicion: lemma collapses to near-surface consonants inconsistently
    if tool_name == "qalsadi":
        # Example target: "كتب" -> "تب" (lemma too short + contains missing pattern)
        if len(lemma_str) <= max(2, int(0.5 * len(surface))):
            flags.append("qalsadi_suspicious_lemma")

    return list(dict.fromkeys(flags))

