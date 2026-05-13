from __future__ import annotations

import re
from typing import Optional


ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")


DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")


def strip_diacritics(text: Optional[str]) -> str:
    if not text:
        return ""
    return DIACRITICS_RE.sub("", text)


def normalize_whitespace(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_arabic_token(text: str) -> bool:
    if not text:
        return False
    arabic_chars = ARABIC_LETTER_RE.findall(text)
    return len(arabic_chars) >= max(1, int(0.6 * len(text)))


def normalize_pos_tag(pos: Optional[str], mapping: dict[str, str]) -> Optional[str]:
    if not pos:
        return None
    key = pos.strip().upper()
    return mapping.get(key, key)


def validate_surface(surface: str) -> bool:
    # minimal check: contains at least one arabic character
    return bool(ARABIC_LETTER_RE.search(surface))

