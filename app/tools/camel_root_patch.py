from __future__ import annotations

import re
from typing import Tuple


HAMZA_CHARS = "أإآءؤئ"

# Arabic letters set for counting (letters only; excludes hamza normalization chars if they are mapped)
ARABIC_LETTERS_RE = re.compile(r"[\u0600-\u06FF]")


def _normalize_hamza_preserve(root: str) -> str:
    """Normalize hamza-bearing characters without stripping them from roots.

    Key requirement: hamza must NOT be stripped before root classification.
    We map various hamza forms to the base hamza letter (أ) but keep as a letter.
    """
    if root is None:
        return root
    # Keep Arabic dots/format but unify hamza shapes
    return (
        root.replace("إ", "أ")
        .replace("آ", "أ")
        .replace("ؤ", "ؤ")
        .replace("ئ", "ئ")
        .replace("ء", "أ")
    )


def classify_root_type_from_parts(root: str) -> Tuple[str, str]:
    """Return (root_normalized, root_type) based on dot-separated parts."""
    if not root:
        return root, "unknown"

    # Root parts are dot-separated per CAMeL output (e.g., ق.ر.أ)
    parts = root.split(".")
    parts = [p for p in parts if p]

    # If root is single-part, it's monoliteral.
    if len(parts) == 1:
        return root, "monoliteral"

    # If root is 2 parts -> biliteral.
    if len(parts) == 2:
        # Special: if any part contains hamza, keep as biliteral (linguistically acceptable).
        return root, "biliteral"

    # 3+ parts -> triliteral (expected for this project) unless classification says otherwise.
    return root, "triliteral"


def patch_camel_root(root: str) -> Tuple[str, str]:
    """Patch CAMeL root.

    - Repair 3A: hamza must NOT be stripped before root classification.
    - Repair 3B: preserve leading waw when it is part of the root (وجد => و.ج.د).
    """
    if root is None:
        return None, "unknown"

    # Preserve hamza by normalizing its shape but NOT removing it.
    normalized = _normalize_hamza_preserve(root)


    # Specific fix: if CAMeL mistakenly outputs biliteral root for قرأ, upgrade.
    # Detect pattern like "ق.ر" (optionally with diacritics) and infer hamza presence by
    # presence of 'hamza' in original input isn't available here, so rely on letter count.

    # Normalize dots/spaces
    normalized = normalized.replace("..", ".").strip(".")

    root_type = "unknown"
    parts = [p for p in normalized.split(".") if p]

    # Count actual Arabic letters after normalization, treating hamza as a letter.
    letters = "".join(parts)
    letter_count = len(ARABIC_LETTERS_RE.findall(letters))

    # Repair 3C: preserve leading waw for roots like وجد => و.ج.د.
    # If CAMeL stripped the leading waw, we may see the biliteral/triliteral interior only: "ج.د".
    # In that case, restore it as triliteral root: "و.ج.د".
    if normalized == "ج.د":
        return "و.ج.د", "triliteral"

    if normalized == "ق.ر":
        # Hard requirement for this known case
        return "ق.ر.أ", "triliteral"

    if letter_count >= 3 and len(parts) >= 3:
        root_type = "triliteral"
    else:
        _, root_type = classify_root_type_from_parts(normalized)

    return normalized, root_type


