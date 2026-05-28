from __future__ import annotations

"""Runtime compatibility helpers.

camel-tools (and/or farasapy) may import `emoji.EMOJI_DATA`.
Newer versions of `emoji` removed it.

This module patches the runtime `emoji` module by creating a minimal
`EMOJI_DATA` mapping when missing.
"""

from typing import Any, Dict


def ensure_emoji_emoji_data() -> None:
    try:
        import emoji  # type: ignore

        if hasattr(emoji, "EMOJI_DATA"):
            return

        # Some libraries expect `EMOJI_DATA` to exist at import-time.
        # We create a minimal placeholder mapping.
        emoji.EMOJI_DATA = {}  # type: ignore[attr-defined]


    except Exception:
        # Never fail startup due to emoji compatibility.
        return

