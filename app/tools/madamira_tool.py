from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from app.utils.logger import logger


madamira_available = False


def load_madamira() -> None:
    global madamira_available

    madamira_home = os.environ.get("MADAMIRA_HOME", "")
    madamira_dir = Path(__file__).resolve().parents[2] / "tools" / "madamira"

    if madamira_home and Path(madamira_home).is_dir():
        madamira_available = True
        logger.info(f"✅ MADAMIRA found: {madamira_home}")
    elif madamira_dir.is_dir():
        madamira_available = True
        logger.info(f"✅ MADAMIRA found: {madamira_dir}")
    else:
        madamira_available = False
        logger.info("⏳ MADAMIRA not configured — marked as optional")


def madamira_analyze(text: str) -> Dict[str, Any]:
    tool = "madamira"
    try:
        if not madamira_available:
            return {
                "tool": tool,
                "status": "unavailable",
                "reason": "MADAMIRA not configured. Set MADAMIRA_HOME or place files in tools/madamira/",
                "input": text,
                "word_count": 0,
                "tokens": [],
            }

        # Future: implement actual MADAMIRA client.
        return {
            "tool": tool,
            "status": "unavailable",
            "reason": "MADAMIRA adapter execution is not implemented in this deployment.",
            "input": text,
            "word_count": 0,
            "tokens": [],
        }

    except Exception as e:
        return {
            "tool": tool,
            "status": "error",
            "reason": str(e),
            "input": text,
            "word_count": 0,
            "tokens": [],
        }

