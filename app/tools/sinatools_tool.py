from __future__ import annotations

from typing import Any, Dict

from app.utils.logger import logger


# ── SinaTools — FUTURE WORK ────────────────────────────────────
# SinaTools was evaluated but excluded due to:
# - Large model size (~880MB lemmas_dic.pickle)
# - Slow loading time on Windows
# - Planned for future microservice deployment
# Reference: Saadiyeh et al. (2024), IJ-AI Journal


sinatools_analyzer = None


def load_sinatools() -> None:
    # Future work — do not load at startup
    logger.info("⏳ SinaTools — Future Work (not loaded)")


def sinatools_analyze(text: str) -> Dict[str, Any]:
    return {
        "tool": "sinatools",
        "status": "future_work",
        "message": "SinaTools planned for future microservice deployment",
        "reason": "Large model size (~880MB) excluded from current implementation",
        "tokens": [],
    }


class SinaToolsTool:
    tool_name = "sinatools"

    def is_loaded(self) -> bool:
        return False

    def analyze(self, text: str) -> Dict[str, Any]:
        return sinatools_analyze(text)

