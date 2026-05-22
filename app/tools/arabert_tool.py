from __future__ import annotations

from typing import Any, Dict

from app.core.tool_registry import has_module, unavailable_result


def arabert_analyze(text: str) -> Dict[str, Any]:
    if not has_module("transformers"):
        return unavailable_result("arabert", "Optional AraBERT unavailable: missing transformers package.", text)
    if not has_module("torch"):
        return unavailable_result("arabert", "Optional AraBERT unavailable: missing torch package.", text)
    return unavailable_result(
        "arabert",
        "AraBERT package dependencies are present, but contextual analysis is not configured in this project.",
        text,
    )


class AraBERTTool:
    tool_name = "arabert"

    def analyze(self, text: str) -> Dict[str, Any]:
        return arabert_analyze(text)

    def is_loaded(self) -> bool:
        return has_module("transformers") and has_module("torch")
