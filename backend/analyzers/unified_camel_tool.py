from __future__ import annotations

from typing import Any, Dict

from backend.analyzers.base_tool import BaseTool
from backend.analyzers.legacy_camel_tool import CamelTool as LegacyCamelTool



class CamelUnifiedTool(BaseTool):
    """Unified wrapper around the existing legacy CAMeL analyzer.

    Note: legacy analyzers are imported from *camel_tool_legacy* to avoid circular imports.
    """

    name = "camel"
    approach = "camel_tools.mle_disambiguator"

    def __init__(self) -> None:
        self._legacy = LegacyCamelTool()

    def analyze(self, text: str) -> Dict[str, Any]:
        try:
            return self._legacy.analyze(text)
        except Exception as e:
            return {"tool": self.name, "status": "error", "error": str(e), "tokens": []}

