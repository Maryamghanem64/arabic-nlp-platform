from __future__ import annotations

from typing import Any, Dict

from backend.analyzers.base_tool import BaseTool
from backend.analyzers.legacy_stanza_tool import StanzaTool as LegacyStanzaTool



class StanzaUnifiedTool(BaseTool):
    name = "stanza"
    approach = "stanza.pipeline"

    def __init__(self) -> None:
        self._legacy = LegacyStanzaTool()

    def analyze(self, text: str) -> Dict[str, Any]:
        try:
            return self._legacy.analyze(text)
        except Exception as e:
            return {"tool": self.name, "status": "error", "error": str(e), "tokens": []}

