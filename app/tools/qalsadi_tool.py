from __future__ import annotations

from typing import Any, Dict

from app.tools.base_tool import BaseTool
from app.tools.qalsadi_wrapper import qalsadi_analyze


class QalsadiTool(BaseTool):
    tool_name = "qalsadi"

    def is_loaded(self) -> bool:
        return True

    def analyze(self, text: str) -> Dict[str, Any]:
        return qalsadi_analyze(text)

