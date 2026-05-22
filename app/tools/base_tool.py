from __future__ import annotations

from typing import Any, Dict


class BaseTool:
    tool_name: str

    def is_loaded(self) -> bool:
        return False

    def analyze(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

