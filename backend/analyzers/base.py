from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Analyzer(ABC):
    tool_name: str

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """Return a tool result dictionary."""
        raise NotImplementedError


def get_tokens(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return result.get("tokens", []) if isinstance(result, dict) else []

