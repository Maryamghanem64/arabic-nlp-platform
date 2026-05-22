from __future__ import annotations

from typing import Any, Dict


# Partner stub — AraBERT via HuggingFace transformers
# Model: aubmindlab/bert-base-arabertv02


class AraBERTTool:
    tool_name = "arabert"

    def analyze(self, text: str) -> Dict[str, Any]:
        return {"tool": "arabert", "status": "not_implemented", "tokens": []}

    def is_loaded(self) -> bool:
        return False

