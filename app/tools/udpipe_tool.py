from __future__ import annotations

from typing import Any, Dict


# Partner stub — UDPipe via REST API
# GET https://lindat.mff.cuni.cz/services/udpipe/api/process


class UDPipeTool:
    tool_name = "udpipe"

    def analyze(self, text: str) -> Dict[str, Any]:
        return {"tool": "udpipe", "status": "not_implemented", "tokens": []}

    def is_loaded(self) -> bool:
        return False

