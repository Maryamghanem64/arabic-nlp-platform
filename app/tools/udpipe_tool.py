from __future__ import annotations

from typing import Any, Dict

from app.core.tool_registry import has_module, unavailable_result


def udpipe_analyze(text: str) -> Dict[str, Any]:
    if not (has_module("ufal.udpipe") or has_module("ufal")):
        return unavailable_result("udpipe", "Optional UDPipe unavailable: missing ufal.udpipe package.", text)
    return unavailable_result("udpipe", "UDPipe dependency detected, but Arabic model path is not configured.", text)


class UDPipeTool:
    tool_name = "udpipe"

    def analyze(self, text: str) -> Dict[str, Any]:
        return udpipe_analyze(text)

    def is_loaded(self) -> bool:
        return has_module("ufal.udpipe") or has_module("ufal")
