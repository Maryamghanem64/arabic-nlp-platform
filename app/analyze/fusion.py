from __future__ import annotations

from typing import Any, Dict

from app.core.startup import run_all_registered_tools
from app.services.fusion_service import fusion_system


def fusion_for_text(
    text: str,
    camel_res: Dict[str, Any],
    stanza_res: Dict[str, Any],
    farasa_res: Dict[str, Any],
    qalsadi_res: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    _ = (camel_res, stanza_res, farasa_res, qalsadi_res)
    all_tool_results = run_all_registered_tools(text)
    return fusion_system(
        text,
        all_tool_results.get("camel", {}),
        all_tool_results.get("stanza", {}),
        all_tool_results.get("farasa", {}),
        qalsadi_res=all_tool_results.get("qalsadi", {}),
        all_tool_results=all_tool_results,
    )

