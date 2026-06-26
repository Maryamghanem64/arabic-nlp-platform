from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.startup import run_all_registered_tools
from app.services.fusion_service import fusion_system
from backend.schemas.unified_schema import AnalysisEnvelope

router = APIRouter()


def _dump_envelope(payload: AnalysisEnvelope) -> dict:
    return payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()


@router.get("/fusion")
def fusion_endpoint(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    all_tool_results = run_all_registered_tools(text)
    camel_res = all_tool_results.get("camel", {})
    farasa_res = all_tool_results.get("farasa", {})
    stanza_res = all_tool_results.get("stanza", {})
    qalsadi_res = all_tool_results.get("qalsadi", {})
    fused = fusion_system(text, camel_res, stanza_res, farasa_res, qalsadi_res=qalsadi_res, all_tool_results=all_tool_results)
    active_tools = sorted(
        [tool for tool, payload in all_tool_results.items() if isinstance(payload, dict) and payload.get("status") == "ok"]
    )
    envelope = AnalysisEnvelope(
        input=text,
        tools=all_tool_results,
        fusion=fused.get("fusion", []),
        active_tools=active_tools,
        meta={
            "tool_count": len(all_tool_results),
            "fusion_tokens": len(fused.get("fusion", [])),
            "fusion_base_tool": (fused.get("meta") or {}).get("base_tool"),
            "degraded_tools": [tool for tool, payload in all_tool_results.items() if isinstance(payload, dict) and payload.get("status") != "ok"],
        },
    )
    return _dump_envelope(envelope)

