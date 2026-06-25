from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.startup import run_all_registered_tools
from app.services.fusion_service import fusion_system

router = APIRouter()


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
    return {"input": text, "qalsadi": qalsadi_res, "fusion_result": fused, "tools": all_tool_results}

