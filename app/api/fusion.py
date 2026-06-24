from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.startup import cached_analyze, camel_analyze, farasa_analyze, stanza_analyze, qalsadi_analyze, run_all_tools
from app.services.fusion_service import fusion_system

router = APIRouter()


@router.get("/fusion")
def fusion_endpoint(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)
    fused = fusion_system(text, camel_res, stanza_res, farasa_res, qalsadi_res)
    return {"input": text, "qalsadi": qalsadi_res, "fusion_result": fused}

