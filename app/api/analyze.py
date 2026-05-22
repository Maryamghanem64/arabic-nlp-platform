from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.startup import cached_analyze, camel_analyze, farasa_analyze, stanza_analyze, qalsadi_analyze, run_all_tools

from backend.services.normalizer import normalize_tool_output

router = APIRouter()


@router.get("/")
def root():
    # Keep JSON shape identical to the original monolithic main.py.
    try:
        from backend.analyzers import get_all_partner_statuses, PARTNER_TOOLS  # type: ignore

        partner_status = get_all_partner_statuses()
        partner_loaded = True
    except Exception:
        partner_status = None
        partner_loaded = False

    # Use resource-loading globals from startup.
    return {
        "platform": "Arabic NLP Comparative Platform",
        "version": "8.3",
        "tools": {
            "camel": {"status": "ok" if camel_analyze else "failed"},
            "farasa": {"status": "ok" if farasa_analyze else "failed"},
            "stanza": {"status": "ok" if stanza_analyze else "failed"},
            "qalsadi": {"status": "ok" if qalsadi_analyze else "failed"},
        },
        "endpoints": [
            "GET /analyze/camel?text=...",
            "GET /analyze/farasa?text=...",
            "GET /analyze/stanza?text=...",
            "GET /analyze/qalsadi?text=...",
            "GET /analyze/{tool}?text=...",
            "GET /analyze-combined?text=...",
            "GET /compare?text=...&tools=camel,farasa,stanza,qalsadi",
            "GET /fusion?text=...",
            "GET /evaluate?text=...",
            "GET /export?text=...&format=json|csv",
            "POST /cache/clear",
        ],
    }


@router.get("/analyze/camel")
def analyze_camel(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    raw = cached_analyze(camel_analyze, text)
    try:
        return normalize_tool_output("camel", raw)
    except Exception as e:
        return {"tool": "camel", "status": "error", "input": text, "word_count": 0, "tokens": [], "error": str(e)}


@router.get("/analyze/farasa")
def analyze_farasa(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    raw = cached_analyze(farasa_analyze, text)
    try:
        return normalize_tool_output("farasa", raw)
    except Exception as e:
        return {"tool": "farasa", "status": "error", "input": text, "word_count": 0, "tokens": [], "error": str(e)}


@router.get("/analyze/stanza")
def analyze_stanza(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    return cached_analyze(stanza_analyze, text)


@router.get("/analyze/qalsadi")
def analyze_qalsadi(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    return cached_analyze(qalsadi_analyze, text)


@router.get("/analyze/{tool}")
def analyze_by_tool(tool: str, text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    tool = tool.strip().lower()

    if tool == "camel":
        return cached_analyze(camel_analyze, text)
    if tool == "farasa":
        return cached_analyze(farasa_analyze, text)
    if tool == "stanza":
        return cached_analyze(stanza_analyze, text)
    if tool == "qalsadi":
        return cached_analyze(qalsadi_analyze, text)

    raise HTTPException(404, "Tool not found. Available: camel, farasa, stanza, qalsadi")


@router.get("/analyze-combined")
def analyze_combined(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)
    return {"input": text, "camel": camel_res, "farasa": farasa_res, "stanza": stanza_res, "qalsadi": qalsadi_res}

