from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.startup import analyze_tool, get_tool_statuses, run_all_registered_tools
from app.core.tool_registry import ALL_TOOLS, log_startup_report
from app.models.api_response import dump_envelope, success_response
from backend.schemas.unified_schema import AnalysisEnvelope

router = APIRouter()


@router.get("/")
def root():
    statuses = get_tool_statuses()
    return success_response({
        "platform": "Arabic NLP Comparative Platform",
        "version": "8.3",
        "tools": statuses,
        "tool_status": {name: payload.get("status", "unknown") for name, payload in statuses.items()},
        "endpoints": [
            "GET /",
            "GET /analyze/{tool}?text=...",
            "GET /analyze-combined?text=...",
            "GET /fusion?text=...",
            "GET /evaluate?text=...",
            "GET /evaluate/dataset",
            "GET /export?text=...&format=json|csv",
            "POST /cache/clear",
        ],
    }, message="Platform status loaded")


@router.get("/health/tools")
def health_tools():
    return success_response({"tools": get_tool_statuses()}, message="Tool health loaded")


@router.post("/health/startup-report")
def startup_report():
    statuses = log_startup_report()
    return success_response({"tools": statuses}, message="Startup report generated")


@router.get("/analyze/arabert")
def analyze_arabert(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    from app.core.tool_registry import cached_analyze
    from app.tools.arabert_tool import arabert_analyze
    return success_response(cached_analyze(arabert_analyze, text), message="AraBERT analysis completed")


@router.get("/analyze/alkhalil")
def analyze_alkhalil(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    from app.core.tool_registry import cached_analyze
    from app.tools.alkhalil_tool import alkhalil_analyze
    return success_response(cached_analyze(alkhalil_analyze, text), message="AlKhalil analysis completed")


@router.get("/analyze/udpipe")
def analyze_udpipe(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    from app.core.tool_registry import cached_analyze
    from app.tools.udpipe_tool import udpipe_analyze
    return success_response(cached_analyze(udpipe_analyze, text), message="UDPipe analysis completed")


@router.get("/analyze/madamira")
def analyze_madamira(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    from app.tools.madamira_tool import madamira_analyze
    return success_response(madamira_analyze(text), message="MADAMIRA analysis completed")


@router.get("/analyze/{tool}")
def analyze_by_tool(tool: str, text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    tool = tool.strip().lower()
    if tool not in ALL_TOOLS:
        return success_response({
            "tool": tool,
            "status": "unavailable",
            "reason": f"Unknown tool. Available tools: {', '.join(ALL_TOOLS)}",
            "input": text,
            "word_count": 0,
            "tokens": [],
        }, message="Unknown analysis tool")
    return success_response(analyze_tool(tool, text), message=f"{tool} analysis completed")


@router.get("/analyze-combined")
def analyze_combined(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    # Run all registered tools in parallel (includes arabert/alkhalil/udpipe/madamira)
    results = run_all_registered_tools(text)
    envelope = AnalysisEnvelope(
        input=text,
        tools=results,
        active_tools=sorted([name for name, payload in results.items() if isinstance(payload, dict) and payload.get("status") == "ok"]),
        meta={
            "active_tools": sorted([name for name, payload in results.items() if isinstance(payload, dict) and payload.get("status") == "ok"]),
            "tool_count": len(results),
        },
    )
    return success_response(dump_envelope(envelope), message="Combined analysis completed")

