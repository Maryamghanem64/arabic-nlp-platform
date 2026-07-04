from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.core.startup import (
    analyze_tool,
    get_demo_tool_health,
    get_lightweight_health,
    get_memory_report,
    run_all_registered_tools,
)
from app.core.tool_registry import ALL_TOOLS, log_startup_report
from app.models.api_response import dump_envelope, success_response
from backend.schemas.unified_schema import AnalysisEnvelope
from app.tools.sinatools_tool import get_sinatools_status_detail, start_sinatools_background_loading

router = APIRouter()


def _madamira_excluded(text: str = "") -> Dict[str, Any]:
    return {
        "tool": "madamira",
        "status": "excluded",
        "reason": "Missing licensed resources",
        "input": text,
        "word_count": 0,
        "tokens": [],
        "lemmas": [],
        "pos": [],
        "runtime_ms": 0,
    }


@router.get("/")
def root():
    health = get_lightweight_health()
    return success_response({
        "platform": "Arabic NLP Comparative Platform",
        "version": "8.3",
        **health,
        "endpoints": [
            "GET /",
            "GET /health",
            "GET /tools/status",
            "POST /tools/sinatools/preload",
            "GET /health/demo-tools?run_sample=false",
            "GET /analyze/{tool}?text=...",
            "GET /analyze-combined?text=...",
            "GET /fusion?text=...",
            "GET /evaluate?text=...",
        ],
    }, message="Platform status loaded")


@router.get("/health")
def health():
    return success_response(get_lightweight_health(), message="Backend health loaded")


@router.get("/health/tools")
def health_tools():
    return success_response({"tools": get_lightweight_health()["tools"]}, message="Tool health loaded")


@router.get("/health/demo-tools")
def health_demo_tools(run_sample: bool = True):
    return success_response(
        {
            "tools": get_demo_tool_health(run_sample=run_sample),
            "memory": get_memory_report(),
        },
        message="Demo tool health loaded",
    )


@router.post("/health/startup-report")
def startup_report():
    statuses = log_startup_report()
    return success_response({"tools": statuses}, message="Startup report generated")


@router.post("/tools/sinatools/preload")
def preload_sinatools() -> Dict[str, Any]:
    start_sinatools_background_loading()
    return success_response(
        {"tool": "sinatools", **get_sinatools_status_detail()},
        message="SinaTools preload started",
    )


@router.get("/tools/status")
def tools_status() -> Dict[str, Any]:
    base = get_lightweight_health()
    tools = dict(base.get("tools", {}))
    sinatools_detail = get_sinatools_status_detail()

    if sinatools_detail.get("loading"):
        sinatools_status = "loading"
    elif sinatools_detail.get("loaded"):
        sinatools_status = "loaded"
    else:
        sinatools_status = sinatools_detail.get("status") or "lazy_not_loaded"

    tools["sinatools"] = {
        **tools.get("sinatools", {}),
        "tool": "sinatools",
        "status": sinatools_status,
        "reason": sinatools_detail.get("reason"),
        "loaded": bool(sinatools_detail.get("loaded")),
        "loading": bool(sinatools_detail.get("loading")),
        "lazy": sinatools_status in {"lazy", "lazy_not_loaded"},
        "excluded": False,
        "model_present": sinatools_detail.get("model_present"),
        "model_path": sinatools_detail.get("model_path"),
        "last_error": sinatools_detail.get("last_error"),
        "runtime_ms": sinatools_detail.get("runtime_ms"),
        "progress_label": sinatools_detail.get("progress_label"),
    }
    tools["madamira"] = {
        **tools.get("madamira", {}),
        "tool": "madamira",
        "status": "excluded",
        "excluded": True,
        "loaded": False,
        "loading": False,
        "reason": "Missing licensed resources",
    }

    return success_response({**base, "tools": tools}, message="Tool status loaded")


@router.get("/analyze/arabert")
def analyze_arabert(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    return success_response(analyze_tool("arabert", text), message="AraBERT analysis completed")


@router.get("/analyze/alkhalil")
def analyze_alkhalil(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    return success_response(analyze_tool("alkhalil", text), message="AlKhalil analysis completed")


@router.get("/analyze/udpipe")
def analyze_udpipe(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    return success_response(analyze_tool("udpipe", text), message="UDPipe analysis completed")


@router.get("/analyze/sinatools")
def analyze_sinatools(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    return success_response(analyze_tool("sinatools", text, use_cache=False), message="SinaTools analysis completed")


@router.get("/analyze/madamira")
def analyze_madamira(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    return success_response(_madamira_excluded(text), message="MADAMIRA excluded")


@router.get("/analyze/{tool}")
def analyze_by_tool(tool: str, text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")
    tool = tool.strip().lower()
    if tool == "madamira":
        return success_response(_madamira_excluded(text), message="MADAMIRA excluded")
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

    results = run_all_registered_tools(text)
    results["madamira"] = _madamira_excluded(text)

    envelope = AnalysisEnvelope(
        input=text,
        tools=results,
        active_tools=sorted([
            name for name, payload in results.items()
            if isinstance(payload, dict) and payload.get("status") in {"ok", "partial", "loaded"}
        ]),
        meta={
            "active_tools": sorted([
                name for name, payload in results.items()
                if isinstance(payload, dict) and payload.get("status") in {"ok", "partial", "loaded"}
            ]),
            "tool_count": len(results),
        },
    )
    return success_response(dump_envelope(envelope), message="Combined analysis completed")
