from __future__ import annotations

import time
import traceback
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.models.api_response import success_response
from app.tools.sinatools_tool import get_sinatools_status_detail
from backend.analyzers.sinatools_tool import start_sinatools_background_loading


router = APIRouter()


@router.post("/tools/sinatools/preload")
def preload_sinatools() -> Dict[str, Any]:
    """Kick off SinaTools loading in a background daemon thread and return immediately."""
    try:
        start_sinatools_background_loading()
        return success_response(
            {
                "tool": "sinatools",
                **get_sinatools_status_detail(),
            },
            message="SinaTools preload started",
        )
    except Exception as exc:
        return success_response(
            {
                "tool": "sinatools",
                "status": "error",
                "reason": str(exc),
                "last_error": str(exc),
            },
            message="Failed to start SinaTools preload",
        )


@router.get("/tools/status")
def tools_status() -> Dict[str, Any]:
    """Return status for all tools. Focus on SinaTools fields needed by the frontend."""
    # Reuse the existing lightweight status structure.
    from app.core.startup import get_lightweight_health

    base = get_lightweight_health()
    tools = base.get("tools", {})

    # Inject SinaTools detailed status
    try:
        sinatools_detail = get_sinatools_status_detail()
    except Exception:
        sinatools_detail = {"status": "error", "last_error": "Failed to read SinaTools status"}

    tools["sinatools"] = {
        **tools.get("sinatools", {}),
        "status": (
            "loading"
            if sinatools_detail.get("loading")
            else ("lazy_not_loaded" if sinatools_detail.get("status") == "lazy_not_loaded" else ("loaded" if sinatools_detail.get("loaded") else sinatools_detail.get("status")))
        ),
        "reason": sinatools_detail.get("reason"),
        "loaded": bool(sinatools_detail.get("loaded")),
        "loading": bool(sinatools_detail.get("loading")),
        "excluded": False,
        "model_present": sinatools_detail.get("model_present"),
        "model_path": sinatools_detail.get("model_path"),
        "last_error": sinatools_detail.get("last_error"),
        "runtime_ms": sinatools_detail.get("runtime_ms"),
        "progress_label": sinatools_detail.get("progress_label"),
    }

    return success_response(
        {
            **base,
            "tools": tools,
        },
        message="Tool status loaded",
    )

