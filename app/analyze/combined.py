from __future__ import annotations

from typing import Any, Dict

from app.core.startup import run_all_registered_tools


def analyze_combined(text: str) -> Dict[str, Any]:
    # Use the unified combined runner; heavy/lazy tools must self-report
    # their loading status without blocking the whole request.
    results = run_all_registered_tools(text)
    return {"input": text, **results}


