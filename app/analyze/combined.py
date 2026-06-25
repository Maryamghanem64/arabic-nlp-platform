from __future__ import annotations

from typing import Any, Dict

from app.core.startup import run_all_registered_tools


def analyze_combined(text: str) -> Dict[str, Any]:
    results = run_all_registered_tools(text)
    return {"input": text, **results}

