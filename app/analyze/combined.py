from __future__ import annotations

from typing import Any, Dict

from app.services.merger_service import run_all_tools


def analyze_combined(text: str) -> Dict[str, Any]:
    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)
    return {"input": text, "camel": camel_res, "farasa": farasa_res, "stanza": stanza_res, "qalsadi": qalsadi_res}

