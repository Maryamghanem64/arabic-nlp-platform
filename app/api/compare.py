from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core.startup import run_all_registered_tools
from backend.services.alignment_engine import align_tools
from backend.services.comparison_service import build_conflicts
from backend.services.normalizer import normalize_tool_output

router = APIRouter()

VALID_COMPARE_TOOLS = {"camel", "farasa", "stanza", "qalsadi", "alkhalil", "udpipe"}


def _parse_tools(tools: str) -> list[str]:
    return [t.strip().lower() for t in tools.split(",") if t.strip()]


@router.get("/compare")
def compare(text: str, tools: str = Query("camel,farasa,stanza,qalsadi,alkhalil,udpipe")):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    requested = [tool for tool in _parse_tools(tools) if tool in VALID_COMPARE_TOOLS]
    if not requested:
        raise HTTPException(400, "No supported tools requested")

    all_results = run_all_registered_tools(text)
    normalized = {name: normalize_tool_output(name, payload) for name, payload in all_results.items() if name in VALID_COMPARE_TOOLS}

    present_tools = [tool for tool in requested if tool in normalized and (normalized[tool].get("tokens") or [])]
    base_tool = "farasa" if "farasa" in present_tools else next((tool for tool in requested if tool in present_tools), None)
    if base_tool is None:
        base_tool = "farasa" if "farasa" in normalized else next((tool for tool in requested if tool in normalized), None)
    if base_tool is None:
        raise HTTPException(503, "No compare-capable tool returned tokens")

    base_tokens = normalized.get(base_tool, {}).get("tokens", []) or []
    tools_tokens = {tool: normalized.get(tool, {}).get("tokens", []) or [] for tool in requested if tool in normalized}

    aligned, _meta = align_tools(base_tokens=base_tokens, tools_tokens=tools_tokens)

    comparison = []
    for index, row in enumerate(aligned):
        row_conflicts = build_conflicts(
            camel_tok=row.tools.get("camel"),
            stanza_tok=row.tools.get("stanza"),
            qalsadi_tok=row.tools.get("qalsadi"),
            alkhalil_tok=row.tools.get("alkhalil"),
            udpipe_tok=row.tools.get("udpipe"),
        )
        comparison.append(
            {
                "index": index,
                "word": row.base.get("surface") or f"#{index + 1}",
                "tools": {tool: row.tools.get(tool) or {} for tool in tools_tokens.keys()},
                "conflicts": row_conflicts,
            }
        )

    return {
        "input": text,
        "comparison": comparison,
        "active_tools": present_tools,
        "normalized_tools": {tool: normalized.get(tool, {}) for tool in requested},
        "base_tool": base_tool,
    }
