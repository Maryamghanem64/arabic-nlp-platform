from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict

from app.core.tool_registry import ALL_TOOLS, CORE_TOOLS, detect_tool_status, safe_analyze, unavailable_result
from app.services.cache_service import cached_analyze, clear_cache
from app.services.eval_service import evaluate_tools
from app.services.fusion_service import fusion_system
from app.tools.alkhalil_tool import alkhalil_analyze
from app.tools.arabert_tool import arabert_analyze
from app.tools.camel_tool import camel_analyze
from app.tools.farasa_tool import farasa_analyze
from app.tools.madamira_tool import madamira_analyze
from app.tools.qalsadi_tool import qalsadi_analyze
from app.tools.sinatools_tool import sinatools_analyze
from app.tools.stanza_tool import stanza_analyze
from app.tools.udpipe_tool import udpipe_analyze


ANALYZERS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "camel": camel_analyze,
    "farasa": farasa_analyze,
    "stanza": stanza_analyze,
    "qalsadi": qalsadi_analyze,
    "arabert": arabert_analyze,
    "alkhalil": alkhalil_analyze,
    "udpipe": udpipe_analyze,
    "madamira": madamira_analyze,
    "sinatools": sinatools_analyze,
}


def analyze_tool(tool: str, text: str) -> Dict[str, Any]:
    tool = tool.strip().lower()
    analyzer = ANALYZERS.get(tool)
    if analyzer is None:
        return unavailable_result(tool, f"Unknown tool. Available tools: {', '.join(ALL_TOOLS)}", text)

    status = detect_tool_status().get(tool, {})
    if status.get("status") not in (None, "ok") and tool != "sinatools":
        return unavailable_result(tool, status.get("reason", f"{tool} is not available."), text)

    def runner(value: str) -> Dict[str, Any]:
        return safe_analyze(tool, analyzer, value)

    runner.__name__ = f"{tool}_safe_analyze"
    return cached_analyze(runner, text)


def run_core_tools(text: str) -> Dict[str, Dict[str, Any]]:
    threaded_tools = tuple(tool for tool in CORE_TOOLS if tool != "qalsadi")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(threaded_tools)) as executor:
        futures = {tool: executor.submit(analyze_tool, tool, text) for tool in threaded_tools}
        results = {tool: future.result() for tool, future in futures.items()}
    results["qalsadi"] = analyze_tool("qalsadi", text)
    return results


def run_all_registered_tools(text: str) -> Dict[str, Dict[str, Any]]:
    threaded_tools = tuple(tool for tool in ALL_TOOLS if tool != "qalsadi")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(threaded_tools), 8)) as executor:
        futures = {tool: executor.submit(analyze_tool, tool, text) for tool in threaded_tools}
        results = {tool: future.result() for tool, future in futures.items()}
    results["qalsadi"] = analyze_tool("qalsadi", text)
    return {tool: results[tool] for tool in ALL_TOOLS}


def run_all_tools(text: str):
    results = run_core_tools(text)
    return results["camel"], results["farasa"], results["stanza"], results["qalsadi"]


def get_tool_statuses() -> Dict[str, Dict[str, Any]]:
    return detect_tool_status()


__all__ = [
    "ANALYZERS",
    "analyze_tool",
    "camel_analyze",
    "farasa_analyze",
    "stanza_analyze",
    "qalsadi_analyze",
    "cached_analyze",
    "clear_cache",
    "run_all_tools",
    "run_core_tools",
    "run_all_registered_tools",
    "get_tool_statuses",
    "fusion_system",
    "evaluate_tools",
]
