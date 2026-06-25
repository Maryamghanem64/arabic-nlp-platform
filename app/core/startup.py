from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict

from app.core.tool_registry import ALL_TOOLS, CORE_TOOLS, detect_tool_status, safe_analyze, unavailable_result
from app.services.cache_service import cached_analyze, clear_cache
from app.services.eval_service import evaluate_tools
from app.services.fusion_service import fusion_system
from app.tools.alkhalil_tool import alkhalil_analyze
from app.tools.arabert_tool import arabert_analyze, load_arabert
from app.tools.camel_tool import camel_analyze
from app.tools.farasa_tool import farasa_analyze
from app.tools.madamira_tool import madamira_analyze, load_madamira
from app.tools.qalsadi_tool import qalsadi_analyze
from app.tools.sinatools_tool import sinatools_analyze
from app.tools.stanza_tool import stanza_analyze
from app.tools.udpipe_tool import udpipe_analyze, load_udpipe


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
    if status.get("status") not in (None, "ok") and tool not in {"sinatools", "alkhalil"}:
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


import threading


# Singleflight implementation for run_all_tools(text)
# Ensures:
# - Exactly one leader executes run_core_tools() per unique text
# - Followers wait without busy-waiting
# - Followers get the exact same result object
# - Exceptions propagate to all waiters

_inflight_run_lock = threading.Lock()
_inflight_run_state: dict[str, "_SingleflightState"] = {}


class _SingleflightState:
    __slots__ = ("event", "result", "exc")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result = None
        self.exc = None


def run_all_tools(text: str):
    """Run core tools (camel, farasa, stanza, qalsadi) with singleflight dedupe.

    Thread-safe across concurrent FastAPI requests.
    """
    key = f"run_all_tools::{text}"

    with _inflight_run_lock:
        state = _inflight_run_state.get(key)
        if state is None:
            state = _SingleflightState()
            _inflight_run_state[key] = state
            leader = True
        else:
            leader = False

    if not leader:
        state.event.wait()
        if state.exc is not None:
            raise state.exc
        packed = state.result
        # packed is a tuple(camel, farasa, stanza, qalsadi)
        return packed[0], packed[1], packed[2], packed[3]

    try:
        results = run_core_tools(text)
        state.result = (
            results["camel"],
            results["farasa"],
            results["stanza"],
            results["qalsadi"],
        )
        return state.result[0], state.result[1], state.result[2], state.result[3]
    except BaseException as e:
        state.exc = e
        raise
    finally:
        # Set event only after result/exception is stored.
        state.event.set()
        # Cleanup AFTER all followers can safely read state.result/state.exc.
        # The state remains accessible to followers even after set(), because we wait on event.
        # Remove from map after releasing leader lock section to minimize contention.
        with _inflight_run_lock:
            # Another follower may already have captured `state` reference; it still points to the state object.
            _inflight_run_state.pop(key, None)





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
