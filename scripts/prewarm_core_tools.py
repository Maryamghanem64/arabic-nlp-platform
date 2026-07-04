from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Ensure no heavy optional model loads during this script.
# AraBERT/MADAMIRA/SinaTools are intentionally NOT preloaded.
os.environ.setdefault("ARABERT_FORCE_FULL_LOAD", "0")

from app.core.startup import (
    DEMO_SAMPLE_TEXT,
    analyze_tool,
    get_lightweight_tool_statuses,
    get_memory_report,
)
from app.core.tool_registry import ALL_TOOLS


CORE_TOOLS = ["camel", "farasa", "stanza", "qalsadi", "alkhalil", "udpipe"]


def _token_output(payload: Dict[str, Any]) -> bool:
    tokens = payload.get("tokens")
    return isinstance(tokens, list) and len(tokens) > 0


def _missing_resources(status: Dict[str, Any]) -> list[Any]:
    missing = status.get("missing")
    if isinstance(missing, list) and missing:
        return missing

    required = status.get("required_resources")
    if isinstance(required, dict):
        return [name for name, found in required.items() if not found]

    return []


def main() -> int:
    print("Prewarm core tools (no MADAMIRA, no SinaTools, no AraBERT full load)")
    print(f"sample: {DEMO_SAMPLE_TEXT}")

    before = get_lightweight_tool_statuses()
    results: Dict[str, Dict[str, Any]] = {}

    for tool in CORE_TOOLS:
        try:
            print(f"[prewarm] {tool}...")
            results[tool] = analyze_tool(tool, DEMO_SAMPLE_TEXT, statuses=None, use_cache=True)
        except Exception as exc:
            results[tool] = {
                "tool": tool,
                "status": "error",
                "reason": str(exc),
                "tokens": [],
            }

    after = get_lightweight_tool_statuses()
    memory = get_memory_report()

    print(f"RAM available MB: {memory.get('available_mb')}")
    print(f"mode: {memory.get('mode')} low_memory_guards_enabled={memory.get('low_memory_guards_enabled')}")

    print()
    print(f"{'tool':10s} {'status':12s} {'tokens':7s} {'runtime_ms':12s} reason")
    print("-" * 90)

    all_ok = True
    missing_report: Dict[str, Any] = {}

    for tool in CORE_TOOLS:
        payload = results.get(tool, {})
        status = after.get(tool, before.get(tool, {}))
        tool_status = payload.get("status", status.get("status", "missing_result"))
        tokens_yes = "yes" if _token_output(payload) else "no"
        runtime_ms = payload.get("runtime_ms")
        reason = payload.get("reason") or status.get("reason") or ""
        missing = _missing_resources(status)
        if missing:
            missing_report[tool] = missing
        if tool_status != "ok":
            all_ok = False

        print(
            f"{tool:10s} {str(tool_status):12s} {tokens_yes:7s} {str(runtime_ms):12s} {reason}"
        )

    if missing_report:
        print()
        print("missing resources:")
        print(json.dumps(missing_report, ensure_ascii=False, indent=2))

    # Return non-zero if any core tool did not reach ok.
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

