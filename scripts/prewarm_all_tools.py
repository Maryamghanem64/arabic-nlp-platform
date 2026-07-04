from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ARABERT_FORCE_FULL_LOAD", "1")

from app.core.startup import DEMO_SAMPLE_TEXT, warm_up_all_tools, get_memory_report, get_tool_statuses
from app.core.tool_registry import ALL_TOOLS


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
    print("Arabic NLP demo prewarm")
    print(f"sample: {DEMO_SAMPLE_TEXT}")

    before_statuses = get_tool_statuses()
    results = warm_up_all_tools(DEMO_SAMPLE_TEXT)
    after_statuses = get_tool_statuses()
    memory = get_memory_report()

    print(f"RAM available MB: {memory.get('available_mb')}")
    print(f"mode: {memory.get('mode')} low_memory_guards_enabled={memory.get('low_memory_guards_enabled')}")
    print()
    print(f"{'tool':12s} {'status':18s} {'tokens':8s} {'runtime_ms':10s} reason")
    print("-" * 90)

    all_ok = True
    missing_report: Dict[str, Any] = {}
    for tool in ALL_TOOLS:
        payload = results.get(tool, {})
        status = after_statuses.get(tool, before_statuses.get(tool, {}))
        tool_status = payload.get("status", "missing_result")
        tokens_yes = "yes" if _token_output(payload) else "no"
        runtime_ms = payload.get("runtime_ms")
        reason = payload.get("reason") or status.get("reason") or ""
        missing = _missing_resources(status)
        if missing:
            missing_report[tool] = missing
        if tool_status != "ok":
            all_ok = False

        print(f"{tool:12s} {tool_status:18s} {tokens_yes:8s} {str(runtime_ms):10s} {reason}")

    if missing_report:
        print()
        print("missing resources:")
        print(json.dumps(missing_report, ensure_ascii=False, indent=2))

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
