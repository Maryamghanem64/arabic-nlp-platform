from __future__ import annotations

import sys

from app.core.tool_registry import detect_tool_status, startup_report_lines


def main() -> int:
    statuses = detect_tool_status()
    for line in startup_report_lines(statuses):
        print(line)

    required = ("camel", "farasa", "stanza", "qalsadi")
    missing_required = [name for name in required if statuses.get(name, {}).get("status") != "ok"]
    if missing_required:
        print()
        print("[WARN] Some core tools are not fully ready on this machine:")
        for name in missing_required:
            info = statuses[name]
            print(f"  - {name}: {info.get('status')} - {info.get('reason')}")
        print("The backend will still start and return safe fallback responses.")
        return 0

    print()
    print("[OK] Core toolchain is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
