from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

from app.core.tool_registry import PROJECT_ROOT, unavailable_result


def _madamira_home() -> Path:
    configured = os.environ.get("MADAMIRA_HOME")
    return Path(configured) if configured else PROJECT_ROOT / "tools" / "madamira"


def madamira_analyze(text: str) -> Dict[str, Any]:
    if not shutil.which("java"):
        return unavailable_result("madamira", "Optional MADAMIRA unavailable: Java executable is missing from PATH.", text)
    home = _madamira_home()
    if not home.exists():
        return unavailable_result("madamira", f"Optional MADAMIRA unavailable: MADAMIRA_HOME not found at {home}.", text)
    return unavailable_result("madamira", "MADAMIRA files detected, but adapter execution is not configured.", text)


class MADAMIRATool:
    tool_name = "madamira"

    def analyze(self, text: str) -> Dict[str, Any]:
        return madamira_analyze(text)

    def is_loaded(self) -> bool:
        return shutil.which("java") is not None and _madamira_home().exists()
