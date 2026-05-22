from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

from app.core.tool_registry import PROJECT_ROOT, unavailable_result


def _jar_path() -> Path:
    configured = os.environ.get("ALKHALIL_JAR")
    return Path(configured) if configured else PROJECT_ROOT / "tools" / "alkhalil" / "alkhalil.jar"


def alkhalil_analyze(text: str) -> Dict[str, Any]:
    if not shutil.which("java"):
        return unavailable_result("alkhalil", "Optional AlKhalil unavailable: Java executable is missing from PATH.", text)
    jar_path = _jar_path()
    if not jar_path.exists():
        return unavailable_result("alkhalil", f"Optional AlKhalil unavailable: JAR file not found at {jar_path}.", text)
    return unavailable_result("alkhalil", "AlKhalil JAR detected, but adapter execution is not configured.", text)


class AlKhalilTool:
    tool_name = "alkhalil"

    def analyze(self, text: str) -> Dict[str, Any]:
        return alkhalil_analyze(text)

    def is_loaded(self) -> bool:
        return shutil.which("java") is not None and _jar_path().exists()
