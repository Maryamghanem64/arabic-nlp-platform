from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ToolMeta:
    tool_type: str
    strengths: list[str]
    weaknesses: list[str]
    supported_tasks: list[str]
    confidence_weight: float


class Settings:
    def __init__(self, root_dir: Path):
        self._root_dir = root_dir
        self._tool_metadata_path = root_dir / "backend" / "config" / "tool_metadata.json"

        if not self._tool_metadata_path.exists():
            raise FileNotFoundError(f"tool_metadata.json not found at: {self._tool_metadata_path}")

        raw = json.loads(self._tool_metadata_path.read_text(encoding="utf-8"))
        self.tool_metadata: Dict[str, ToolMeta] = {
            tool: ToolMeta(**meta) for tool, meta in raw.items()
        }


def get_settings() -> Settings:
    # project root: c:/Users/.../arabic-nlp-platform
    root_dir = Path(__file__).resolve().parents[2]
    return Settings(root_dir=root_dir)

