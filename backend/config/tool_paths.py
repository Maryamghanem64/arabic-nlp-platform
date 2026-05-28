from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Single source of truth for tool model/binary paths.
# Uses pathlib.Path everywhere and keeps Windows compatibility.


@dataclass(frozen=True)
class AlKhalilPaths:
    jar_filename: str = "AlKhalil.jar"
    rel_jar_path: Path = Path("app") / "tools" / "alkhalil" / "AlKhalil1.1" / "AlKhalil.jar"
    legacy_rel_jar_paths: tuple[Path, ...] = (
        Path("tools") / "alkhalil" / "alkhalil.jar",
        Path("app") / "tools" / "alkhalil" / "alkhalil.jar",
        Path("app") / "tools" / "alkhalil" / "AlKhalil1.1" / "Alkhalil.jar",  # historical wrong-cased filename
    )

    def resolve(self) -> Path:
        project_root = Path(__file__).resolve().parents[2]

        configured = os.environ.get("ALKHALIL_JAR")
        if configured:
            return Path(configured)

        default = project_root / self.rel_jar_path
        return default

    def resolved_existing(self) -> Optional[Path]:
        project_root = Path(__file__).resolve().parents[2]

        configured = os.environ.get("ALKHALIL_JAR")
        if configured:
            p = Path(configured)
            return p if p.exists() and p.is_file() else None

        default = project_root / self.rel_jar_path
        if default.exists() and default.is_file():
            return default

        for rel in self.legacy_rel_jar_paths:
            p = project_root / rel
            if p.exists() and p.is_file():
                return p

        return None


@dataclass(frozen=True)
class UDPipePaths:
    model_filename: str = "ar-ud-udpipe.udpipe"
    rel_default_model_path: Path = Path("app") / "tools" / "udpipe" / "ar-ud-ud-udpipe.udpipe"
    legacy_rel_default_model_paths: tuple[Path, ...] = (
        Path("tools") / "udpipe" / "ar-ud-udpipe.udpipe",
    )

    def resolve(self) -> Path:
        project_root = Path(__file__).resolve().parents[2]

        configured = os.environ.get("UDPIP_MODEL")
        if configured:
            return Path(configured)

        return project_root / self.rel_default_model_path

    def resolved_existing(self) -> Optional[Path]:
        project_root = Path(__file__).resolve().parents[2]

        configured = os.environ.get("UDPIP_MODEL")
        if configured:
            p = Path(configured)
            return p if p.exists() and p.is_file() else None

        default = project_root / self.rel_default_model_path
        if default.exists() and default.is_file():
            return default

        for rel in self.legacy_rel_default_model_paths:
            p = project_root / rel
            if p.exists() and p.is_file():
                return p

        return None


alkhalil_paths = AlKhalilPaths()
udpipe_paths = UDPipePaths()

