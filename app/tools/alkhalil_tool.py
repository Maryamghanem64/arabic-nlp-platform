from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from app.utils.logger import logger
from backend.config.tool_paths import AlKhalilPaths


alkhalil_jar_path: Optional[str] = None
_alkhalil_paths = AlKhalilPaths()



def load_alkhalil() -> None:
    """Resolve AlKhalil JAR via centralized resolver.

    Keeps backward compatibility with ALKHALIL_JAR env override and legacy
    jar casing/locations.
    """
    global alkhalil_jar_path
    existing = _alkhalil_paths.resolved_existing()
    if existing:
        alkhalil_jar_path = str(existing)
        logger.info("✅ AlKhalil JAR found: %s", alkhalil_jar_path)
        return

    alkhalil_jar_path = None
    logger.warning("⚠️ AlKhalil JAR not found (resolved_existing returned None).")



def _run_alkhalil(text: str) -> str:
    if not alkhalil_jar_path:
        return ""
    try:
        proc = subprocess.run(
            ["java", "-jar", alkhalil_jar_path],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        return proc.stdout or ""
    except subprocess.TimeoutExpired:
        return ""
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _parse_alkhalil_lines(stdout: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Accept pipe-separated or space-separated formats.
        # Expected: word|lemma|pos|root
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            parts = line.split()

        if len(parts) >= 4:
            word, lemma, pos, root = parts[0], parts[1], parts[2], parts[3]
            tokens.append(
                {
                    "surface": word,
                    "lemma": lemma if lemma and lemma != "_" else None,
                    "pos": pos if pos and pos != "_" else None,
                    "root": root if root and root != "_" else None,
                    "gloss": None,
                }
            )
        elif len(parts) == 1:
            tokens.append(
                {
                    "surface": parts[0],
                    "lemma": None,
                    "pos": None,
                    "root": None,
                    "gloss": None,
                }
            )

    return tokens


def alkhalil_analyze(text: str) -> Dict[str, Any]:
    tool = "alkhalil"
    try:
        global alkhalil_jar_path
        if alkhalil_jar_path is None:
            load_alkhalil()

        jar_path = alkhalil_jar_path
        if jar_path is None:
            resolved = _alkhalil_paths.resolve()
            return {
                "tool": tool,
                "status": "unavailable",
                "reason": f"AlKhalil JAR not found. Expected at: {resolved}",
                "input": text,
                "word_count": 0,
                "tokens": [],
            }


        if not shutil.which("java"):
            return {
                "tool": tool,
                "status": "unavailable",
                "reason": "Java not in PATH",
                "input": text,
                "word_count": 0,
                "tokens": [],
            }

        # Keep output deterministic: set env ALKHALIL_JAR so any downstream logic
        # that expects this env var will see the resolved jar.
        if os.environ.get("ALKHALIL_JAR") is None:
            os.environ["ALKHALIL_JAR"] = jar_path

        stdout = _run_alkhalil(text or "")

        if not stdout.strip():
            return {
                "tool": tool,
                "status": "error",
                "reason": "AlKhalil produced empty output",
                "input": text,
                "word_count": 0,
                "tokens": [],
            }

        tokens = _parse_alkhalil_lines(stdout)
        if not tokens:
            return {
                "tool": tool,
                "status": "error",
                "reason": "AlKhalil output unparseable",
                "input": text,
                "word_count": 0,
                "tokens": [],
            }

        return {
            "tool": tool,
            "status": "ok",
            "reason": "",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
        }

    except Exception as e:
        logger.exception("[AlKhalil] error")
        return {
            "tool": tool,
            "status": "error",
            "reason": str(e),
            "input": text,
            "word_count": 0,
            "tokens": [],
        }

