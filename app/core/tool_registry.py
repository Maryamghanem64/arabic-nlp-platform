from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List

from app.utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_TOOLS = ("camel", "farasa", "stanza", "qalsadi")
OPTIONAL_TOOLS = ("arabert", "alkhalil", "udpipe", "madamira", "sinatools")
ALL_TOOLS = CORE_TOOLS + OPTIONAL_TOOLS


def unified_result(
    tool: str,
    status: str,
    tokens: List[Any] | None = None,
    lemmas: List[Any] | None = None,
    pos: List[Any] | None = None,
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "tool": tool,
        "status": status,
        "tokens": tokens or [],
        "lemmas": lemmas or [],
        "pos": pos or [],
        "reason": reason or "",
    }


def unavailable_result(tool: str, reason: str, text: str = "") -> Dict[str, Any]:
    # Keep signature for existing callers, but always return the unified schema.
    return unified_result(tool=tool, status="unavailable", tokens=[], lemmas=[], pos=[], reason=reason)


def safe_import(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"]), None
    except Exception as exc:
        return None, str(exc)


def has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def java_status() -> Dict[str, Any]:
    java = shutil.which("java")
    if not java:
        return {"status": "missing_java", "reason": "Java executable was not found in PATH."}
    try:
        proc = subprocess.run(
            [java, "-version"],
            capture_output=True,
            text=True,
            timeout=8,
            shell=False,
            encoding="utf-8",
            errors="replace",
        )
        version = (proc.stderr or proc.stdout).splitlines()[0] if (proc.stderr or proc.stdout) else "java detected"
        return {"status": "ok", "path": java, "version": version}
    except Exception as exc:
        return {"status": "java_error", "path": java, "reason": str(exc)}


def farasa_bins_present() -> bool:
    candidates = [
        PROJECT_ROOT / "Farasa_bin" / "farasapy-toolkit-bins-released" / "farasa" / "farasa_bin" / "lib" / "FarasaSegmenterJar.jar",
        PROJECT_ROOT / "Farasa_bin" / "farasapy-toolkit-bins-released" / "build" / "lib" / "farasa" / "farasa_bin" / "lib" / "FarasaSegmenterJar.jar",
    ]
    return any(path.exists() for path in candidates)


def stanza_model_present() -> bool:
    possible_paths = [
        Path.home() / "stanza_resources" / "ar",
        Path.home() / "AppData" / "Local" / "StanfordNLP" / "stanza" / "resources" / "ar",
        Path.home() / "AppData" / "Local" / "StanfordNLP" / "stanza" / "Cache" / "1.12.0" / "resources" / "ar",
    ]

    return any(path.exists() for path in possible_paths)


def model_path_status(env_name: str, default_relative: str, missing_status: str) -> Dict[str, Any]:
    configured = os.environ.get(env_name)
    path = Path(configured) if configured else PROJECT_ROOT / default_relative
    if path.exists():
        return {"status": "ok", "path": str(path)}
    return {
        "status": missing_status,
        "reason": f"{env_name} is not set and {path} does not exist.",
        "path": str(path),
    }


def detect_tool_status() -> Dict[str, Dict[str, Any]]:
    from backend.config.tool_paths import AlKhalilPaths

    jar_resolver = AlKhalilPaths()
    from app.tools.farasa_tool import get_farasa_status
    from app.tools.udpipe_tool import get_udpipe_status

    java = java_status()
    statuses: Dict[str, Dict[str, Any]] = {}

    statuses["camel"] = (
        {"status": "ok", "reason": "camel-tools package detected."}
        if has_module("camel_tools")
        else {"status": "missing_dependency", "reason": "Install camel-tools."}
    )

    farasa_status = get_farasa_status()
    if java["status"] != "ok" and farasa_status.get("status") == "ok":
        farasa_status = {
            "status": "missing_java",
            "reason": "Farasa requires Java.",
            "java": java,
            **{k: v for k, v in farasa_status.items() if k not in {"status", "reason"}},
        }
    statuses["farasa"] = farasa_status

    if has_module("stanza"):
        statuses["stanza"] = (
            {"status": "ok", "reason": "stanza package and Arabic model detected."}
            if stanza_model_present()
            else {"status": "missing_model", "reason": "Run python install_models.py to download stanza Arabic models."}
        )
    else:
        statuses["stanza"] = {"status": "missing_dependency", "reason": "Install stanza."}

    statuses["qalsadi"] = (
        {"status": "ok", "reason": "qalsadi package detected."}
        if has_module("qalsadi") and has_module("camel_tools")
        else {"status": "missing_dependency", "reason": "Install qalsadi and camel-tools."}
    )

    try:
        from app.tools import arabert_tool as _arabert_tool

        arabert_status = _arabert_tool.get_arabert_status_detail()
        statuses["arabert"] = {
            "status": arabert_status.get("status", "unknown"),
            "reason": arabert_status.get("reason", ""),
            "model_id": arabert_status.get("model_id"),
            "loaded": arabert_status.get("loaded", False),
        }
    except Exception:
        statuses["arabert"] = (
            {"status": "ok", "reason": "transformers and torch detected. Model will lazy-load on first request."}
            if has_module("transformers") and has_module("torch")
            else {"status": "missing_dependency", "reason": "Optional AraBERT requires transformers and torch."}
        )

    try:
        from app.tools.alkhalil_tool import get_alkhalil_status

        statuses["alkhalil"] = get_alkhalil_status()
    except Exception:
        alkhalil_jar = jar_resolver.resolve()
        alkhalil_file_exists = alkhalil_jar.exists() and alkhalil_jar.is_file()
        statuses["alkhalil"] = {
            "status": "ok" if java["status"] == "ok" and alkhalil_file_exists else ("missing_java" if java["status"] != "ok" else "missing_model"),
            "reason": (
                f"AlKhalil JAR detected at {alkhalil_jar}."
                if alkhalil_file_exists
                else "AlKhalil JAR not found. Set ALKHALIL_JAR or ensure the jar exists under app/tools/alkhalil/AlKhalil1.1/Alkhalil.jar."
            ),
            "resolved_jar": str(alkhalil_jar),
            "jar_exists": bool(alkhalil_file_exists),
        }

    statuses["udpipe"] = get_udpipe_status()

    if java["status"] != "ok":
        statuses["madamira"] = {"status": "disabled", "reason": "MADAMIRA requires Java.", "java": java}
    else:
        madamira_path = model_path_status("MADAMIRA_HOME", "tools/madamira", "missing_model")
        statuses["madamira"] = (
            madamira_path
            if madamira_path.get("status") == "ok"
            else {"status": "disabled", "reason": madamira_path.get("reason", "MADAMIRA not configured."), "path": madamira_path.get("path")}
        )

    statuses["sinatools"] = {
        "status": "future_work",
        "reason": "SinaTools is tracked as a future microservice because of large model size.",
    }

    return statuses


def is_available(status: Dict[str, Any]) -> bool:
    return status.get("status") == "ok"


def safe_analyze(tool: str, analyzer: Callable[[str], Dict[str, Any]], text: str) -> Dict[str, Any]:
    try:
        result = analyzer(text)
        if not isinstance(result, dict):
            return unavailable_result(tool, "Analyzer returned an invalid response.", text)

        result.setdefault("tool", tool)
        result.setdefault("status", "ok")
        result.setdefault("tokens", [])
        result.setdefault("lemmas", [])
        result.setdefault("pos", [])
        result.setdefault("reason", "")

        return {
            "tool": result.get("tool", tool),
            "status": result.get("status", "ok"),
            "tokens": result.get("tokens", []) or [],
            "lemmas": result.get("lemmas", []) or [],
            "pos": result.get("pos", []) or [],
            "reason": result.get("reason", "") or "",
        }
    except Exception as exc:
        logger.exception("[%s] safe analyzer failure", tool)
        return unified_result(tool=tool, status="error", tokens=[], lemmas=[], pos=[], reason=str(exc))


def startup_report_lines(statuses: Dict[str, Dict[str, Any]] | None = None) -> List[str]:
    statuses = statuses or detect_tool_status()
    lines = ["Arabic NLP Platform startup validation:"]
    for tool in ALL_TOOLS:
        status = statuses.get(tool, {})
        marker = "OK" if status.get("status") == "ok" else "WARN"
        reason = status.get("reason", status.get("status", "unknown"))
        lines.append(f"[{marker}] {tool.upper():10s} {status.get('status', 'unknown')} - {reason}")
    return lines


def log_startup_report() -> Dict[str, Dict[str, Any]]:
    statuses = detect_tool_status()
    for line in startup_report_lines(statuses):
        logger.info(line)
    return statuses

