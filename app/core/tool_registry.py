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


def unavailable_result(tool: str, reason: str, text: str = "") -> Dict[str, Any]:
    return {
        "tool": tool,
        "status": "unavailable",
        "reason": reason,
        "input": text,
        "word_count": 0,
        "tokens": [],
    }


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
    stanza_home = Path(os.environ.get("STANZA_RESOURCES_DIR", Path.home() / "stanza_resources"))
    return (stanza_home / "ar").exists()


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
    java = java_status()
    statuses: Dict[str, Dict[str, Any]] = {}

    statuses["camel"] = (
        {"status": "ok", "reason": "camel-tools package detected."}
        if has_module("camel_tools")
        else {"status": "missing_dependency", "reason": "Install camel-tools."}
    )

    if has_module("farasa"):
        statuses["farasa"] = (
            {"status": "ok", "reason": "farasapy package and Farasa binaries detected."}
            if java["status"] == "ok" and farasa_bins_present()
            else {
                "status": "missing_model" if java["status"] == "ok" else "missing_java",
                "reason": "Farasa requires Java and local Farasa JAR files.",
                "java": java,
            }
        )
    else:
        statuses["farasa"] = {"status": "missing_dependency", "reason": "Install farasapy or local Farasa package."}

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

    statuses["arabert"] = (
        {"status": "ok", "reason": "transformers and torch detected. Model download may still be needed."}
        if has_module("transformers") and has_module("torch")
        else {"status": "missing_dependency", "reason": "Optional AraBERT requires transformers and torch."}
    )

    alkhalil_path = model_path_status("ALKHALIL_JAR", "tools/alkhalil/alkhalil.jar", "missing_model")
    statuses["alkhalil"] = alkhalil_path if java["status"] == "ok" else {"status": "missing_java", "reason": "AlKhalil requires Java.", "java": java}

    statuses["udpipe"] = (
        {"status": "ok", "reason": "ufal.udpipe package detected."}
        if has_module("ufal.udpipe") or has_module("ufal")
        else {"status": "missing_dependency", "reason": "Optional UDPipe requires ufal.udpipe and Arabic model files."}
    )

    madamira_path = model_path_status("MADAMIRA_HOME", "tools/madamira", "missing_model")
    statuses["madamira"] = madamira_path if java["status"] == "ok" else {"status": "missing_java", "reason": "MADAMIRA requires Java.", "java": java}

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
        result.setdefault("tokens", [])
        result.setdefault("word_count", len(result.get("tokens", []) or []))
        return result
    except Exception as exc:
        logger.exception("[%s] safe analyzer failure", tool)
        return {
            "tool": tool,
            "status": "error",
            "reason": str(exc),
            "input": text,
            "word_count": 0,
            "tokens": [],
        }


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
