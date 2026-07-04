from __future__ import annotations

import concurrent.futures
import ctypes
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict

from app.core.tool_registry import ALL_TOOLS, CORE_TOOLS, PROJECT_ROOT, detect_tool_status, farasa_bins_present, has_module, safe_analyze, stanza_model_present, unavailable_result, unified_result
from app.utils.logger import logger
from app.services.cache_service import cached_analyze, clear_cache
from app.services.eval_service import evaluate_tools
from app.services.fusion_service import fusion_system
from app.tools.alkhalil_tool import alkhalil_analyze
from app.tools.arabert_tool import arabert_analyze, load_arabert
from app.tools.camel_tool import camel_analyze
from app.tools.farasa_tool import farasa_analyze
from app.tools.madamira_tool import madamira_analyze, load_madamira
from app.tools.qalsadi_tool import qalsadi_analyze
from app.tools.sinatools_tool import sinatools_analyze
from app.tools.stanza_tool import stanza_analyze
from app.tools.udpipe_tool import udpipe_analyze, load_udpipe


DEMO_SAMPLE_TEXT = "ذهب الولد إلى المدرسة"

ANALYZERS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "camel": camel_analyze,
    "farasa": farasa_analyze,
    "stanza": stanza_analyze,
    "qalsadi": qalsadi_analyze,
    "arabert": arabert_analyze,
    "alkhalil": alkhalil_analyze,
    "udpipe": udpipe_analyze,
    "madamira": madamira_analyze,
    "sinatools": sinatools_analyze,
}

CORE_DASHBOARD_TOOLS = ("camel", "farasa", "qalsadi", "alkhalil", "udpipe")
HEAVY_LAZY_TOOLS = {"stanza", "sinatools", "arabert"}
EXCLUDED_TOOLS = {"madamira"}

_LAST_TOOL_HEALTH_LOCK = threading.Lock()
_LAST_TOOL_HEALTH: Dict[str, Dict[str, Any]] = {
    tool: {
        "tool": tool,
        "status": "unknown",
        "loaded": False,
        "resources_found": False,
        "last_error": None,
        "runtime_ms": None,
    }
    for tool in ALL_TOOLS
}


def _excluded_result(tool: str, text: str, reason: str = "Missing licensed resources") -> Dict[str, Any]:
    return {
        "tool": tool,
        "status": "excluded",
        "reason": reason,
        "input": text,
        "word_count": 0,
        "tokens": [],
        "lemmas": [],
        "pos": [],
        "runtime_ms": 0,
    }


def analyze_tool(
    tool: str,
    text: str,
    statuses: Dict[str, Dict[str, Any]] | None = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run one analyzer without fake timeout output.

    Core analyzers are allowed to return real output or real errors.
    MADAMIRA is excluded immediately.
    SinaTools stays lazy/non-blocking inside its own adapter.
    """
    tool = tool.strip().lower()

    if tool in EXCLUDED_TOOLS:
        return _excluded_result(tool, text, "Missing licensed resources")

    analyzer = ANALYZERS.get(tool)
    if analyzer is None:
        return unavailable_result(tool, f"Unknown tool. Available tools: {', '.join(ALL_TOOLS)}", text)

    started_at = time.perf_counter()
    status_map = statuses or get_lightweight_tool_statuses()
    status = status_map.get(tool, {})

    if status.get("status") not in (None, "ok", "lazy", "loading", "lazy_not_loaded") and tool not in {"sinatools", "alkhalil", "arabert"}:
        result = unavailable_result(tool, status.get("reason", f"{tool} is not available."), text)
        _record_tool_health(tool, result, status, started_at)
        return result

    if _low_memory_guards_enabled():
        available_mb = _available_physical_memory_mb()
        threshold_mb = _low_memory_threshold_mb(tool)
        if available_mb is not None and threshold_mb is not None and available_mb < threshold_mb:
            result = _skipped_low_memory_result(
                tool,
                text,
                available_mb=available_mb,
                threshold_mb=threshold_mb,
            )
            _record_tool_health(tool, result, status, started_at)
            return result

    def runner(value: str) -> Dict[str, Any]:
        return safe_analyze(tool, analyzer, value)

    try:
        # Do not cache SinaTools lazy/loading states.
        # TEMP FIX: disable shared cache because it returns CAMeL payload for other tools.
# This avoids cross-tool cache pollution before final demo.
        result = runner(text)
    except Exception as exc:
        logger.exception("[%s] analyzer failed", tool)
        result = unified_result(tool=tool, status="error", tokens=[], lemmas=[], pos=[], reason=str(exc))
        result["input"] = text
        result["word_count"] = 0

    _record_tool_health(tool, result, status, started_at)
    return result


def run_core_tools(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Run core analyzers in parallel.

    Core tools must return real output or real error.
    No fake timeout results.
    MADAMIRA and SinaTools are not part of core execution.
    """
    statuses = get_lightweight_tool_statuses()
    core_tools = ("camel", "farasa", "stanza", "qalsadi", "alkhalil", "udpipe")

    results: Dict[str, Dict[str, Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(core_tools)) as executor:
        futures = {
            tool: executor.submit(analyze_tool, tool, text, statuses, True)
            for tool in core_tools
        }

        for tool, future in futures.items():
            try:
                result = future.result()
                results[tool] = result if isinstance(result, dict) else unified_result(
                    tool=tool,
                    status="error",
                    tokens=[],
                    lemmas=[],
                    pos=[],
                    reason="Analyzer returned invalid result.",
                )
            except Exception as exc:
                results[tool] = unified_result(
                    tool=tool,
                    status="error",
                    tokens=[],
                    lemmas=[],
                    pos=[],
                    reason=str(exc),
                )

    return results


def run_all_registered_tools(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Stable combined runner.

    - Core tools run in parallel.
    - MADAMIRA is excluded immediately.
    - SinaTools is lazy/non-blocking.
    - AraBERT is allowed to return fallback if model is not loaded.
    - No fake timeout results.
    """
    statuses = get_lightweight_tool_statuses()

    results: Dict[str, Dict[str, Any]] = {}

    # 1. Run core tools in parallel.
    core_results = run_core_tools(text)
    results.update(core_results)

    # 2. Run AraBERT separately.
    try:
        results["arabert"] = analyze_tool("arabert", text, statuses=statuses, use_cache=True)
    except Exception as exc:
        results["arabert"] = unified_result(
            tool="arabert",
            status="error",
            tokens=[],
            lemmas=[],
            pos=[],
            reason=str(exc),
        )

    # 3. Run SinaTools lazily.
    # Its adapter must return lazy_not_loaded/loading immediately if model is not ready.
    try:
        results["sinatools"] = analyze_tool("sinatools", text, statuses=statuses, use_cache=False)
    except Exception as exc:
        results["sinatools"] = unified_result(
            tool="sinatools",
            status="error",
            tokens=[],
            lemmas=[],
            pos=[],
            reason=str(exc),
        )

    # 4. MADAMIRA is excluded.
    results["madamira"] = _excluded_result(
        "madamira",
        text,
        "Missing licensed resources",
    )

    return results

def _tool_timeout_seconds(tool: str | None = None) -> float:
    """Tiered per-tool timeouts (cold vs warm).

    This controls each individual analyzer timeout; combined timeout should
    never prematurely cancel core tools.
    """

    tool_name = str(tool or "").lower()

    cold_timeouts = {
        "camel": 30.0,
        "farasa": 45.0,
        "stanza": 60.0,
        "qalsadi": 20.0,
        "alkhalil": 45.0,
        "udpipe": 20.0,
        "arabert": 60.0,
        "sinatools": 60.0,
        "madamira": 0.0,  # should never be scheduled in combined
    }
    warm_timeouts = {
        "camel": 10.0,
        "farasa": 15.0,
        "stanza": 20.0,
        "qalsadi": 10.0,
        "alkhalil": 15.0,
        "udpipe": 10.0,
        "arabert": 20.0,
        "sinatools": 20.0,
        "madamira": 0.0,
    }

    # Allow explicit override via env for fine-tuning.
    env_key = f"{tool_name.upper()}_TIMEOUT_SECONDS" if tool_name else ""
    if env_key and os.environ.get(env_key):
        try:
            return max(3.0, min(float(os.environ[env_key]), 180.0))
        except (TypeError, ValueError):
            pass

    cold_default = cold_timeouts.get(tool_name, 25.0)
    warm_default = warm_timeouts.get(tool_name, 15.0)

    # Detect warm/cold by checking module singletons/pipelines.
    # Keep this lightweight; if detection fails, fall back to cold.
    is_warm = False
    try:
        if tool_name == "camel":
            from app.tools import camel_tool as _t

            is_warm = getattr(_t, "camel_disambiguator", None) is not None and getattr(_t, "camel_db", None) is not None
        elif tool_name == "farasa":
            from app.tools import farasa_tool as _t

            is_warm = getattr(_t, "farasa_segmenter", None) is not None
        elif tool_name == "stanza":
            from app.tools import stanza_tool as _t

            is_warm = getattr(_t, "stanza_pipeline", None) is not None
        elif tool_name == "qalsadi":
            # qalsadi wrapper is typically lightweight; treat as warm.
            is_warm = True
        elif tool_name == "alkhalil":
            from app.tools import alkhalil_tool as _t

            jar_path = getattr(_t, "alkhalil_jar_path", None)
            is_warm = bool(jar_path)
        elif tool_name == "udpipe":
            from app.tools import udpipe_tool as _t

            is_warm = getattr(_t, "udpipe_pipeline_obj", None) is not None
        elif tool_name == "arabert":
            from app.tools import arabert_tool as _t

            is_warm = bool(getattr(_t, "arabert_model", None)) and bool(getattr(_t, "tokenizer", None))
        elif tool_name == "sinatools":
            from app.tools import sinatools_tool as _t

            is_warm = getattr(_t, "sinatools_loaded", False) is True
    except Exception:
        is_warm = False

    return (warm_default if is_warm else cold_default)



def _combined_timeout_seconds() -> float:
    """Response-safety budget for /analyze-combined.

    Must be high enough that core tools can complete and return real outputs.
    Per-tool timeouts still apply inside each analyzer.

    Cold default: 60s, Warm default: 15s.
    """

    # Prefer warm/cold split via env.
    # If explicit ARABIC_NLP_COMBINED_TIMEOUT_SECONDS is set, honor it.
    explicit = os.environ.get("ARABIC_NLP_COMBINED_TIMEOUT_SECONDS")
    if explicit:
        try:
            return max(5.0, min(float(explicit), 180.0))
        except (TypeError, ValueError):
            pass

    # Heuristic: if core tools are already loaded, consider this warm.
    try:
        from app.tools import farasa_tool as _farasa
        from app.tools import stanza_tool as _stanza

        warm = (
            getattr(_farasa, "farasa_segmenter", None) is not None
            and getattr(_stanza, "stanza_pipeline", None) is not None
        )
    except Exception:
        warm = False

    return 15.0 if warm else 60.0



def _timed_out_result(tool: str, timeout_s: float, text: str) -> Dict[str, Any]:
    payload = unified_result(
        tool=tool,
        status="timeout",
        tokens=[],
        lemmas=[],
        pos=[],
        reason=f"{tool} exceeded the {timeout_s:g}s demo safety timeout.",
    )
    payload["input"] = text
    payload["word_count"] = 0
    payload["runtime_ms"] = int(timeout_s * 1000)
    return payload


def _run_tool_batch(tools: tuple[str, ...], text: str, *, timeout_s: float) -> Dict[str, Dict[str, Any]]:
    started_at = time.perf_counter()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(len(tools), 8)))
    futures = {tool: executor.submit(analyze_tool, tool, text) for tool in tools}
    done, pending = concurrent.futures.wait(futures.values(), timeout=timeout_s)

    results: Dict[str, Dict[str, Any]] = {}
    for tool, future in futures.items():
        if future in done:
            try:
                result = future.result()
                if isinstance(result, dict):
                    result.setdefault("runtime_ms", int((time.perf_counter() - started_at) * 1000))
                results[tool] = result
            except Exception as exc:
                results[tool] = unified_result(tool=tool, status="error", tokens=[], lemmas=[], pos=[], reason=str(exc))
        else:
            future.cancel()
            results[tool] = _timed_out_result(tool, timeout_s, text)

    # Do not wait for non-cooperative third-party analyzers. They are isolated
    # behind timeout results so API responses remain demo-safe.
    executor.shutdown(wait=False, cancel_futures=True)
    return results


def _run_with_timeout(tool: str, analyzer: Callable[[str], Dict[str, Any]], text: str, *, timeout_s: float) -> Dict[str, Any]:
    result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put(("ok", analyzer(text)))
        except Exception as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=target, name=f"{tool}-timeout", daemon=True)
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        logger.warning("[%s] timed out after %.1fs", tool, timeout_s)
        return _timed_out_result(tool, timeout_s, text)

    try:
        kind, payload = result_queue.get_nowait()
    except queue.Empty:
        return unified_result(tool=tool, status="error", tokens=[], lemmas=[], pos=[], reason="Analyzer finished without returning a result.")

    if kind == "ok":
        if isinstance(payload, dict):
            payload.setdefault("runtime_ms", None)
        return payload

    exc = payload
    if isinstance(exc, Exception):
        logger.exception("[%s] timeout wrapper failure", tool)
        return unified_result(tool=tool, status="error", tokens=[], lemmas=[], pos=[], reason=str(exc))
    return unified_result(tool=tool, status="error", tokens=[], lemmas=[], pos=[], reason=str(exc))


# Singleflight implementation for run_all_tools(text)
# Ensures:
# - Exactly one leader executes run_core_tools() per unique text
# - Followers wait without busy-waiting
# - Followers get the exact same result object
# - Exceptions propagate to all waiters

_inflight_run_lock = threading.Lock()
_inflight_run_state: dict[str, "_SingleflightState"] = {}


class _SingleflightState:
    __slots__ = ("event", "result", "exc")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result = None
        self.exc = None


def run_all_tools(text: str):
    """Run core tools (camel, farasa, stanza, qalsadi) with singleflight dedupe.

    Thread-safe across concurrent FastAPI requests.
    """
    key = f"run_all_tools::{text}"

    with _inflight_run_lock:
        state = _inflight_run_state.get(key)
        if state is None:
            state = _SingleflightState()
            _inflight_run_state[key] = state
            leader = True
        else:
            leader = False

    if not leader:
        state.event.wait()
        if state.exc is not None:
            raise state.exc
        packed = state.result
        # packed is a tuple(camel, farasa, stanza, qalsadi)
        return packed[0], packed[1], packed[2], packed[3]

    try:
        results = run_core_tools(text)
        state.result = (
            results["camel"],
            results["farasa"],
            results["stanza"],
            results["qalsadi"],
        )
        return state.result[0], state.result[1], state.result[2], state.result[3]
    except BaseException as e:
        state.exc = e
        raise
    finally:
        # Set event only after result/exception is stored.
        state.event.set()
        # Cleanup AFTER all followers can safely read state.result/state.exc.
        # The state remains accessible to followers even after set(), because we wait on event.
        # Remove from map after releasing leader lock section to minimize contention.
        with _inflight_run_lock:
            # Another follower may already have captured `state` reference; it still points to the state object.
            _inflight_run_state.pop(key, None)

def _available_physical_memory_mb() -> float | None:
    try:
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return float(status.ullAvailPhys) / (1024.0 * 1024.0)
    except Exception:
        return None


def _skipped_low_memory_result(tool: str, text: str, *, available_mb: float, threshold_mb: float) -> Dict[str, Any]:
    payload = unified_result(
        tool=tool,
        status="skipped_low_memory",
        tokens=[],
        lemmas=[],
        pos=[],
        reason=f"{tool} skipped because available RAM was {available_mb:.0f} MB below the {threshold_mb:.0f} MB safety threshold.",
    )
    payload["input"] = text
    payload["word_count"] = 0
    payload["available_memory_mb"] = round(available_mb, 1)
    payload["memory_threshold_mb"] = round(threshold_mb, 1)
    return payload


def _execution_mode() -> str:
    return (os.environ.get("ARABIC_NLP_MODE") or os.environ.get("ARABIC_NLP_RUN_MODE") or "demo").strip().lower()


def _low_memory_guards_enabled() -> bool:
    return _execution_mode() not in {"demo", "full"}


def _low_memory_threshold_mb(tool: str) -> float | None:
    if tool in {"arabert", "madamira"}:
        return float(os.environ.get("ARABIC_NLP_LOW_MEMORY_THRESHOLD_MB", "1536"))
    if tool == "stanza":
        return float(os.environ.get("STANZA_LOW_MEMORY_THRESHOLD_MB", "4096"))
    if tool == "sinatools":
        return float(os.environ.get("SINATOOLS_LOW_MEMORY_THRESHOLD_MB", "6144"))
    return None


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def warm_up_heavy_tools(sample_text: str = DEMO_SAMPLE_TEXT) -> None:
    tools = ["stanza", "arabert"]
    if _truthy_env("SINATOOLS_PRELOAD", "false"):
        tools.append("sinatools")

    for tool in tools:
        if _low_memory_guards_enabled():
            available_mb = _available_physical_memory_mb()
            threshold_mb = _low_memory_threshold_mb(tool)
            if available_mb is not None and threshold_mb is not None and available_mb < threshold_mb:
                logger.warning(
                    "[warmup] %s skipped_low_memory available_mb=%.0f threshold_mb=%.0f",
                    tool.upper(),
                    available_mb,
                    threshold_mb,
                )
                continue

        started_at = time.perf_counter()
        logger.info("[warmup] %s starting", tool.upper())
        try:
            result = analyze_tool(tool, sample_text)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "[warmup] %s finished status=%s runtime_ms=%s reason=%s",
                tool.upper(),
                result.get("status", "unknown") if isinstance(result, dict) else "invalid",
                elapsed_ms,
                (result.get("reason", "") if isinstance(result, dict) else "invalid result") or "",
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.warning("[warmup] %s failed runtime_ms=%s reason=%s", tool.upper(), elapsed_ms, exc)


def start_heavy_tool_warmup(sample_text: str = DEMO_SAMPLE_TEXT) -> threading.Thread:
    thread = threading.Thread(target=warm_up_heavy_tools, args=(sample_text,), name="heavy-tool-warmup", daemon=True)
    thread.start()
    return thread


def get_tool_statuses() -> Dict[str, Dict[str, Any]]:
    return detect_tool_status()


def get_lightweight_tool_statuses() -> Dict[str, Dict[str, Any]]:
    java_present = shutil.which("java") is not None
    statuses: Dict[str, Dict[str, Any]] = {}

    statuses["camel"] = (
        {"status": "ok", "reason": "camel-tools package detected."}
        if has_module("camel_tools")
        else {"status": "missing_dependency", "reason": "Install camel-tools."}
    )

    try:
        from app.tools import farasa_tool as _farasa_tool

        farasa_loaded = _farasa_tool.farasa_segmenter is not None
        farasa_path = _farasa_tool.farasa_model_path
    except Exception:
        farasa_loaded = False
        farasa_path = None
    statuses["farasa"] = {
        "status": "ok" if java_present and (farasa_loaded or farasa_bins_present()) else ("missing_java" if not java_present else "missing_model"),
        "reason": "Farasa JAR is registered for on-demand analysis.",
        "loaded": farasa_loaded,
        "jar_path": str(farasa_path) if farasa_path else None,
    }

    try:
        from app.tools import stanza_tool as _stanza_tool

        stanza_loaded = _stanza_tool.stanza_pipeline is not None
    except Exception:
        stanza_loaded = False
    statuses["stanza"] = (
        {"status": "ok", "reason": "stanza package and Arabic model detected.", "loaded": stanza_loaded}
        if has_module("stanza") and stanza_model_present()
        else {"status": "missing_model" if has_module("stanza") else "missing_dependency", "reason": "Stanza package/model is not ready.", "loaded": False}
    )

    statuses["qalsadi"] = (
        {"status": "ok", "reason": "qalsadi package detected."}
        if has_module("qalsadi") and has_module("camel_tools")
        else {"status": "missing_dependency", "reason": "Install qalsadi and camel-tools."}
    )

    arabert_cache = _arabert_local_cache_path()
    statuses["arabert"] = (
        {
            "status": "ok",
            "reason": "AraBERT dependencies/resources are registered for on-demand loading.",
            "loaded": False,
            "model_path": str(arabert_cache) if arabert_cache else None,
        }
        if has_module("transformers") and has_module("torch") and arabert_cache is not None
        else {
            "status": "missing_model",
            "reason": "AraBERT model is not available in the local cache.",
            "loaded": False,
            "model_path": str(arabert_cache) if arabert_cache else None,
        }
    )

    alkhalil_jar = Path(os.environ.get("ALKHALIL_JAR", "")) if os.environ.get("ALKHALIL_JAR") else PROJECT_ROOT / "app" / "tools" / "alkhalil" / "AlKhalil1.1" / "AlKhalil.jar"
    statuses["alkhalil"] = {
        "status": "ok" if java_present and alkhalil_jar.exists() else ("missing_java" if not java_present else "missing_model"),
        "reason": "AlKhalil JAR is registered for on-demand analysis.",
        "resolved_jar": str(alkhalil_jar),
        "jar_exists": alkhalil_jar.exists(),
    }

    try:
        from app.tools import udpipe_tool as _udpipe_tool

        udpipe_loaded = _udpipe_tool.udpipe_pipeline_obj is not None
        udpipe_path = _udpipe_tool.udpipe_model_path or PROJECT_ROOT / "app" / "tools" / "udpipe" / "arabic.udpipe"
    except Exception:
        udpipe_loaded = False
        udpipe_path = PROJECT_ROOT / "app" / "tools" / "udpipe" / "arabic.udpipe"
    statuses["udpipe"] = {
        "status": "ok" if Path(udpipe_path).exists() else "missing_model",
        "reason": "UDPipe model is registered for on-demand analysis.",
        "loaded": udpipe_loaded,
        "path": str(udpipe_path),
    }

    statuses["madamira"] = {
        "status": "excluded",
        "reason": "MADAMIRA is intentionally excluded until licensed resources are configured.",
        "loaded": False,
        "resources_found": False,
    }

    sinatools_path = _sinatools_lemma_path()
    statuses["sinatools"] = {
        "status": "ok" if sinatools_path is not None else "missing_resources",
        "reason": (
            f"SinaTools lemma dictionary is present at {sinatools_path}. It will lazy-load once on first request."
            if sinatools_path
            else "Missing SinaTools lemma pickle."
        ),
        "loaded": False,
        "resource_path": str(sinatools_path) if sinatools_path else None,
        "resources_found": sinatools_path is not None,
        "required_resources": {"lemma_pickle": sinatools_path is not None},
    }

    return statuses


def _arabert_local_cache_path() -> Path | None:
    configured = os.environ.get("ARABERT_MODEL_PATH")
    if configured and (Path(configured) / "config.json").exists():
        return Path(configured)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--aubmindlab--bert-base-arabertv2"
    ref_main = cache_root / "refs" / "main"
    if not ref_main.exists():
        return None
    try:
        snapshot = ref_main.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    candidate = cache_root / "snapshots" / snapshot
    return candidate if (candidate / "config.json").exists() else None


def _sinatools_lemma_path() -> Path | None:
    configured = os.environ.get("SINATOOLS_LEMMA_PICKLE") or os.environ.get("SINATOOLS_LEMMAS_PICKLE")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / "app" / "tools" / "sinatools" / "lemma.pickle",
            PROJECT_ROOT / "app" / "tools" / "sinatools" / "lemmas_dic.pickle",
            PROJECT_ROOT / "app" / "tools" / "sinatools" / "resources" / "lemma.pickle",
            PROJECT_ROOT / "app" / "tools" / "sinatools" / "resources" / "lemmas_dic.pickle",
            Path.home() / "AppData" / "Roaming" / "sinatools" / "lemmas_dic.pickle",
            Path.home() / "AppData" / "Roaming" / "sinatools" / "lemma.pickle",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resources_found(status: Dict[str, Any]) -> bool:
    if status.get("status") in {"missing_resources", "missing_model", "missing_dependency", "missing_java"}:
        return False
    missing = status.get("missing")
    if isinstance(missing, list) and missing:
        return False
    required = status.get("required_resources")
    if isinstance(required, dict):
        return all(bool(value) for value in required.values())
    return status.get("status") == "ok"


def _record_tool_health(tool: str, result: Dict[str, Any], status: Dict[str, Any], started_at: float) -> None:
    runtime_ms = int((time.perf_counter() - started_at) * 1000)
    if isinstance(result, dict) and result.get("runtime_ms") is None:
        result["runtime_ms"] = runtime_ms

    result_status = result.get("status", "unknown") if isinstance(result, dict) else "invalid"
    last_error = None
    if result_status != "ok":
        last_error = result.get("reason") if isinstance(result, dict) else "Invalid analyzer response."
    elif status.get("status") not in {None, "ok"}:
        last_error = status.get("reason")

    with _LAST_TOOL_HEALTH_LOCK:
        _LAST_TOOL_HEALTH[tool] = {
            "tool": tool,
            "status": result_status,
            "loaded": bool(status.get("loaded")) or result_status == "ok",
            "resources_found": _resources_found(status) or result_status == "ok",
            "last_error": last_error,
            "runtime_ms": runtime_ms,
        }


def _dashboard_status(tool: str, status: Dict[str, Any], prior: Dict[str, Any] | None = None) -> str:
    prior = prior or {}
    raw_status = str(status.get("status") or prior.get("status") or "unknown").lower()
    loaded = bool(status.get("loaded")) or bool(prior.get("loaded"))

    if tool in EXCLUDED_TOOLS:
        return "excluded"
    if raw_status == "loading":
        return "loading"
    if tool in HEAVY_LAZY_TOOLS and raw_status == "ok" and not loaded:
        return "lazy"
    return raw_status


def _health_payload_from_status(tool: str, status: Dict[str, Any], prior: Dict[str, Any] | None = None) -> Dict[str, Any]:
    prior = prior or {}
    health_status = _dashboard_status(tool, status, prior)
    loaded = bool(status.get("loaded")) or bool(prior.get("loaded"))
    resources_found = _resources_found(status)
    reason = prior.get("last_error") or status.get("reason") or ""
    registered = tool in ALL_TOOLS
    excluded = health_status == "excluded"

    return {
        "tool": tool,
        "status": health_status,
        "registered": registered,
        "loaded": loaded,
        "lazy": health_status == "lazy",
        "loading": health_status == "loading",
        "available": health_status in {"ok", "partial", "lazy", "loading"},
        "unavailable": health_status not in {"ok", "partial", "lazy", "loading"} and not excluded,
        "excluded": excluded,
        "resources_found": resources_found,
        "category": "excluded" if excluded else ("heavy_lazy" if tool in HEAVY_LAZY_TOOLS else "core"),
        "reason": reason,
        "last_error": prior.get("last_error") or (status.get("reason") if status.get("status") != "ok" else None),
        "runtime_ms": prior.get("runtime_ms"),
    }


def get_lightweight_health() -> Dict[str, Any]:
    statuses = get_lightweight_tool_statuses()
    with _LAST_TOOL_HEALTH_LOCK:
        last_health = {tool: dict(payload) for tool, payload in _LAST_TOOL_HEALTH.items()}

    tools = {
        tool: _health_payload_from_status(tool, statuses.get(tool, {}), last_health.get(tool, {}))
        for tool in ALL_TOOLS
    }
    counted_tools = [tool for tool, payload in tools.items() if payload["registered"] and not payload["excluded"]]
    active_tools = [
        tool
        for tool, payload in tools.items()
        if tool in counted_tools and payload["status"] in {"ok", "partial", "lazy", "loading"}
    ]
    degraded_tools = [
        tool
        for tool, payload in tools.items()
        if payload["excluded"] or payload["status"] not in {"ok", "partial", "lazy", "loading"}
    ]

    return {
        "backend": {"status": "online", "ready": True},
        "tools": tools,
        "tool_status": {tool: payload["status"] for tool, payload in tools.items()},
        "registered_tools": counted_tools,
        "active_tools": active_tools,
        "lazy_tools": [tool for tool, payload in tools.items() if payload["lazy"]],
        "loading_tools": [tool for tool, payload in tools.items() if payload["loading"]],
        "excluded_tools": [tool for tool, payload in tools.items() if payload["excluded"]],
        "degraded_tools": degraded_tools,
        "counts": {
            "registered": len(counted_tools),
            "active": len(active_tools),
            "lazy": sum(1 for payload in tools.values() if payload["lazy"]),
            "loading": sum(1 for payload in tools.values() if payload["loading"]),
            "excluded": sum(1 for payload in tools.values() if payload["excluded"]),
            "degraded": len(degraded_tools),
        },
        "memory": get_memory_report(),
    }


def get_demo_tool_health(*, run_sample: bool = True, sample_text: str = DEMO_SAMPLE_TEXT) -> Dict[str, Dict[str, Any]]:
    statuses = detect_tool_status() if run_sample else get_lightweight_tool_statuses()
    if run_sample:
        run_all_registered_tools(sample_text)
        statuses = detect_tool_status()

    with _LAST_TOOL_HEALTH_LOCK:
        last_health = {tool: dict(payload) for tool, payload in _LAST_TOOL_HEALTH.items()}

    health: Dict[str, Dict[str, Any]] = {}
    for tool in ALL_TOOLS:
        status = statuses.get(tool, {})
        prior = last_health.get(tool, {})
        health[tool] = _health_payload_from_status(tool, status, prior)
    return health


def get_memory_report() -> Dict[str, Any]:
    available_mb = _available_physical_memory_mb()
    return {
        "available_mb": None if available_mb is None else round(available_mb, 1),
        "mode": _execution_mode(),
        "low_memory_guards_enabled": _low_memory_guards_enabled(),
    }


def warm_up_all_tools(sample_text: str = DEMO_SAMPLE_TEXT) -> Dict[str, Dict[str, Any]]:
    return run_all_registered_tools(sample_text)


__all__ = [
    "ANALYZERS",
    "analyze_tool",
    "camel_analyze",
    "farasa_analyze",
    "stanza_analyze",
    "qalsadi_analyze",
    "cached_analyze",
    "clear_cache",
    "run_all_tools",
    "run_core_tools",
    "run_all_registered_tools",
    "start_heavy_tool_warmup",
    "warm_up_all_tools",
    "get_tool_statuses",
    "get_lightweight_tool_statuses",
    "get_lightweight_health",
    "get_demo_tool_health",
    "get_memory_report",
    "DEMO_SAMPLE_TEXT",
    "fusion_system",
    "evaluate_tools",
]
