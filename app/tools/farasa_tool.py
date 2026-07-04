from __future__ import annotations

import os
import site
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.tool_registry import unavailable_result
from app.tools.base_tool import BaseTool
from app.utils.logger import logger, log_time


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FarasaSegmenter = None
simple_word_tokenize = None
farasa_segmenter = None
farasa_import_error = None
farasa_model_path: Optional[Path] = None
farasa_status: Dict[str, Any] = {
    "status": "unavailable",
    "reason": "Farasa has not been initialized yet.",
}
_farasa_load_attempted = False
_FARASA_TIMEOUT_S = float(os.environ.get("FARASA_TIMEOUT_SECONDS", "30"))


def _fallback_tokenize(text: str) -> List[str]:
    return [part for part in str(text or "").split() if part]


def _ensure_imports() -> bool:
    global FarasaSegmenter, simple_word_tokenize, farasa_import_error
    if FarasaSegmenter and simple_word_tokenize:
        return True
    try:
        # Farasapy may transitively import camel-tools, which expects
        # emoji.EMOJI_DATA at import-time. Patch it before imports.
        from backend.utils.emoji_compat import ensure_emoji_emoji_data

        ensure_emoji_emoji_data()

        from camel_tools.tokenizers.word import simple_word_tokenize as _simple_word_tokenize
        from farasa.segmenter import FarasaSegmenter as _FarasaSegmenter

        FarasaSegmenter = _FarasaSegmenter
        simple_word_tokenize = _simple_word_tokenize
        farasa_import_error = None
        return True
    except Exception as exc:
        farasa_import_error = str(exc)
        return False


def _candidate_farasa_jars() -> List[Path]:
    candidates: List[Path] = []

    configured = os.environ.get("FARASA_JAR")
    if configured:
        candidates.append(Path(configured).expanduser())

    local_roots = [
        PROJECT_ROOT / "app" / "tools" / "farasa",
        PROJECT_ROOT / "Farasa_bin",
    ]

    for root in local_roots:
        if root.exists():
            candidates.extend(sorted(p for p in root.glob("**/FarasaSegmenterJar.jar") if p.is_file()))

    for package_root in site.getsitepackages() + [site.getusersitepackages()]:
        if not package_root:
            continue
        root_path = Path(package_root)
        if root_path.exists():
            candidates.extend(sorted(p for p in root_path.glob("**/FarasaSegmenterJar.jar") if p.is_file()))

    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def resolve_farasa_jar() -> Optional[Path]:
    for candidate in _candidate_farasa_jars():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _set_farasa_status(status: str, reason: str, jar_path: Optional[Path] = None) -> None:
    global farasa_status
    farasa_status = {"status": status, "reason": reason}
    if jar_path is not None:
        farasa_status["jar_path"] = str(jar_path)


def load_farasa() -> None:
    global farasa_segmenter, farasa_model_path, _farasa_load_attempted
    if farasa_segmenter is not None:
        _set_farasa_status("ok", "Farasa is loaded.", farasa_model_path)
        return
    if _farasa_load_attempted:
        return

    _farasa_load_attempted = True

    if not _ensure_imports():
        message = farasa_import_error or "Farasa package or camel-tools tokenizer is unavailable."
        logger.warning("Farasa unavailable: %s", message)
        _set_farasa_status("unavailable", message)
        return

    jar_path = resolve_farasa_jar()
    if jar_path is None:
        message = "Farasa local JAR not found. Set FARASA_JAR or place FarasaSegmenterJar.jar under app/tools/farasa, Farasa_bin, or site-packages."
        logger.warning("Farasa unavailable: %s", message)
        farasa_segmenter = None
        farasa_model_path = None
        _set_farasa_status("unavailable", message)
        return

    try:
        farasa_model_path = jar_path
        logger.info("Farasa local JAR resolved: %s", jar_path)
        farasa_segmenter = FarasaSegmenter(binary_path=str(jar_path), interactive=False)
        logger.info("Farasa loaded from %s", jar_path)
        _set_farasa_status("ok", f"Farasa loaded from {jar_path}", jar_path)
    except Exception as exc:
        logger.warning("Farasa failed to load from %s: %s", jar_path, exc)
        farasa_segmenter = None
        _set_farasa_status("error", f"Farasa internal error while loading local JAR at {jar_path}: {exc}", jar_path)


def get_farasa_status() -> Dict[str, Any]:
    if farasa_segmenter is not None:
        return {
            "status": "ok",
            "reason": f"Farasa loaded from {farasa_model_path}" if farasa_model_path else "Farasa loaded.",
            "jar_path": str(farasa_model_path) if farasa_model_path else None,
        }

    if not _farasa_load_attempted:
        load_farasa()

    payload = dict(farasa_status)
    if farasa_model_path is not None and "jar_path" not in payload:
        payload["jar_path"] = str(farasa_model_path)
    return payload


def _segment_with_local_jar(text: str) -> str:
    if farasa_model_path is None:
        raise RuntimeError("Farasa JAR path is not resolved.")

    tmp_dir = PROJECT_ROOT / ".tmp" / "farasa"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix=".in", mode="w", encoding="utf-8") as input_file:
        input_file.write(text or "")
        input_path = Path(input_file.name)

    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix=".out", mode="w", encoding="utf-8") as output_file:
        output_path = Path(output_file.name)

    try:
        proc = subprocess.run(
            [
                "java",
                "-Dfile.encoding=UTF-8",
                "-jar",
                str(farasa_model_path),
                "-i",
                str(input_path),
                "-o",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_FARASA_TIMEOUT_S,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise RuntimeError(stderr or stdout or f"Farasa exited with code {proc.returncode}.")
        return output_path.read_text(encoding="utf-8", errors="replace").strip()
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Farasa exceeded {_FARASA_TIMEOUT_S:g}s") from exc
    finally:
        for path in (input_path, output_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def farasa_analyze(text: str) -> Dict[str, Any]:
    global farasa_segmenter

    status = get_farasa_status()
    if status.get("status") != "ok":
        return {
            "tool": "farasa",
            "status": status.get("status", "error"),
            "reason": status.get("reason", "Farasa is unavailable."),
            "input": text,
            "word_count": 0,
            "tokens": [],
        }

    t0 = time.time()
    try:
        segmented = _segment_with_local_jar(text or "")
        raw_tokens = simple_word_tokenize(text) if simple_word_tokenize else _fallback_tokenize(text)
        raw_segs = segmented.split()

        token_outputs: List[Dict[str, Any]] = []
        for i, token in enumerate(raw_tokens):
            seg = raw_segs[i] if i < len(raw_segs) else token
            parts = [p for p in seg.split("+") if p]
            token_outputs.append({"surface": token, "segmentation": parts, "analyses": []})

        log_time("farasa", text, time.time() - t0)
        return {
            "tool": "farasa",
            "status": "ok",
            "input": text,
            "word_count": len(token_outputs),
            "segmented_text": segmented,
            "tokens": token_outputs,
        }
    except TimeoutError as exc:
        logger.warning("[FARASA] timeout: %s", exc)
        return {
            "tool": "farasa",
            "status": "error",
            "reason": f"Farasa timeout: {exc}",
            "input": text,
            "word_count": 0,
            "tokens": [],
        }
    except Exception as exc:
        logger.warning("[FARASA] error: %s", exc)
        return {
            "tool": "farasa",
            "status": "error",
            "reason": str(exc),
            "input": text,
            "word_count": 0,
            "tokens": [],
        }


class FarasaTool(BaseTool):
    tool_name = "farasa"

    def is_loaded(self) -> bool:
        return farasa_segmenter is not None

    def analyze(self, text: str) -> Dict[str, Any]:
        return farasa_analyze(text)
