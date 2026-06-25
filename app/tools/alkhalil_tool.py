from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logger import logger
from app.utils.helpers import strip_diacritics
from backend.config.tool_paths import AlKhalilPaths


alkhalil_jar_path: Optional[str] = None
_alkhalil_paths = AlKhalilPaths()



def _build_command(jar_path: str, input_path: Optional[Path] = None) -> List[str]:
    cmd = ["java", "-Dfile.encoding=UTF-8", "-jar", jar_path]
    if input_path is not None:
        cmd.append(str(input_path))
    return cmd


def _normalize_alkhalil_pos(raw_pos: Optional[str]) -> Optional[str]:
    if not raw_pos:
        return None
    pos = str(raw_pos).strip()
    if not pos:
        return None
    lowered = pos.lower()
    if "فعل" in pos or "verb" in lowered:
        return "VERB"
    if "اسم" in pos or "noun" in lowered:
        return "NOUN"
    if "حرف جر" in pos or "adposition" in lowered or lowered == "adp":
        return "ADP"
    if "ضمير" in pos or "pron" in lowered:
        return "PRON"
    if "صفة" in pos or "adj" in lowered:
        return "ADJ"
    if "ظرف" in pos or "adv" in lowered:
        return "ADV"
    if "part" in lowered:
        return "PART"
    return None


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _fallback_tokenize(text: str) -> List[str]:
    try:
        from pyarabic import araby

        return [part for part in araby.tokenize(text or "") if part.strip()]
    except Exception:
        return [part for part in str(text or "").split() if part]


def _normalize_fallback_surface(token: str) -> str:
    try:
        from pyarabic import araby

        return araby.strip_tatweel(token or "")
    except Exception:
        return str(token or "").replace("ـ", "")


def _pyarabic_fallback(text: str, reason: str) -> Dict[str, Any]:
    tokens: List[Dict[str, Any]] = []
    lemmas: List[str] = []

    for token in _fallback_tokenize(text):
        surface = token
        normalized_surface = _normalize_fallback_surface(token)
        try:
            from pyarabic import araby

            lemma = araby.strip_diacritics(normalized_surface)
        except Exception:
            lemma = strip_diacritics(normalized_surface)

        tokens.append(
            {
                "surface": surface,
                "lemma": lemma,
                "root": None,
                "pos": None,
                "upos": None,
                "normalized": True,
                "note": "pyarabic fallback - AlKhalil CLI unavailable",
                "analyses": [
                    {
                        "lemma": lemma,
                        "root": None,
                        "pos": None,
                        "gender": None,
                        "number": None,
                        "tense": None,
                        "gloss": None,
                    }
                ],
            }
        )
        lemmas.append(lemma)

    return {
        "tool": "alkhalil",
        "status": "partial",
        "reason": "pyarabic fallback",
        "input": text,
        "word_count": len(tokens),
        "tokens": tokens,
        "lemmas": lemmas,
    }


def _diagnose_gui_only_jar() -> Optional[str]:
    source_path = Path(__file__).resolve().parent / "alkhalil" / "AlKhalil1.1" / "src" / "AlKhalil" / "AlKhalil.java"
    if not source_path.exists():
        return None

    try:
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    if "new Gui(" in source_text or "Gui fen = new Gui()" in source_text:
        return "Bundled AlKhalil build is GUI-only (main() instantiates Gui) and does not expose a CLI analyzer."
    return None


def _parse_alkhalil_output(stdout: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "\t" in line:
            parts = [part.strip() for part in line.split("\t")]
        elif "|" in line:
            parts = [part.strip() for part in line.split("|")]
        else:
            parts = line.split()

        if len(parts) >= 5:
            surface, lemma, root, raw_pos, features = parts[:5]
            upos = _normalize_alkhalil_pos(raw_pos)
            tokens.append(
                {
                    "surface": surface or None,
                    "lemma": lemma or None,
                    "root": root or None,
                    "pos": raw_pos or None,
                    "upos": upos,
                    "raw_pos": raw_pos or None,
                    "features": features or None,
                    "analyses": [
                        {
                            "lemma": lemma or None,
                            "root": root or None,
                            "pos": upos,
                            "gender": None,
                            "number": None,
                            "tense": None,
                            "gloss": None,
                            "features": features or None,
                        }
                    ],
                }
            )
        elif len(parts) >= 4:
            surface, lemma, root, raw_pos = parts[:4]
            upos = _normalize_alkhalil_pos(raw_pos)
            tokens.append(
                {
                    "surface": surface or None,
                    "lemma": lemma or None,
                    "root": root or None,
                    "pos": raw_pos or None,
                    "upos": upos,
                    "raw_pos": raw_pos or None,
                    "features": None,
                    "analyses": [
                        {
                            "lemma": lemma or None,
                            "root": root or None,
                            "pos": upos,
                            "gender": None,
                            "number": None,
                            "tense": None,
                            "gloss": None,
                            "features": None,
                        }
                    ],
                }
            )
        elif len(parts) == 1:
            tokens.append(
                {
                    "surface": parts[0],
                    "lemma": None,
                    "root": None,
                    "pos": None,
                    "upos": None,
                    "raw_pos": None,
                    "features": None,
                    "analyses": [],
                }
            )

    return tokens


def _run_alkhalil_file_mode(jar_path: str, text: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="alkhalil_") as temp_dir:
        workdir = Path(temp_dir)
        input_path = workdir / "input.txt"
        output_path = workdir / "output.txt"
        input_path.write_text(text or "", encoding="utf-8")

        cmd = ["java", "-Dfile.encoding=UTF-8", "-jar", jar_path, "-i", str(input_path), "-o", str(output_path)]
        logger.info("[AlKhalil] command: %s", " ".join(cmd))
        logger.info("[AlKhalil] input bytes: %s", len((text or "").encode("utf-8")))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                cwd=str(workdir),
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            output_text = _read_text_file(output_path) if output_path.exists() else ""
            logger.info("[AlKhalil] return code: %s", proc.returncode)
            logger.info("[AlKhalil] stdout: %s", stdout)
            logger.info("[AlKhalil] stderr: %s", stderr)
            logger.info("[AlKhalil] output file exists: %s", output_path.exists())
            if output_text:
                logger.info("[AlKhalil] output file content: %s", output_text)
            return {
                "mode": "file",
                "stdout": stdout,
                "stderr": stderr,
                "output_text": output_text,
                "returncode": proc.returncode,
                "timeout": False,
            }
        except subprocess.TimeoutExpired as exc:
            logger.warning("[AlKhalil] file-mode timeout: %s", exc)
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            logger.info("[AlKhalil] stdout: %s", stdout)
            logger.info("[AlKhalil] stderr: %s", stderr)
            return {
                "mode": "file",
                "stdout": stdout,
                "stderr": stderr,
                "output_text": _read_text_file(output_path) if output_path.exists() else "",
                "returncode": None,
                "timeout": True,
            }


def _run_alkhalil_stdin_mode(jar_path: str, text: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="alkhalil_") as temp_dir:
        workdir = Path(temp_dir)
        cmd = ["java", "-Dfile.encoding=UTF-8", "-jar", jar_path]
        logger.info("[AlKhalil] fallback command: %s", " ".join(cmd))
        logger.info("[AlKhalil] input bytes: %s", len((text or "").encode("utf-8")))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workdir),
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(input=(text or "").encode("utf-8"), timeout=15)
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            logger.info("[AlKhalil] return code: %s", proc.returncode)
            logger.info("[AlKhalil] stdout: %s", stdout)
            logger.info("[AlKhalil] stderr: %s", stderr)
            return {
                "mode": "stdin",
                "stdout": stdout,
                "stderr": stderr,
                "output_text": "",
                "returncode": proc.returncode,
                "timeout": False,
            }
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            stdout_bytes, stderr_bytes = proc.communicate()
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            if exc.stdout:
                stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout)
            if exc.stderr:
                stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
            logger.warning("[AlKhalil] stdin-mode timeout")
            logger.info("[AlKhalil] stdout: %s", stdout)
            logger.info("[AlKhalil] stderr: %s", stderr)
            return {
                "mode": "stdin",
                "stdout": stdout,
                "stderr": stderr,
                "output_text": "",
                "returncode": None,
                "timeout": True,
            }


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



def _run_alkhalil(text: str) -> Dict[str, Any]:
    if not alkhalil_jar_path:
        return {"stdout": "", "stderr": "", "output_text": "", "timeout": False, "mode": "none"}

    gui_only_reason = _diagnose_gui_only_jar()
    if gui_only_reason:
        logger.warning("[AlKhalil] %s", gui_only_reason)
        return {
            "stdout": "",
            "stderr": gui_only_reason,
            "output_text": "",
            "timeout": False,
            "mode": "gui_only",
            "reason": gui_only_reason,
        }

    primary = _run_alkhalil_file_mode(alkhalil_jar_path, text)
    if primary.get("output_text") or (primary.get("stdout") and primary.get("stdout").strip()):
        return primary

    fallback = _run_alkhalil_stdin_mode(alkhalil_jar_path, text)
    if fallback.get("stdout") or fallback.get("stderr"):
        return fallback

    return fallback


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
                    "analyses": [
                        {
                            "lemma": lemma if lemma and lemma != "_" else None,
                            "root": root if root and root != "_" else None,
                            "pos": pos if pos and pos != "_" else None,
                            "gender": None,
                            "number": None,
                            "tense": None,
                            "gloss": None,
                        }
                    ],
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
                    "analyses": [],
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

        gui_only_reason = _diagnose_gui_only_jar()
        if gui_only_reason:
            return _pyarabic_fallback(text or "", f"AlKhalil GUI-only JAR. Using pyarabic fallback. {gui_only_reason}")


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

        run_result = _run_alkhalil(text or "")
        stdout = (run_result.get("output_text") or run_result.get("stdout") or "").strip()
        stderr = (run_result.get("stderr") or "").strip()

        if not stdout:
            reason = stderr or "AlKhalil produced no stdout/stderr output"
            if gui_only_reason:
                return _pyarabic_fallback(text or "", f"AlKhalil GUI-only JAR. Using pyarabic fallback. {reason}")
            return {
                "tool": tool,
                "status": "error",
                "reason": reason,
                "input": text,
                "word_count": 0,
                "tokens": [],
            }

        tokens = _parse_alkhalil_output(stdout)
        if not tokens:
            return {
                "tool": tool,
                "status": "error",
                "reason": stderr or stdout or "AlKhalil output unparseable",
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
            "lemmas": [tok.get("lemma") for tok in tokens if isinstance(tok, dict) and tok.get("lemma")],
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

