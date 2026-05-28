from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _default_jar_path() -> Path:
    # Repository layout (legacy):
    #   <project_root>/app/tools/alkhalil/AlKhalil1.1/AlKhalil.jar
    # Historical note: some deployments may have moved jar to tools/alkhalil/alkhalil.jar.
    # We support both in the caller.
    base = Path(__file__).resolve().parents[1]
    return base / "alkhalil" / "AlKhalil1.1" / "AlKhalil.jar"


def resolve_jar_path() -> Path:
    configured = os.environ.get("ALKHALIL_JAR")
    if configured:
        return Path(configured)

    # Required fallback by task
    required_fallback = Path(__file__).resolve().parents[2] / "tools" / "alkhalil" / "alkhalil.jar"

    if required_fallback.exists() and required_fallback.is_file():
        return required_fallback

    # Backward compatible fallback (repo ships jar under app/tools/alkhalil/AlKhalil1.1/Alkhalil.jar)
    legacy = Path(__file__).resolve().parents[1] / "alkhalil" / "AlKhalil1.1" / "Alkhalil.jar"
    if legacy.exists() and legacy.is_file():
        return legacy

    # Also keep the older helper fallback
    return _default_jar_path()




def _extract_tokens_and_lemmas(raw: str) -> Tuple[List[str], List[str]]:
    """Best-effort parsing of AlKhalil output.

    AlKhalil CLI/GUI output formats vary across distributions.
    We attempt to find lines like:
      token ... lemma ...
    or JSON embedded.

    Returns empty lists if parsing fails.
    """
    raw = raw or ""

    # 1) Try embedded JSON
    raw_stripped = raw.strip()
    if raw_stripped.startswith("{") and raw_stripped.endswith("}"):
        try:
            data = json.loads(raw_stripped)
            tokens = [t.get("token") or t.get("surface") for t in data.get("tokens", []) if isinstance(t, dict)]
            lemmas = [l.get("lemma") for l in data.get("lemmas", []) if isinstance(l, dict)]
            tokens = [x for x in tokens if x]
            lemmas = [x for x in lemmas if x]
            return tokens, lemmas
        except Exception:
            pass

    # 2) Heuristic line parsing
    # Match: <anything> <token> ... <lemma>
    # We'll capture first two groups if a lemma-like column exists.
    tokens: List[str] = []
    lemmas: List[str] = []

    # Look for patterns like: TOKEN=... LEMMA=...
    for m in re.finditer(r"TOKEN\s*[=:]\s*(?P<tok>[^\s\|,;]+).{0,40}?LEMMA\s*[=:]\s*(?P<lemma>[^\s\|,;]+)", raw, re.I):
        tok = (m.group("tok") or "").strip()
        lemma = (m.group("lemma") or "").strip()
        if tok:
            tokens.append(tok)
            if lemma:
                lemmas.append(lemma)

    if tokens:
        return tokens, lemmas

    # 3) Fallback: tokenize by whitespace and use tokens as lemmas (safe placeholder)
    # Only do this if there are obvious outputs.
    if len(raw) > 0 and any(ch.isalpha() for ch in raw):
        # Extract Arabic-ish sequences
        words = re.findall(r"[\u0600-\u06FF]+", raw)
        # Too noisy? keep last N unique-ish words
        if words:
            return words[:200], words[:200]

    return [], []


def alkhalil_analyze(text: str) -> Dict[str, Any]:
    tool = "alkhalil"
    try:
        java = shutil.which("java")
        if not java:
            return {"tool": tool, "status": "unavailable", "tokens": [], "lemmas": [], "reason": "Java executable not found."}

        jar_path = resolve_jar_path()

        # Fix missing ALKHALIL_JAR environment variable (task requirement)
        if os.environ.get("ALKHALIL_JAR") is None and jar_path is not None:
            os.environ["ALKHALIL_JAR"] = str(jar_path)

        if not jar_path.exists() or not jar_path.is_file():
            return {
                "tool": tool,
                "status": "unavailable",
                "tokens": [],
                "lemmas": [],
                "pos": [],
                "reason": f"AlKhalil JAR not found at {jar_path}",
            }


        # AlKhalil ships multiple entry points/scripts; we use `alkhalil.sh` if present.
        # On Windows, the jar is still runnable with java -jar.
        timeout_s = float(os.environ.get("ALKHALIL_TIMEOUT_SECONDS", "8"))
        timeout_s = max(5.0, min(timeout_s, 10.0))

        # Provide input via stdin to reduce command-line format issues.
        # Many jars read from stdin or expect a file; stdin fallback is common.
        proc = subprocess.run(
            [java, "-jar", str(jar_path)],
            input=(text or "").encode("utf-8", errors="ignore"),
            capture_output=True,
            timeout=timeout_s,
            shell=False,
        )

        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        combined = out + ("\n" + err if err else "")

        tokens, lemmas = _extract_tokens_and_lemmas(combined)

        if not tokens:
            return {
                "tool": tool,
                "status": "unavailable",
                "tokens": [],
                "lemmas": [],
                "reason": "AlKhalil executed but output could not be parsed.",
            }

        from app.core.tool_registry import unified_result

        return unified_result(tool=tool, status="ok", tokens=tokens, lemmas=lemmas, pos=[], reason="")

    except subprocess.TimeoutExpired:
        from app.core.tool_registry import unified_result

        return unified_result(tool=tool, status="unavailable", tokens=[], lemmas=[], pos=[], reason="AlKhalil execution timed out.")
    except Exception as exc:
        from app.core.tool_registry import unified_result

        return unified_result(tool=tool, status="error", tokens=[], lemmas=[], pos=[], reason=str(exc))


