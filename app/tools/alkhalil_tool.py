from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.helpers import strip_diacritics
from app.utils.logger import logger
from backend.config.tool_paths import AlKhalilPaths


alkhalil_jar_path: Optional[str] = None
_alkhalil_paths = AlKhalilPaths()
_bridge_lock = threading.Lock()
_bridge_process: Optional[subprocess.Popen[str]] = None
_bridge_ready = False
_bridge_error: Optional[str] = None
_bridge_helper_dir: Optional[Path] = None
_bridge_source_path: Optional[Path] = None
_bridge_class_name = "AlKhalilBridge"


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
    if "verb" in lowered or "فعل" in pos:
        return "VERB"
    if "noun" in lowered or "اسم" in pos:
        return "NOUN"
    if "adposition" in lowered or lowered == "adp" or "حرف جر" in pos:
        return "ADP"
    if "pron" in lowered or "ضمير" in pos:
        return "PRON"
    if "adj" in lowered or "صفة" in pos:
        return "ADJ"
    if "adv" in lowered or "ظرف" in pos:
        return "ADV"
    if "part" in lowered or "حرف" in pos:
        return "PART"
    return None


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


def _confidence_from_rank(rank: int) -> Dict[str, Any]:
    score = max(0.15, round(1.0 - (rank * 0.2), 4))
    level = "high" if score >= 0.9 else "medium" if score >= 0.6 else "low"
    return {"score": score, "level": level}


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
                "gloss": None,
                "features": {
                    "gender": None,
                    "number": None,
                    "tense": None,
                    "person": None,
                    "case": None,
                    "definite": None,
                    "voice": None,
                },
                "segmentation": [surface],
                "dependency": {"head": None, "head_text": None, "deprel": None},
                "confidence": {"score": 0.0, "level": "low"},
                "meta": {
                    "source": "pyarabic",
                    "note": "pyarabic fallback - real AlKhalil bridge unavailable",
                },
                "normalized": True,
                "note": "pyarabic fallback - real AlKhalil bridge unavailable",
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
        "reason": reason,
        "input": text,
        "word_count": len(tokens),
        "tokens": tokens,
        "lemmas": lemmas,
    }


def _bridge_source() -> str:
    return r"""
import AlKhalil.analyse.Analyzer;
import AlKhalil.result.Result;
import AlKhalil.token.Tokens;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Iterator;
import java.util.HashMap;
import java.util.List;

public class AlKhalilBridge {
    private static String readAll(BufferedReader reader) throws Exception {
        StringBuilder sb = new StringBuilder();
        String line;
        boolean first = true;
        while ((line = reader.readLine()) != null) {
            if (!first) {
                sb.append('\n');
            }
            sb.append(line);
            first = false;
        }
        return sb.toString();
    }

    private static String jsonEscape(String value) {
        if (value == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder();
        sb.append('"');
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            switch (ch) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\b': sb.append("\\b"); break;
                case '\f': sb.append("\\f"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (ch < 0x20) {
                        sb.append(String.format("\\u%04x", (int) ch));
                    } else {
                        sb.append(ch);
                    }
            }
        }
        sb.append('"');
        return sb.toString();
    }

    private static String toJsonValue(String value) {
        return value == null || value.length() == 0 ? "null" : jsonEscape(value);
    }

    private static String buildAnalysisJson(Result result) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        sb.append("\"lemma\":").append(toJsonValue(result.getStem())).append(',');
        sb.append("\"root\":").append(toJsonValue(result.getWordroot())).append(',');
        sb.append("\"pos\":").append(toJsonValue(result.getPos())).append(',');
        sb.append("\"gender\":null,");
        sb.append("\"number\":null,");
        sb.append("\"tense\":null,");
        sb.append("\"gloss\":").append(toJsonValue(result.getWordtype())).append(',');
        sb.append("\"prefix\":").append(toJsonValue(result.getPrefix())).append(',');
        sb.append("\"stem\":").append(toJsonValue(result.getStem())).append(',');
        sb.append("\"type\":").append(toJsonValue(result.getWordtype())).append(',');
        sb.append("\"pattern\":").append(toJsonValue(result.getWordpattern())).append(',');
        sb.append("\"suffix\":").append(toJsonValue(result.getSuffix())).append(',');
        sb.append("\"priority\":").append(toJsonValue(result.getPriority()));
        sb.append('}');
        return sb.toString();
    }

    private static String buildTokenJson(String surface, List results, int tokenIndex) {
        StringBuilder sb = new StringBuilder();
        String bestLemma = null;
        String bestRoot = null;
        String bestPos = null;
        String bestGloss = null;

        sb.append('{');
        sb.append("\"surface\":").append(toJsonValue(surface)).append(',');
        sb.append("\"lemma\":null,");
        sb.append("\"root\":null,");
        sb.append("\"pos\":null,");
        sb.append("\"gloss\":null,");
        sb.append("\"features\":{");
        sb.append("\"gender\":null,");
        sb.append("\"number\":null,");
        sb.append("\"tense\":null,");
        sb.append("\"person\":null,");
        sb.append("\"case\":null,");
        sb.append("\"definite\":null,");
        sb.append("\"voice\":null");
        sb.append("},");
        sb.append("\"segmentation\":[");
        sb.append(toJsonValue(surface));
        sb.append("],");
        sb.append("\"dependency\":{\"head\":null,\"head_text\":null,\"deprel\":null},");
        sb.append("\"confidence\":").append("{\"score\":").append(String.format(java.util.Locale.US, "%.4f", Math.max(0.15, 1.0 - (tokenIndex * 0.2)))).append(",\"level\":\"").append(tokenIndex == 0 ? "high" : (tokenIndex == 1 ? "medium" : "low")).append("\"},");
        sb.append("\"meta\":{\"source\":\"alkhalil-java\",\"bridge\":\"Analyzer\",\"rank\":").append(tokenIndex).append("},");
        sb.append("\"normalized\":true,");
        sb.append("\"note\":\"real Analyzer bridge\",");
        sb.append("\"analyses\":[");

        if (results != null) {
            Iterator it = results.iterator();
            int analysisIndex = 0;
            while (it.hasNext() && analysisIndex < 3) {
                Object item = it.next();
                if (!(item instanceof Result)) {
                    continue;
                }
                Result result = (Result) item;
                String analysisJson = buildAnalysisJson(result);
                if (analysisIndex > 0) {
                    sb.append(',');
                }
                sb.append(analysisJson);
                if (analysisIndex == 0) {
                    bestLemma = result.getStem();
                    bestRoot = result.getWordroot();
                    bestPos = result.getPos();
                    bestGloss = result.getWordtype();
                }
                analysisIndex++;
            }
        }

        sb.append(']');

        if (bestLemma != null || bestRoot != null || bestPos != null || bestGloss != null) {
            sb.append(',');
            sb.append("\"lemma\":").append(toJsonValue(bestLemma)).append(',');
            sb.append("\"root\":").append(toJsonValue(bestRoot)).append(',');
            sb.append("\"pos\":").append(toJsonValue(bestPos)).append(',');
            sb.append("\"gloss\":").append(toJsonValue(bestGloss));
        } else {
            sb.append(',');
            sb.append("\"lemma\":null,");
            sb.append("\"root\":null,");
            sb.append("\"pos\":null,");
            sb.append("\"gloss\":null");
        }

        sb.append('}');
        return sb.toString();
    }

    private static String buildResponse(String text) throws Exception {
        Analyzer analyzer = new Analyzer();
        try {
            HashMap roots = analyzer.db.LoadRoots("db/AllRoots2.txt");
            if (roots != null && !roots.isEmpty()) {
                analyzer.VRoots = roots;
                analyzer.NRoots = roots;
            } else {
                roots = analyzer.db.LoadRoots("db/AllRoots1.txt");
                if (roots != null) {
                    analyzer.VRoots = roots;
                    analyzer.NRoots = roots;
                }
            }
        } catch (Exception ignored) {
        }

        Tokens tokens = new Tokens(text == null ? "" : text);
        List normalizedTokens = tokens.getNormalizedTokens();
        List unvoweledTokens = tokens.getUnvoweledTokens();

        StringBuilder sb = new StringBuilder();
        sb.append('{');
        sb.append("\"tool\":\"alkhalil\",");
        sb.append("\"status\":\"ok\",");
        sb.append("\"reason\":\"\",");
        sb.append("\"input\":").append(toJsonValue(text)).append(',');
        sb.append("\"word_count\":").append(normalizedTokens.size()).append(',');
        sb.append("\"tokens\":[");

        for (int i = 0; i < normalizedTokens.size(); i++) {
            String surface = (String) normalizedTokens.get(i);
            String unvoweled = (String) unvoweledTokens.get(i);
            List results = analyzer.Analyze(surface, unvoweled);
            if (i > 0) {
                sb.append(',');
            }
            sb.append(buildTokenJson(surface, results, i));
        }

        sb.append("],");
        sb.append("\"lemmas\":[");
        for (int i = 0; i < normalizedTokens.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            String normalized = (String) normalizedTokens.get(i);
            String unvoweled = (String) unvoweledTokens.get(i);
            List results = analyzer.Analyze(normalized, unvoweled);
            String lemma = null;
            if (results != null && !results.isEmpty()) {
                Object first = results.get(0);
                if (first instanceof Result) {
                    lemma = ((Result) first).getStem();
                }
            }
            sb.append(toJsonValue(lemma));
        }
        sb.append(']');
        sb.append('}');
        return sb.toString();
    }

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        OutputStreamWriter writer = new OutputStreamWriter(System.out, StandardCharsets.UTF_8);
        Analyzer analyzer = new Analyzer();
        writer.write("READY\n");
        writer.flush();
        String line;
        while ((line = reader.readLine()) != null) {
            String text = new String(Base64.getDecoder().decode(line), StandardCharsets.UTF_8);
            String response = buildResponse(text);
            writer.write(response);
            writer.write('\n');
            writer.flush();
        }
    }
}
""".strip()


def _bridge_workdir(jar_path: Path) -> Path:
    return jar_path.parent


def _bridge_dir() -> Path:
    global _bridge_helper_dir
    if _bridge_helper_dir is None:
        _bridge_helper_dir = Path(tempfile.gettempdir()) / "codex_alkhalil_bridge"
        _bridge_helper_dir.mkdir(parents=True, exist_ok=True)
    return _bridge_helper_dir


def _bridge_source_file() -> Path:
    global _bridge_source_path
    if _bridge_source_path is None:
        _bridge_source_path = _bridge_dir() / f"{_bridge_class_name}.java"
    return _bridge_source_path


def _compile_bridge(jar_path: Path) -> Path:
    source_path = _bridge_source_file()
    class_path = _bridge_dir() / f"{_bridge_class_name}.class"
    if not source_path.exists() or source_path.read_text(encoding="utf-8", errors="replace") != _bridge_source():
        source_path.write_text(_bridge_source(), encoding="utf-8")

    if class_path.exists() and class_path.stat().st_mtime >= source_path.stat().st_mtime:
        return class_path

    javac = shutil.which("javac")
    if not javac:
        raise RuntimeError("javac not found; cannot compile the AlKhalil bridge helper.")

    cmd = [
        javac,
        "-encoding",
        "UTF-8",
        "-cp",
        str(jar_path),
        str(source_path),
    ]
    logger.info("[AlKhalil] compiling bridge: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_bridge_dir()),
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"AlKhalil bridge compilation failed: {proc.stderr or proc.stdout}")
    return class_path


def _start_bridge_process(jar_path: Path) -> subprocess.Popen[str]:
    global _bridge_process, _bridge_ready, _bridge_error

    class_path = _compile_bridge(jar_path)
    helper_dir = class_path.parent
    cmd = [
        "java",
        "-Dfile.encoding=UTF-8",
        "-cp",
        os.pathsep.join([str(helper_dir), str(jar_path)]),
        _bridge_class_name,
    ]
    logger.info("[AlKhalil] starting bridge: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_bridge_workdir(jar_path)),
    )

    ready = proc.stdout.readline().strip() if proc.stdout else ""
    if ready != "READY":
        stderr = ""
        if proc.stderr:
            try:
                stderr = proc.stderr.read(4096)
            except Exception:
                stderr = ""
        proc.kill()
        raise RuntimeError(f"AlKhalil bridge failed to start: {ready or stderr or 'no READY banner'}")

    _bridge_process = proc
    _bridge_ready = True
    _bridge_error = None
    return proc


def _ensure_bridge_process(jar_path: Path) -> subprocess.Popen[str]:
    global _bridge_process, _bridge_ready

    if _bridge_process is not None and _bridge_process.poll() is None and _bridge_ready:
        return _bridge_process

    _bridge_ready = False
    _bridge_process = _start_bridge_process(jar_path)
    return _bridge_process


def _bridge_analyze(text: str, jar_path: Path) -> Dict[str, Any]:
    global _bridge_error, _bridge_process, _bridge_ready

    with _bridge_lock:
        proc = _ensure_bridge_process(jar_path)
        encoded = base64.b64encode((text or "").encode("utf-8")).decode("ascii")
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("AlKhalil bridge streams are unavailable.")

        try:
            proc.stdin.write(encoded + "\n")
            proc.stdin.flush()
            raw = proc.stdout.readline()
            if not raw:
                raise RuntimeError("AlKhalil bridge produced no output.")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("AlKhalil bridge returned an invalid payload.")
            return payload
        except Exception as exc:
            _bridge_error = str(exc)
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            _bridge_process = None
            _bridge_ready = False
            raise


def load_alkhalil() -> None:
    """Resolve the bundled AlKhalil JAR without starting the heavy bridge."""
    global alkhalil_jar_path
    existing = _alkhalil_paths.resolved_existing()
    if existing:
        alkhalil_jar_path = str(existing)
        logger.info("AlKhalil JAR found: %s", alkhalil_jar_path)
        return

    alkhalil_jar_path = None
    logger.warning("AlKhalil JAR not found (resolved_existing returned None).")


def get_alkhalil_status() -> Dict[str, Any]:
    jar = _alkhalil_paths.resolved_existing()
    java = shutil.which("java")
    if not java:
        return {
            "status": "missing_java",
            "reason": "AlKhalil requires Java.",
            "java": {"status": "missing_java", "reason": "Java executable was not found in PATH."},
            "integration": "java-bridge",
            "resolved_jar": str(_alkhalil_paths.resolve()),
            "jar_exists": bool(jar and jar.exists() and jar.is_file()),
        }

    if jar is None or not jar.exists() or not jar.is_file():
        return {
            "status": "missing_model",
            "reason": "AlKhalil JAR not found. Set ALKHALIL_JAR or ensure the jar exists under app/tools/alkhalil/AlKhalil1.1/AlKhalil.jar.",
            "integration": "java-bridge",
            "resolved_jar": str(_alkhalil_paths.resolve()),
            "jar_exists": False,
        }

    return {
        "status": "ok",
        "reason": "Bundled AlKhalil Analyzer is available through the Java bridge.",
        "integration": "java-bridge",
        "resolved_jar": str(jar),
        "jar_exists": True,
    }


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


def _finalize_alkhalil_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the final AlKhalil payload so `pos` is UPOS and the raw tag is preserved."""
    if not isinstance(result, dict):
        return result

    tokens = result.get("tokens")
    if not isinstance(tokens, list):
        return result

    finalized_tokens: List[Dict[str, Any]] = []
    for tok in tokens:
        if not isinstance(tok, dict):
            finalized_tokens.append(tok)
            continue

        token = dict(tok)
        raw_pos = token.get("pos_raw") or token.get("raw_pos") or token.get("pos") or token.get("upos")
        normalized_pos = _normalize_alkhalil_pos(raw_pos)

        if raw_pos:
            token["pos_raw"] = raw_pos
        if normalized_pos:
            token["pos"] = normalized_pos
        elif token.get("upos"):
            token["pos"] = token.get("upos")
        else:
            token["pos"] = None

        if isinstance(token.get("analyses"), list):
            for analysis in token["analyses"]:
                if isinstance(analysis, dict) and token.get("pos"):
                    analysis["pos"] = token["pos"]

        finalized_tokens.append(token)

    finalized = dict(result)
    finalized["tokens"] = finalized_tokens
    return finalized


def alkhalil_analyze(text: str) -> Dict[str, Any]:
    tool = "alkhalil"
    try:
        global alkhalil_jar_path
        if alkhalil_jar_path is None:
            load_alkhalil()

        jar_path = _alkhalil_paths.resolved_existing()
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

        if os.environ.get("ALKHALIL_JAR") is None:
            os.environ["ALKHALIL_JAR"] = str(jar_path)

        try:
            bridge_result = _bridge_analyze(text or "", jar_path)
            if bridge_result.get("status") == "ok" and bridge_result.get("tokens"):
                return _finalize_alkhalil_result(bridge_result)
            bridge_reason = bridge_result.get("reason") or "AlKhalil bridge returned no tokens."
            logger.warning("[AlKhalil] bridge returned fallback-worthy result: %s", bridge_reason)
        except Exception as exc:
            bridge_reason = str(exc)
            logger.warning("[AlKhalil] bridge failed: %s", bridge_reason)

        return _finalize_alkhalil_result(_pyarabic_fallback(text or "", f"Real AlKhalil bridge failed: {bridge_reason}"))

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
