from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logger import logger
from backend.config.tool_paths import UDPipePaths


udpipe_model = None
udpipe_pipeline_obj = None
udpipe_model_path: Optional[Path] = None
udpipe_lock = threading.Lock()

_udp_paths = UDPipePaths()
_udpipe_load_attempted = False
_udpipe_status: Dict[str, Any] = {
    "status": "unavailable",
    "reason": "UDPipe has not been initialized yet.",
}


def _resolve_model_path() -> Optional[Path]:
    # Centralized resolver with strict search order.
    return _udp_paths.resolved_existing()


def _set_status(status: str, reason: str, model_path: Optional[Path] = None) -> None:
    global _udpipe_status
    _udpipe_status = {"status": status, "reason": reason}
    if model_path is not None:
        _udpipe_status["path"] = str(model_path)


def load_udpipe() -> None:
    """Load UDPipe model once and cache the pipeline in module state."""

    global udpipe_model, udpipe_pipeline_obj, udpipe_model_path, _udpipe_load_attempted

    if udpipe_pipeline_obj is not None:
        _set_status("ok", f"UDPipe model loaded from {udpipe_model_path}", udpipe_model_path)
        return
    if _udpipe_load_attempted:
        return

    _udpipe_load_attempted = True

    model_path = _resolve_model_path()
    if model_path is None:
        message = (
            "UDPipe model not found. Set UDPIPE_MODEL or place an arabic*.udpipe file under "
            "app/tools/udpipe, app/tools/udpipe/models, or C:/Users/*/Desktop/*/tools/udpipe."
        )
        logger.warning(message)
        _set_status("unavailable", message)
        return

    try:
        import ufal.udpipe

        udpipe_model = ufal.udpipe.Model.load(str(model_path))
        if udpipe_model is None:
            raise RuntimeError(f"UDPipe model load returned None for {model_path}")

        # Keep the pipeline instance hot for all requests.
        udpipe_pipeline_obj = ufal.udpipe.Pipeline(udpipe_model, "tokenize", "tag", "parse")
        udpipe_model_path = model_path
        logger.info("UDPipe model loaded from %s", model_path)
        _set_status("ok", f"UDPipe model loaded from {model_path}", model_path)
    except Exception as exc:
        udpipe_model = None
        udpipe_pipeline_obj = None
        udpipe_model_path = None
        logger.warning("UDPipe load failed from %s: %s", model_path, exc)
        _set_status("error", f"UDPipe internal error while loading {model_path}: {exc}", model_path)


def get_udpipe_status() -> Dict[str, Any]:
    if udpipe_pipeline_obj is not None:
        return {
            "status": "ok",
            "reason": f"UDPipe model loaded from {udpipe_model_path}" if udpipe_model_path else "UDPipe loaded.",
            "path": str(udpipe_model_path) if udpipe_model_path else None,
        }

    if not _udpipe_load_attempted:
        load_udpipe()

    payload = dict(_udpipe_status)
    if udpipe_model_path is not None and "path" not in payload:
        payload["path"] = str(udpipe_model_path)
    return payload


def _parse_features(raw: Any) -> Dict[str, Optional[str]]:
    case = None
    if isinstance(raw, str):
        for item in raw.split("|"):
            if item.startswith("Case="):
                case = item.split("=", 1)[1] or None
                break
    elif isinstance(raw, dict):
        case = raw.get("Case") or raw.get("case")
    return {"case": case}


def _token_from_word(word: Any, *, id_to_surface: Dict[int, str]) -> Optional[Dict[str, Any]]:
    surface = getattr(word, "form", None)
    if surface is None:
        surface = getattr(word, "surface", None)
    if surface is None:
        surface = getattr(word, "text", None)
    if surface is None:
        return None

    token_id = getattr(word, "id", None)
    try:
        token_id_int = int(token_id) if token_id is not None else None
    except Exception:
        token_id_int = None
    if token_id_int is not None and token_id_int <= 0:
        return None

    lemma = getattr(word, "lemma", None)
    upos = getattr(word, "upos", None)
    head = getattr(word, "head", None)
    deprel = getattr(word, "deprel", None)
    feats = getattr(word, "feats", None)
    if feats is None:
        feats = getattr(word, "feats_str", None)

    head_value: Optional[int]
    try:
        head_value = int(head) if head is not None and str(head).strip() != "" else None
    except Exception:
        head_value = None

    dependency = {
        "head": head_value,
        "head_text": id_to_surface.get(head_value) if head_value is not None else None,
        "deprel": None if deprel is None else str(deprel),
    }

    return {
        "surface": str(surface),
        "lemma": None if lemma is None else str(lemma),
        "upos": None if upos is None else str(upos),
        "case": _parse_features(feats).get("case"),
        "dependency": dependency,
    }


def _extract_tokens(processed: Any) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []

    if processed is None:
        return tokens

    if isinstance(processed, str):
        return _parse_conllu(processed)

    try:
        iterable = list(processed)
    except TypeError:
        iterable = [processed]

    for sentence in iterable:
        if sentence is None:
            continue

        words = getattr(sentence, "words", None)
        if words:
            id_to_surface: Dict[int, str] = {}
            sentence_tokens: List[Dict[str, Any]] = []
            for word in words:
                token = _token_from_word(word, id_to_surface={})
                if not token:
                    continue
                try:
                    token_id = int(getattr(word, "id", 0))
                except Exception:
                    token_id = 0
                if token_id > 0:
                    id_to_surface[token_id] = token["surface"]
                    sentence_tokens.append(token)
            for token in sentence_tokens:
                head = token.get("dependency", {}).get("head")
                token["dependency"]["head_text"] = id_to_surface.get(head) if isinstance(head, int) else None
            tokens.extend(sentence_tokens)
            continue

        # Fallback for iterable sentence/token containers.
        sentence_words: List[Any]
        try:
            sentence_words = list(sentence)
        except TypeError:
            sentence_words = []

        id_to_surface = {}
        sentence_tokens = []
        for word in sentence_words:
            token = _token_from_word(word, id_to_surface={})
            if not token:
                continue
            try:
                token_id = int(getattr(word, "id", 0))
            except Exception:
                token_id = 0
            if token_id > 0:
                id_to_surface[token_id] = token["surface"]
                sentence_tokens.append(token)
        for token in sentence_tokens:
            head = token.get("dependency", {}).get("head")
            token["dependency"]["head_text"] = id_to_surface.get(head) if isinstance(head, int) else None
        tokens.extend(sentence_tokens)

    return tokens


def _parse_conllu(conllu_text: str) -> List[Dict[str, Any]]:
    """Parse CoNLL-U text into the required token schema."""

    rows: List[Dict[str, Any]] = []
    current_rows: List[Dict[str, Any]] = []
    id_to_surface: Dict[int, str] = {}

    def flush_rows() -> None:
        nonlocal current_rows, id_to_surface
        for row in current_rows:
            head = row["dependency"].get("head")
            if isinstance(head, int):
                row["dependency"]["head_text"] = id_to_surface.get(head)
        rows.extend(current_rows)
        current_rows = []
        id_to_surface = {}

    for line in (conllu_text or "").splitlines():
        line = line.strip()
        if not line:
            flush_rows()
            continue
        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 8:
            continue
        token_id_raw = parts[0]
        if "-" in token_id_raw or "." in token_id_raw:
            continue

        try:
            token_id = int(token_id_raw)
        except Exception:
            continue

        surface = parts[1] if parts[1] != "_" else None
        lemma = parts[2] if parts[2] != "_" else None
        upos = parts[3] if parts[3] != "_" else None
        feats = parts[5] if len(parts) > 5 else None
        head_raw = parts[6] if len(parts) > 6 else "_"
        deprel = parts[7] if len(parts) > 7 else "_"

        try:
            head = int(head_raw) if head_raw not in {"", "_"} else None
        except Exception:
            head = None

        if surface is None:
            continue

        token = {
            "surface": str(surface),
            "lemma": None if lemma is None else str(lemma),
            "upos": None if upos is None else str(upos),
            "case": _parse_features(feats).get("case"),
            "dependency": {
                "head": head,
                "head_text": None,
                "deprel": None if deprel in {"", "_"} else str(deprel),
            },
        }
        current_rows.append(token)
        if token_id > 0:
            id_to_surface[token_id] = token["surface"]

    if current_rows:
        flush_rows()

    return rows


def udpipe_analyze(text: str) -> Dict[str, Any]:
    tool = "udpipe"

    try:
        global udpipe_pipeline_obj

        status = get_udpipe_status()
        if status.get("status") != "ok":
            return {
                "tool": tool,
                "status": status.get("status", "unavailable"),
                "reason": status.get("reason", "UDPipe model/pipeline not available"),
                "input": text,
                "word_count": 0,
                "tokens": [],
            }

        if udpipe_pipeline_obj is None:
            load_udpipe()
            status = get_udpipe_status()
            if status.get("status") != "ok":
                return {
                    "tool": tool,
                    "status": status.get("status", "unavailable"),
                    "reason": status.get("reason", "UDPipe model/pipeline not available"),
                    "input": text,
                    "word_count": 0,
                    "tokens": [],
                }

        sentence = text or ""

        with udpipe_lock:
            processed = udpipe_pipeline_obj.process(sentence)

        tokens = _extract_tokens(processed)
        if not tokens:
            return {
                "tool": tool,
                "status": "error",
                "reason": "UDPipe output produced no tokens",
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

    except TimeoutError as exc:
        logger.warning("[UDPIPE] timeout: %s", exc)
        return {
            "tool": tool,
            "status": "error",
            "reason": f"UDPipe timeout: {exc}",
            "input": text,
            "word_count": 0,
            "tokens": [],
        }
    except Exception as exc:
        logger.warning("[UDPIPE] error: %s", exc)
        return {
            "tool": tool,
            "status": "error",
            "reason": str(exc),
            "input": text,
            "word_count": 0,
            "tokens": [],
        }

