from __future__ import annotations

import urllib.request
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logger import logger
from app.utils.helpers import strip_diacritics
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

UDPIPE_DOWNLOAD_URLS = (
    "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4964/arabic-padt-ud-2.10-220711.udpipe",
    "https://ufal.mff.cuni.cz/~straka/papers/2017-conll_udpipe/udpipe-ud-2.0-conll17-170315-1-udpipe_models/arabic-ud-2.0-conll17-170315.udpipe",
    "https://github.com/jwijffels/udpipe.models.ud.2.0/raw/master/inst/udpipe-ud-2.0-170801/arabic-ud-2.0-170801.udpipe",
    "https://huggingface.co/datasets/universal-dependencies/arabic-padt/resolve/main/arabic-padt-ud-2.6-200830.udpipe",
)
UDPIPE_DOWNLOAD_PATH = Path(__file__).resolve().parent / "udpipe" / "arabic.udpipe"
UDPIPE_MANUAL_HELP = "Download manually from: https://ufal.mff.cuni.cz/udpipe/2/models and place the file in app/tools/udpipe/"


def _resolve_model_path() -> Optional[Path]:
    # Centralized resolver with strict search order.
    return _udp_paths.resolved_existing()


def _set_status(status: str, reason: str, model_path: Optional[Path] = None) -> None:
    global _udpipe_status
    _udpipe_status = {"status": status, "reason": reason}
    if model_path is not None:
        _udpipe_status["path"] = str(model_path)


def _download_candidate(url: str, target: Path) -> Optional[Path]:
    tmp_target = target.with_suffix(target.suffix + ".part")
    logger.info("[UDPIPE] trying download url: %s", url)
    try:
        with urllib.request.urlopen(url, timeout=30) as response, open(tmp_target, "wb") as handle:
            total = response.headers.get("Content-Length")
            expected = int(total) if total and total.isdigit() else None
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if expected:
                    logger.info("[UDPIPE] download progress: %s/%s bytes", downloaded, expected)
                else:
                    logger.info("[UDPIPE] download progress: %s bytes", downloaded)

        size = tmp_target.stat().st_size
        if size < 1_000_000:
            logger.warning("[UDPIPE] downloaded file too small (%s bytes), likely an error page", size)
            tmp_target.unlink(missing_ok=True)
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target.replace(target)
        logger.info("[UDPIPE] model downloaded to %s", target)
        return target
    except Exception as exc:
        logger.warning("[UDPIPE] download failed from %s: %s", url, exc)
        tmp_target.unlink(missing_ok=True)
        return None


def download_model() -> Path:
    """Download the Arabic UDPipe model to the local tools directory."""
    target = UDPIPE_DOWNLOAD_PATH
    for url in UDPIPE_DOWNLOAD_URLS:
        downloaded = _download_candidate(url, target)
        if downloaded is not None:
            return downloaded
    raise RuntimeError(
        "UDPipe model download failed. "
        f"{UDPIPE_MANUAL_HELP}"
    )


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
        try:
            model_path = download_model()
        except Exception as exc:
            message = (
                "UDPipe model not found and auto-download failed. "
                f"{UDPIPE_MANUAL_HELP}. Error: {exc}"
            )
            logger.warning(message)
            _set_status("unavailable", message)
            return

    if model_path is None or not model_path.exists():
        message = (
            "UDPipe model not found. Set UDPIPE_MODEL or place an arabic*.udpipe file under "
            "app/tools/udpipe, app/tools/udpipe/models, or C:/Users/*/Desktop/*/tools/udpipe. "
            f"{UDPIPE_MANUAL_HELP}."
        )
        logger.warning(message)
        _set_status("unavailable", message)
        return

    try:
        import ufal.udpipe

        udpipe_model = ufal.udpipe.Model.load(str(model_path))
        if udpipe_model is None:
            _set_status(
                "unavailable",
                f"UDPipe model not found or could not be loaded from {model_path}",
                model_path,
            )
            udpipe_model = None
            udpipe_pipeline_obj = None
            udpipe_model_path = None
            return

        # Keep the pipeline instance hot for all requests.
        pipeline_cls = ufal.udpipe.Pipeline
        udpipe_pipeline_obj = pipeline_cls(udpipe_model, "tokenize", pipeline_cls.DEFAULT, pipeline_cls.DEFAULT, "conllu")
        udpipe_model_path = model_path
        logger.info("UDPipe model loaded from %s", model_path)
        _set_status("ok", f"UDPipe model loaded from {model_path}", model_path)
    except FileNotFoundError as exc:
        udpipe_model = None
        udpipe_pipeline_obj = None
        udpipe_model_path = None
        logger.warning("UDPipe model missing at %s: %s", model_path, exc)
        _set_status("unavailable", f"UDPipe model missing at {model_path}", model_path)
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


def fix_udpipe_lemma(surface: Optional[str], lemma: Optional[str]) -> Optional[str]:
    if not lemma:
        return lemma
    if not surface:
        return lemma

    surface_text = str(surface)
    lemma_text = str(lemma)

    if surface_text.startswith("ال") and lemma_text.startswith("أل"):
        lemma_text = "ا" + lemma_text[1:]

    surface_no_diac = strip_diacritics(surface_text)
    lemma_no_diac = strip_diacritics(lemma_text)
    if surface_no_diac.endswith("أ") and lemma_no_diac.endswith("ا"):
        lemma_text = lemma_text[:-1] + "أ"
    elif surface_no_diac.endswith("إ") and lemma_no_diac.endswith("ا"):
        lemma_text = lemma_text[:-1] + "إ"

    return lemma_text


def _fix_udpipe_lemma(surface: Optional[str], lemma: Optional[str]) -> Optional[str]:
    return fix_udpipe_lemma(surface, lemma)


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

    lemma = fix_udpipe_lemma(surface, getattr(word, "lemma", None))
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
        "pos": None if upos is None else str(upos),
        "upos": None if upos is None else str(upos),
        "case": _parse_features(feats).get("case"),
        "dependency": dependency,
        "analyses": [
            {
                "lemma": None if lemma is None else str(lemma),
                "root": None,
                "pos": None if upos is None else str(upos),
                "gender": None,
                "number": None,
                "tense": None,
                "gloss": None,
            }
        ],
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
            "lemma": fix_udpipe_lemma(str(surface), None if lemma is None else str(lemma)),
            "pos": None if upos is None else str(upos),
            "upos": None if upos is None else str(upos),
            "case": _parse_features(feats).get("case"),
            "dependency": {
                "head": head,
                "head_text": None,
                "deprel": None if deprel in {"", "_"} else str(deprel),
            },
            "analyses": [
                {
                    "lemma": None if lemma is None else str(lemma),
                    "root": None,
                    "pos": None if upos is None else str(upos),
                    "gender": None,
                    "number": None,
                    "tense": None,
                    "gloss": None,
                }
            ],
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
