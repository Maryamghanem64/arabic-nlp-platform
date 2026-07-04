from __future__ import annotations

import os
import pickle
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from app.utils.logger import logger
except Exception:
    import logging
    logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "app" / "tools" / "sinatools"

LEMMA_PICKLE_NAME = "lemma.pickle"
PACKAGE_LEMMA_PICKLE_NAME = "lemmas_dic.pickle"

_dictionary: Dict[str, Any] = {}
_sinatools_loaded = False
_sinatools_loading = False
_sinatools_last_error: Optional[str] = None
_sinatools_resource_path: Optional[Path] = None
_sinatools_loader_thread: Optional[threading.Thread] = None
_sinatools_runtime_ms: Optional[int] = None
_sinatools_progress_label = ""
_sinatools_lock = threading.RLock()


def _appdata_dir() -> Path:
    return Path.home() / "AppData" / "Roaming" / "sinatools"


def _lemma_candidates() -> List[Path]:
    candidates = [
        RESOURCE_ROOT / PACKAGE_LEMMA_PICKLE_NAME,
        RESOURCE_ROOT / LEMMA_PICKLE_NAME,
        Path("C:/Users/lenovo/AppData/Roaming/sinatools/lemmas_dic.pickle"),
        _appdata_dir() / PACKAGE_LEMMA_PICKLE_NAME,
        _appdata_dir() / LEMMA_PICKLE_NAME,
    ]

    configured = os.environ.get("SINATOOLS_LEMMA_PICKLE") or os.environ.get("SINATOOLS_LEMMAS_PICKLE")
    if configured:
        candidates.insert(0, Path(configured).expanduser())

    seen = set()
    result = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _resolve_lemma_pickle() -> Optional[Path]:
    for path in _lemma_candidates():
        if path.exists() and path.is_file():
            return path
    return None


def _normalize_token(token: str) -> str:
    text = str(token or "").strip()
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = text.replace("\u0640", "")
    return text


def _tokenize(text: str) -> List[str]:
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize
        return [t for t in simple_word_tokenize(text or "") if t.strip()]
    except Exception:
        return [t for t in str(text or "").split() if t.strip()]


def _solution_to_analysis(token: str, solution: Any) -> Dict[str, Any]:
    if isinstance(solution, dict):
        lemma = solution.get("lemma") or solution.get("lem") or token
        root = solution.get("root") or token
        pos = solution.get("pos") or solution.get("POS") or None
        lemma_id = solution.get("lemma_id") or solution.get("id") or None
        frequency = solution.get("frequency") or solution.get("freq") or 0
    elif isinstance(solution, (list, tuple)):
        frequency = solution[1] if len(solution) > 1 else 0
        lemma = solution[2] if len(solution) > 2 else token
        lemma_id = solution[3] if len(solution) > 3 else None
        root = solution[4] if len(solution) > 4 else token
        pos = solution[5] if len(solution) > 5 else None
    else:
        lemma = token
        root = token
        pos = None
        lemma_id = None
        frequency = 0

    try:
        frequency = int(frequency)
    except Exception:
        frequency = 0

    return {
        "lemma": None if lemma in {"", "0", 0} else str(lemma),
        "lemma_id": None if lemma_id in {"", "0", 0, None} else str(lemma_id),
        "root": None if root in {"", "0", 0} else str(root),
        "pos": None if not pos else str(pos),
        "gender": None,
        "number": None,
        "tense": None,
        "gloss": None,
        "frequency": frequency,
    }


def _lookup(token: str) -> List[Any]:
    normalized = _normalize_token(token)
    variants = [
        normalized,
        normalized[2:] if normalized.startswith("ال") and len(normalized) > 3 else normalized,
        normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا"),
    ]

    seen = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        solutions = _dictionary.get(variant)
        if solutions:
            return solutions if isinstance(solutions, list) else [solutions]
    return []


def load_sinatools() -> bool:
    global _dictionary, _sinatools_loaded, _sinatools_loading
    global _sinatools_last_error, _sinatools_resource_path
    global _sinatools_runtime_ms, _sinatools_progress_label

    with _sinatools_lock:
        if _sinatools_loaded:
            return True
    

        started = time.time()
        _sinatools_loading = True
        _sinatools_progress_label = "loading"
        _sinatools_last_error = None

    path = _resolve_lemma_pickle()

    if path is None:
        with _sinatools_lock:
            _dictionary = {}
            _sinatools_loaded = False
            _sinatools_loading = False
            _sinatools_resource_path = None
            _sinatools_last_error = f"Missing SinaTools lemma pickle under {RESOURCE_ROOT}"
            _sinatools_runtime_ms = int((time.time() - started) * 1000)
            _sinatools_progress_label = "missing_model"
        return False

    try:
        with path.open("rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise RuntimeError(f"Expected dict in {path}, got {type(data).__name__}")

        with _sinatools_lock:
            _dictionary = data
            _sinatools_loaded = True
            _sinatools_loading = False
            _sinatools_resource_path = path
            _sinatools_last_error = None
            _sinatools_runtime_ms = int((time.time() - started) * 1000)
            _sinatools_progress_label = "loaded"

        logger.info("SinaTools loaded from %s", path)
        return True

    except Exception as exc:
        with _sinatools_lock:
            _dictionary = {}
            _sinatools_loaded = False
            _sinatools_loading = False
            _sinatools_resource_path = path
            _sinatools_last_error = str(exc)
            _sinatools_runtime_ms = int((time.time() - started) * 1000)
            _sinatools_progress_label = "error"

        logger.warning("SinaTools failed to load: %s", exc)
        return False


def _background_load() -> None:
    time.sleep(0.05)
    load_sinatools()


def start_sinatools_background_loading() -> bool:
    global _sinatools_loader_thread, _sinatools_loading, _sinatools_progress_label

    if _sinatools_loaded:
        return True

    with _sinatools_lock:
        if _sinatools_loaded:
            return True

        if _sinatools_loader_thread and _sinatools_loader_thread.is_alive():
            _sinatools_loading = True
            return True

        _sinatools_loading = True
        _sinatools_progress_label = "loading"

        _sinatools_loader_thread = threading.Thread(
            target=_background_load,
            name="sinatools-loader",
            daemon=True,
        )
        _sinatools_loader_thread.start()
        return True


def get_sinatools_status_detail() -> Dict[str, Any]:
    path = _resolve_lemma_pickle()

    if _sinatools_loaded:
        status = "loaded"
        reason = f"SinaTools loaded from {_sinatools_resource_path}"
    elif _sinatools_loading:
        status = "loading"
        reason = "SinaTools model is loading in background."
    elif path:
        status = "lazy_not_loaded"
        reason = f"SinaTools model is present at {path}, but not loaded yet."
    else:
        status = "error"
        reason = f"Missing SinaTools lemma pickle. Put it under {RESOURCE_ROOT}"

    return {
        "status": status,
        "reason": reason,
        "model_present": path is not None,
        "model_path": str(path) if path else None,
        "loaded": _sinatools_loaded,
        "loading": _sinatools_loading,
        "last_error": _sinatools_last_error,
        "runtime_ms": _sinatools_runtime_ms,
        "progress_label": _sinatools_progress_label,
        "dictionary_size": len(_dictionary) if _sinatools_loaded else 0,
    }


def sinatools_analyze(text: str) -> Dict[str, Any]:
    started = time.time()

    if not _sinatools_loaded:
        start_sinatools_background_loading()
        detail = get_sinatools_status_detail()
        return {
            "tool": "sinatools",
            "status": "loading" if detail.get("loading") else "lazy_not_loaded",
            "reason": detail.get("reason"),
            "tokens": [],
            "lemmas": [],
            "pos": [],
            "input": text,
            "word_count": 0,
            "runtime_ms": int((time.time() - started) * 1000),
            "meta": detail,
        }

    tokens = []
    lemmas = []
    pos_tags = []

    for surface in _tokenize(text):
        solutions = _lookup(surface)
        analyses = [_solution_to_analysis(surface, s) for s in solutions[:3]]

        if not analyses:
            analyses = [_solution_to_analysis(surface, [surface, 0, surface, 0, surface, ""])]

        best = analyses[0]

        token_payload = {
            "surface": surface,
            "lemma": best.get("lemma"),
            "root": best.get("root"),
            "pos": best.get("pos"),
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
            "confidence": {
                "score": 1.0 if solutions else 0.0,
                "level": "high" if solutions else "low",
            },
            "meta": {
                "source": "sinatools-lemma-pickle",
                "frequency": best.get("frequency"),
                "lemma_id": best.get("lemma_id"),
                "matched": bool(solutions),
            },
            "analyses": analyses,
        }

        tokens.append(token_payload)
        lemmas.append(best.get("lemma"))
        pos_tags.append(best.get("pos"))

    return {
        "tool": "sinatools",
        "status": "ok",
        "reason": "",
        "tokens": tokens,
        "lemmas": lemmas,
        "pos": pos_tags,
        "input": text,
        "word_count": len(tokens),
        "runtime_ms": int((time.time() - started) * 1000),
        "meta": get_sinatools_status_detail(),
    }


class SinaToolsTool:
    tool_name = "sinatools"

    def is_loaded(self) -> bool:
        return _sinatools_loaded

    def load(self) -> bool:
        return load_sinatools()

    def get_status(self) -> Dict[str, Any]:
        return get_sinatools_status_detail()

    def analyze(self, text: str) -> Dict[str, Any]:
        return sinatools_analyze(text)


sinatools_tool = SinaToolsTool()


__all__ = [
    "SinaToolsTool",
    "get_sinatools_status_detail",
    "load_sinatools",
    "start_sinatools_background_loading",
    "sinatools_analyze",
    "sinatools_tool",
]