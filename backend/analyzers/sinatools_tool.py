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
    configured = os.environ.get("SINATOOLS_LEMMA_PICKLE") or os.environ.get("SINATOOLS_LEMMAS_PICKLE")

    candidates: List[Path] = []

    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            RESOURCE_ROOT / PACKAGE_LEMMA_PICKLE_NAME,
            RESOURCE_ROOT / LEMMA_PICKLE_NAME,
            RESOURCE_ROOT / "resources" / PACKAGE_LEMMA_PICKLE_NAME,
            RESOURCE_ROOT / "resources" / LEMMA_PICKLE_NAME,
            _appdata_dir() / PACKAGE_LEMMA_PICKLE_NAME,
            _appdata_dir() / LEMMA_PICKLE_NAME,
        ]
    )

    seen = set()
    result: List[Path] = []

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


def _strip_diacritics(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = text.replace("\u0640", "")
    return text.strip()


def _normalize_arabic_letters(value: Any) -> str:
    text = _strip_diacritics(value)
    text = text.replace("ٱ", "ا")
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")
    return text.strip()


def _normalize_token(token: str) -> str:
    return _normalize_arabic_letters(token)


def _tokenize(text: str) -> List[str]:
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize

        return [t for t in simple_word_tokenize(text or "") if t.strip()]
    except Exception:
        return [t for t in str(text or "").split() if t.strip()]


def _split_lemma_alternatives(lemma: Any) -> List[str]:
    text = str(lemma or "").strip()

    if not text:
        return []

    parts = [p.strip() for p in text.split("|") if p.strip()]
    return parts or [text]


def _choose_display_lemma(lemma: Any) -> Optional[str]:
    alternatives = _split_lemma_alternatives(lemma)

    if not alternatives:
        return None

    # Prefer the shortest clean lemma because SinaTools sometimes stores:
    # "كَتَبَ | كَتَبَ إلى | كَتَبَ في | كَتَبَ ل"
    alternatives = sorted(alternatives, key=lambda x: (len(_strip_diacritics(x)), len(x)))
    selected = alternatives[0].strip()

    return selected or None


def _normalize_root(root: Any) -> Optional[str]:
    text = _normalize_arabic_letters(root)

    if not text or text in {"0", "#", "null", "None"}:
        return None

    text = re.sub(r"[.\s\-ـ]+", ".", text).strip(".")
    letters = [p for p in text.split(".") if p]

    if not letters:
        compact = re.sub(r"[.\s\-ـ]+", "", text)
        if len(compact) >= 2:
            return ".".join(list(compact))
        return None

    return ".".join(letters)


def _normalize_pos(pos: Any) -> Optional[str]:
    raw = str(pos or "").strip()

    if not raw or raw in {"0", "#", "null", "None"}:
        return None

    if "فعل" in raw:
        return "VERB"

    if "اسم علم" in raw:
        return "PROPN"

    if "صفة" in raw:
        return "ADJ"

    if "حرف جر" in raw:
        return "ADP"

    if "ضمير" in raw:
        return "PRON"

    if "اسم" in raw:
        return "NOUN"

    if "حرف" in raw:
        return "PART"

    upper = raw.upper()
    if upper in {"NOUN", "VERB", "ADJ", "ADV", "ADP", "PRON", "PART", "PROPN", "NUM"}:
        return upper

    return raw


def _lookup_variants(token: str) -> List[str]:
    normalized = _normalize_token(token)

    variants = [
        normalized,
        normalized[2:] if normalized.startswith("ال") and len(normalized) > 3 else normalized,
    ]

    # simple clitic-aware variants for common Arabic proclitics
    for prefix in ("و", "ف", "ب", "ك", "ل"):
        if normalized.startswith(prefix) and len(normalized) > 2:
            variants.append(normalized[1:])

    # definite article after proclitic: بالمدرسة -> مدرسة
    for prefix in ("وال", "فال", "بال", "كال", "لال"):
        if normalized.startswith(prefix) and len(normalized) > len(prefix) + 1:
            variants.append(normalized[len(prefix):])

    # common pronominal suffix variants
    for suffix in ("ها", "ه", "هم", "هما", "نا", "كم", "كن", "ي"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
            variants.append(normalized[: -len(suffix)])

    variants.append(normalized.replace("ة", "ه"))

    seen = set()
    clean: List[str] = []

    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        clean.append(variant)

    return clean


def _lookup(token: str) -> List[Any]:
    for variant in _lookup_variants(token):
        solutions = _dictionary.get(variant)
        if solutions:
            if isinstance(solutions, list):
                return _sort_solutions(solutions)
            return [solutions]
    return []


def _solution_frequency(solution: Any) -> int:
    try:
        if isinstance(solution, dict):
            return int(solution.get("frequency") or solution.get("freq") or 0)

        if isinstance(solution, (list, tuple)) and len(solution) > 1:
            return int(solution[1] or 0)
    except Exception:
        return 0

    return 0


def _sort_solutions(solutions: List[Any]) -> List[Any]:
    return sorted(solutions, key=_solution_frequency, reverse=True)


def _solution_to_analysis(token: str, solution: Any, *, matched: bool = True) -> Dict[str, Any]:
    if isinstance(solution, dict):
        raw_lemma = solution.get("lemma") or solution.get("lem") or None
        raw_root = solution.get("root") or None
        raw_pos = solution.get("pos") or solution.get("POS") or None
        lemma_id = solution.get("lemma_id") or solution.get("id") or None
        frequency = solution.get("frequency") or solution.get("freq") or 0

    elif isinstance(solution, (list, tuple)):
        frequency = solution[1] if len(solution) > 1 else 0
        raw_lemma = solution[2] if len(solution) > 2 else None
        lemma_id = solution[3] if len(solution) > 3 else None
        raw_root = solution[4] if len(solution) > 4 else None
        raw_pos = solution[5] if len(solution) > 5 else None

    else:
        raw_lemma = None
        raw_root = None
        raw_pos = None
        lemma_id = None
        frequency = 0

    try:
        frequency = int(frequency)
    except Exception:
        frequency = 0

    lemma_alternatives = _split_lemma_alternatives(raw_lemma)
    lemma = _choose_display_lemma(raw_lemma)
    root = _normalize_root(raw_root)
    pos = _normalize_pos(raw_pos)

    # Do not fabricate linguistic values if the dictionary did not match.
    if not matched:
        lemma = None
        root = None
        pos = None
        lemma_alternatives = []

    return {
        "lemma": lemma,
        "lemma_raw": None if raw_lemma in {"", "0", 0, None} else str(raw_lemma),
        "lemma_alternatives": lemma_alternatives,
        "lemma_id": None if lemma_id in {"", "0", 0, None} else str(lemma_id),
        "root": root,
        "root_raw": None if raw_root in {"", "0", 0, None, "#"} else str(raw_root),
        "pos": pos,
        "pos_raw": None if raw_pos in {"", "0", 0, None, "#"} else str(raw_pos),
        "gender": None,
        "number": None,
        "tense": None,
        "gloss": None,
        "frequency": frequency,
        "matched": matched,
    }


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
        logger.info("SinaTools loading started from %s", path)

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

        logger.info("SinaTools loaded from %s with %s entries", path, len(data))
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
    load_sinatools()


def start_sinatools_background_loading() -> bool:
    global _sinatools_loader_thread, _sinatools_loading, _sinatools_progress_label
    global _sinatools_last_error

    with _sinatools_lock:
        if _sinatools_loaded:
            return True

        if _sinatools_loader_thread and _sinatools_loader_thread.is_alive():
            _sinatools_loading = True
            _sinatools_progress_label = "loading"
            return True

        _sinatools_loading = True
        _sinatools_progress_label = "loading"
        _sinatools_last_error = None

        _sinatools_loader_thread = threading.Thread(
            target=_background_load,
            name="sinatools-loader",
            daemon=True,
        )
        _sinatools_loader_thread.start()

    return True


def get_sinatools_status_detail() -> Dict[str, Any]:
    path = _resolve_lemma_pickle()

    with _sinatools_lock:
        loaded = _sinatools_loaded
        loading = _sinatools_loading
        last_error = _sinatools_last_error
        runtime_ms = _sinatools_runtime_ms
        progress = _sinatools_progress_label
        resource_path = _sinatools_resource_path
        dictionary_size = len(_dictionary) if _sinatools_loaded else 0

    if loaded:
        status = "loaded"
        reason = f"SinaTools loaded from {resource_path}"
    elif loading:
        status = "loading"
        reason = "SinaTools model is loading in background."
    elif path:
        status = "lazy_not_loaded"
        reason = f"SinaTools model is present at {path}, but not loaded yet."
    else:
        status = "missing_resources"
        reason = f"Missing SinaTools lemma pickle. Put it under {RESOURCE_ROOT}"

    return {
        "status": status,
        "reason": reason,
        "model_present": path is not None,
        "model_path": str(path) if path else None,
        "loaded": loaded,
        "loading": loading,
        "last_error": last_error,
        "runtime_ms": runtime_ms,
        "progress_label": progress,
        "dictionary_size": dictionary_size,
    }


def _status_payload(text: str, started: float) -> Dict[str, Any]:
    detail = get_sinatools_status_detail()

    if detail.get("loaded"):
        status = "loaded"
    elif detail.get("loading"):
        status = "loading"
    elif detail.get("model_present"):
        status = "lazy_not_loaded"
    else:
        status = "missing_resources"

    return {
        "tool": "sinatools",
        "status": status,
        "reason": detail.get("reason"),
        "tokens": [],
        "lemmas": [],
        "pos": [],
        "input": text,
        "word_count": 0,
        "runtime_ms": int((time.time() - started) * 1000),
        "meta": detail,
    }


def sinatools_analyze(text: str) -> Dict[str, Any]:
    started = time.time()

    # Lazy-load policy:
    # never auto-start loading here.
    # only POST /tools/sinatools/preload may start the background load.
    if not _sinatools_loaded:
        return _status_payload(text, started)

    tokens = []
    lemmas = []
    pos_tags = []

    for surface in _tokenize(text):
        solutions = _lookup(surface)
        analyses = [_solution_to_analysis(surface, s, matched=True) for s in solutions[:3]]

        if not analyses:
            analyses = [_solution_to_analysis(surface, None, matched=False)]

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
                "lemma_raw": best.get("lemma_raw"),
                "lemma_alternatives": best.get("lemma_alternatives"),
                "root_raw": best.get("root_raw"),
                "pos_raw": best.get("pos_raw"),
            },
            "analyses": analyses,
        }

        tokens.append(token_payload)

        if best.get("lemma") is not None:
            lemmas.append(best.get("lemma"))

        if best.get("pos") is not None:
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