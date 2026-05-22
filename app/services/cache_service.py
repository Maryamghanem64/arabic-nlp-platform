from __future__ import annotations

import threading
from typing import Any, Callable, Dict

from app.utils.logger import logger


class InMemoryCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_CACHE = InMemoryCache()


def cached_analyze(func: Callable[[str], Dict[str, Any]], text: str) -> Dict[str, Any]:
    key = f"{func.__name__}::{text}"
    cached = _CACHE.get(key)
    if cached is not None:
        logger.info(f"[CACHE] HIT - {func.__name__}")
        return cached

    result = func(text)
    _CACHE.set(key, result)
    return result


def clear_cache() -> None:
    _CACHE.clear()

