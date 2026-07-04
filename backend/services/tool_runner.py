from __future__ import annotations

import asyncio
import time
import threading
from typing import Any, Dict, Optional, Tuple

from backend.analyzers.base import Analyzer
from backend.config.settings import Settings, get_settings
from backend.services.normalizer import normalize_tool_output


class InMemoryCache:
    """Thread-safe in-memory cache with optional TTL."""

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, *, ttl_s: Optional[float] = None) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            if ttl_s is not None and (time.time() - stored_at) > ttl_s:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)


class ToolRunner:
    def __init__(self, timeout_per_tool_s: float = 20.0, cache_ttl_s: Optional[float] = None):
        self.timeout_per_tool_s = timeout_per_tool_s
        self.cache_ttl_s = cache_ttl_s
        self.settings: Settings = get_settings()
        self.cache = InMemoryCache()

    def _cache_key(self, tool_name: str, text: str) -> str:
        return f"{tool_name}::{text}"

    async def _run_one(self, tool: Analyzer, text: str) -> Dict[str, Any]:
        tool_name = getattr(tool, "tool_name", None) or getattr(tool, "name", "unknown")
        cache_key = self._cache_key(tool_name, text)

        cached = self.cache.get(cache_key, ttl_s=self.cache_ttl_s)
        if cached is not None:
            return cached

        started = time.time()
        try:
            # IMPORTANT: do not use timeout as normal behavior.
            # Let tools run to completion.
            # Only use timeout as a hard safety guard (very rarely).
            # If a tool is warming/loading it should return real "loading" state itself.
            res = await asyncio.to_thread(tool.analyze, text)
            if not isinstance(res, dict):
                res = {"tool": tool_name, "status": "error", "error": "invalid tool result", "tokens": []}

            # Normalize BEFORE caching
            raw_result = res
            normalized = normalize_tool_output(tool_name, raw_result)
            normalized.setdefault("elapsed", time.time() - started)
            self.cache.set(cache_key, normalized)
            return normalized
        except Exception as e:
            # Hard failure (no fake timeout result here)
            normalized = {
                "tool": tool_name,
                "status": "error",
                "error": str(e),
                "input": text,
                "word_count": 0,
                "tokens": [],
            }
            return normalized

    async def run_all(self, text: str, tools: list[Analyzer]) -> Dict[str, Dict[str, Any]]:
        tasks = [self._run_one(t, text) for t in tools]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {r.get("tool", getattr(t, "tool_name", "unknown")): r for r, t in zip(results, tools)}


