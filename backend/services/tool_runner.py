from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from backend.analyzers.base import Analyzer
from backend.config.settings import Settings, get_settings


class InMemoryCache:
    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


class ToolRunner:
    def __init__(self, timeout_per_tool_s: float = 20.0):
        self.timeout_per_tool_s = timeout_per_tool_s
        self.settings: Settings = get_settings()
        self.cache = InMemoryCache()

    def _cache_key(self, tool_name: str, text: str) -> str:
        return f"{tool_name}::{text}"

    async def _run_one(self, tool: Analyzer, text: str) -> Dict[str, Any]:
        cache_key = self._cache_key(tool.tool_name, text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        started = time.time()
        try:
            res = await asyncio.wait_for(asyncio.to_thread(tool.analyze, text), timeout=self.timeout_per_tool_s)
            if isinstance(res, dict):
                res.setdefault("elapsed", time.time() - started)
            self.cache.set(cache_key, res)
            return res
        except asyncio.TimeoutError:
            return {"tool": tool.tool_name, "status": "error", "error": "timeout", "tokens": []}
        except Exception as e:
            return {"tool": tool.tool_name, "status": "error", "error": str(e), "tokens": []}

    async def run_all(self, text: str, tools: list[Analyzer]) -> Dict[str, Dict[str, Any]]:
        tasks = [self._run_one(t, text) for t in tools]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {r.get("tool", t.tool_name): r for r, t in zip(results, tools)}

