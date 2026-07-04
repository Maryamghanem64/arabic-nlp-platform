from __future__ import annotations

import json
import time

from backend.analyzers.sinatools_tool import get_sinatools_status_detail, load_sinatools, sinatools_analyze

TEXT = "وجدت المعلمة طالبة مجتهدة في الفصل"

started = time.perf_counter()
ok = load_sinatools()
elapsed_ms = int((time.perf_counter() - started) * 1000)
status = get_sinatools_status_detail()
result = sinatools_analyze(TEXT) if ok else None

print(json.dumps({
    "loaded": ok,
    "elapsed_ms": elapsed_ms,
    "status": status,
    "sample_status": result.get("status") if isinstance(result, dict) else None,
    "sample_word_count": result.get("word_count") if isinstance(result, dict) else None,
    "sample_tokens": result.get("tokens", [])[:3] if isinstance(result, dict) else [],
}, ensure_ascii=False, indent=2))
