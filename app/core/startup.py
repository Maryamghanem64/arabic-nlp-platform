from __future__ import annotations

import concurrent.futures

from app.services.cache_service import cached_analyze, clear_cache
from app.services.eval_service import evaluate_tools
from app.services.fusion_service import fusion_system
from app.tools.camel_tool import camel_analyze
from app.tools.farasa_tool import farasa_analyze
from app.tools.qalsadi_tool import qalsadi_analyze
from app.tools.stanza_tool import stanza_analyze


def run_all_tools(text: str):
    """Run CAMeL, Farasa, Stanza, and Qalsadi in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_camel = executor.submit(cached_analyze, camel_analyze, text)
        f_farasa = executor.submit(cached_analyze, farasa_analyze, text)
        f_stanza = executor.submit(cached_analyze, stanza_analyze, text)
        f_qalsadi = executor.submit(cached_analyze, qalsadi_analyze, text)
        camel_res = f_camel.result()
        farasa_res = f_farasa.result()
        stanza_res = f_stanza.result()
        qalsadi_res = f_qalsadi.result()
    return camel_res, farasa_res, stanza_res, qalsadi_res


__all__ = [
    "camel_analyze",
    "farasa_analyze",
    "stanza_analyze",
    "qalsadi_analyze",
    "cached_analyze",
    "clear_cache",
    "run_all_tools",
    "fusion_system",
    "evaluate_tools",
]
