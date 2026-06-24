from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logger import logger
from backend.config.tool_paths import UDPipePaths


udpipe_model = None
udpipe_pipeline_obj = None
udpipe_lock = threading.Lock()

_udp_paths = UDPipePaths()


def _resolve_model_path() -> Optional[Path]:
    # Centralized: UDPIP_MODEL env > repository default + legacy candidates.
    return _udp_paths.resolved_existing()


def load_udpipe() -> None:
    """Load UDPipe model + build a robust pipeline."""

    global udpipe_model, udpipe_pipeline_obj
    if udpipe_pipeline_obj is not None:
        return

    model_path = _resolve_model_path()
    if model_path is None:
        logger.warning(
            "⚠️ UDPipe model not found (UDPipePaths.resolved_existing() returned None)"
        )
        return

    try:
        import ufal.udpipe  # noqa: F401
        from ufal.udpipe import Model, Pipeline

        udpipe_model = Model.load(str(model_path))

        # Safer pipeline construction across UDPipe bindings:
        # tokenize + tag + parse.
        udpipe_pipeline_obj = Pipeline(udpipe_model, "tokenize", "tag", "parse")

        logger.info(f"✅ UDPipe loaded: {model_path}")
    except Exception as e:
        udpipe_model = None
        udpipe_pipeline_obj = None
        logger.warning(f"⚠️ UDPipe load failed: {e}")


def _parse_conllu(conllu_text: str) -> List[Dict[str, Any]]:
    """Parse UDPipe-produced CoNLL-U text into token dicts.

    Kept for compatibility if the UDPipe binding returns CoNLL-U text.
    """

    tokens: List[Dict[str, Any]] = []
    for line in (conllu_text or "").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        if "-" in parts[0] or "." in parts[0]:
            continue

        surface = parts[1] if parts[1] != "_" else None
        lemma = parts[2] if parts[2] != "_" else None
        upos = parts[3] if parts[3] != "_" else None

        if not surface:
            continue

        tokens.append(
            {
                "surface": surface,
                "lemma": lemma,
                "pos": upos,
            }
        )

    return tokens


def udpipe_analyze(text: str) -> Dict[str, Any]:
    tool = "udpipe"

    try:
        global udpipe_pipeline_obj

        model_path = _resolve_model_path()
        if os.environ.get("UDPIP_MODEL") is None and model_path is not None:
            # Make runtime consistent with diagnostics.
            os.environ["UDPIP_MODEL"] = str(model_path)

        if udpipe_pipeline_obj is None:
            load_udpipe()

        if udpipe_pipeline_obj is None:
            return {
                "tool": tool,
                "status": "unavailable",
                "reason": "UDPipe model/pipeline not available",
                "input": text,
                "word_count": 0,
                "tokens": [],
            }

        sentence = text or ""

        tokens: List[Dict[str, Any]] = []

        with udpipe_lock:
            processed = udpipe_pipeline_obj.process(sentence)

        # Preferred path: iterate token objects (binding-specific)
        try:
            # processed may be an iterable of Sentence objects
            for s in processed:
                for i in range(s.length()):
                    t = s[i]
                    tok = getattr(t, "form", None)
                    lemma = getattr(t, "lemma", None)
                    pos = getattr(t, "upos", None)
                    if tok:
                        tokens.append(
                            {"surface": str(tok), "lemma": None if lemma is None else str(lemma), "pos": None if pos is None else str(pos)}
                        )
        except Exception:
            # Fallback: treat as CoNLL-U text
            conllu_text = processed if isinstance(processed, str) else str(processed)
            tokens = _parse_conllu(conllu_text)

        return {
            "tool": tool,
            "status": "ok" if tokens else "error",
            "reason": "" if tokens else "UDPipe output produced no tokens",
            "input": text,
            "word_count": len(tokens),
            "tokens": tokens,
        }

    except Exception as e:
        return {
            "tool": tool,
            "status": "error",
            "reason": str(e),
            "input": text,
            "word_count": 0,
            "tokens": [],
        }

