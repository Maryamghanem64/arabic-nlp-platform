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
    global udpipe_model, udpipe_pipeline_obj
    if udpipe_pipeline_obj is not None:
        return

    model_path = _resolve_model_path()
    if model_path is None:
        logger.warning("⚠️ UDPipe model not found")
        return

    try:
        from ufal.udpipe import Model, Pipeline

        udpipe_model = Model.load(str(model_path))
        udpipe_pipeline_obj = Pipeline(
            udpipe_model,
            "tokenize",
            Pipeline.DEFAULT,
            Pipeline.DEFAULT,
            "conllu",
        )
        logger.info(f"✅ UDPipe loaded: {model_path}")
    except Exception as e:
        udpipe_model = None
        udpipe_pipeline_obj = None
        logger.warning(f"⚠️ UDPipe load failed: {e}")


def _parse_conllu(conllu_text: str) -> List[Dict[str, Any]]:
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

        upos = parts[3] if parts[3] != "_" else None
        lemma = parts[2] if parts[2] != "_" else None
        feats = parts[5] if parts[5] != "_" else None

        root = None
        # Best-effort root extraction from feats (not guaranteed).
        if feats and isinstance(feats, str):
            # try common fields
            for key in ["Root", "root", "LEMMA", "OrigLemma"]:
                if f"{key}=" in feats:
                    for kv in feats.split("|"):
                        if kv.startswith(key + "="):
                            root = kv.split("=", 1)[1] or None

        surface = parts[1]
        tokens.append(
            {
                "surface": surface,
                "lemma": lemma,
                "pos": upos,
                "root": root,
                "gloss": None,
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

        with udpipe_lock:
            conllu = udpipe_pipeline_obj.process(sentence)
            conllu_text = conllu if isinstance(conllu, str) else str(conllu)

        tokens = _parse_conllu(conllu_text)
        return {
            "tool": tool,
            "status": "ok" if tokens is not None else "error",
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

