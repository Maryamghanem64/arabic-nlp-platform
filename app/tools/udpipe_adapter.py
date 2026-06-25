from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import ufal

from app.utils.logger import logger
from app.utils.helpers import strip_diacritics


from backend.config.tool_paths import UDPipePaths


_udp_paths = UDPipePaths()


def _resolve_model_path() -> Optional[Path]:
    # Centralized: UDPIP_MODEL env > repository default + legacy candidates.
    return _udp_paths.resolved_existing()



def _udpipe_available() -> bool:
    try:
        import ufal.udpipe as _  # noqa: F401

        return True
    except Exception:
        return False


_model = None
_model_path_cached: Optional[Path] = None


_udpipe_lock = None


def _ensure_lock():
    global _udpipe_lock
    if _udpipe_lock is None:
        import threading

        _udpipe_lock = threading.Lock()


def _get_model():
    global _model, _model_path_cached

    _ensure_lock()
    with _udpipe_lock:
        if _model is not None and _model_path_cached is not None:
            # If UDPIP_MODEL env points elsewhere, reload.
            current = _resolve_model_path()
            if current is not None and _model_path_cached.resolve() == current.resolve():
                return _model

        if not _udpipe_available():
            return None


    model_path = _resolve_model_path()
    if model_path is None or not model_path.exists() or not model_path.is_file():
        return None

    try:
        import ufal.udpipe

        _model = ufal.udpipe.Model(str(model_path))
        _model_path_cached = model_path
        logger.info("UDPipe model loaded: %s", model_path)
        return _model
    except Exception as exc:
        logger.warning("UDPipe model load failed: %s", exc)
        _model = None
        _model_path_cached = None
        return None


def _fix_udpipe_lemma(surface: Optional[str], lemma: Optional[str]) -> Optional[str]:
    if not lemma or not surface:
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


def udpipe_analyze(text: str) -> Dict[str, Any]:
    from app.core.tool_registry import unified_result

    tool = "udpipe"

    try:
        # Make runtime consistent with diagnostics / centralized resolver.
        if os.environ.get("UDPIP_MODEL") is None:
            default = _resolve_model_path()
            if default is not None:
                os.environ["UDPIP_MODEL"] = str(default)


        if not _udpipe_available():
            return unified_result(tool=tool, status="unavailable", tokens=[], lemmas=[], pos=[], reason="ufal.udpipe is not installed.")


        model = _get_model()
        if model is None:
            model_path = _resolve_model_path()
            return unified_result(
                tool=tool,
                status="unavailable",
                tokens=[],
                lemmas=[],
                pos=[],
                reason=f"UDPipe model missing. Set UDPIP_MODEL to a valid .udpipe model path. Current: {model_path}",
            )


        # Processing
        pipeline_cls = ufal.udpipe.Pipeline
        proc = pipeline_cls(model, "tokenize", pipeline_cls.DEFAULT, pipeline_cls.DEFAULT, "conllu")
        # Some UDPipe versions use `tagger` differently; pipeline should be robust.
        sentence = text if text is not None else ""

        sentences = proc.process(sentence)

        tokens: List[Dict[str, Any]] = []
        lemmas: List[str] = []
        pos: List[str] = []
        deps: List[Dict[str, Any]] = []

        for s in sentences:
            # s is Sentence object
            for i in range(s.length()):
                t = s[i]
                tok = t.form
                lemma = _fix_udpipe_lemma(tok, t.lemma)
                upos = t.upos
                # Dependency fields vary by UDPipe build; use getattr defensively.
                # In udpipe, t.head and t.deprel exist.
                head = getattr(t, "head", None)
                deprel = getattr(t, "deprel", None)

                tokens.append({"surface": tok, "lemma": lemma})
                if lemma is not None:
                    lemmas.append(str(lemma))
                if upos is not None:
                    pos.append(str(upos))
                deps.append({"head": int(head) if head is not None and str(head).isdigit() else head, "deprel": deprel})

        # deps are not part of the unified schema; keep only unified keys.
        return unified_result(tool=tool, status="ok", tokens=tokens, lemmas=lemmas, pos=pos, reason="")


    except NameError:
        return unified_result(tool=tool, status="unavailable", tokens=[], lemmas=[], pos=[], reason="UDPipe runtime unavailable.")
    except Exception as exc:
        logger.warning("[UDPIPE] error: %s", exc)
        return unified_result(tool=tool, status="error", tokens=[], lemmas=[], pos=[], reason=str(exc))


