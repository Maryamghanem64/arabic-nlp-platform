"""App-level AraBERT tool facade.

Polish (Fix-3): ensure that when AraBERT is used as a contextual support
model, morphology fields are clearly marked as unsupported/contextual.

This file only re-exports backend analyzer functions; we do not modify
algorithms, and we do not fabricate lemma/root/POS.
"""

from typing import Any, Dict

from backend.analyzers import arabert_tool as _backend_arabert_tool


def _postprocess_arabert_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate each token with explicit contextual/unsupported metadata."""
    try:
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            return payload

        unsupported_note = (
            "AraBERT base model does not provide lemma/root/POS without a fine-tuned head."
        )

        for tok in tokens:
            if not isinstance(tok, dict):
                continue
            # Ensure morphology fields remain null.
            for k in ("lemma", "root", "pos"):
                tok[k] = None

            meta = tok.get("meta")
            if isinstance(meta, dict):
                meta["role"] = "contextual support / disambiguation"
                meta["morphology_supported"] = False
                meta["supported_features"] = ["contextual"]
                meta["unsupported_features"] = [
                    "lemma",
                    "root",
                    "pos",
                    "segmentation",
                    "dependency",
                ]
                meta["display_note"] = unsupported_note

            caps = tok.get("capabilities")
            if isinstance(caps, dict):
                caps.update(
                    {
                        "contextual": True,
                        "lemma": False,
                        "root": False,
                        "pos": False,
                        "segmentation": False,
                        "dependency": False,
                    }
                )

        return payload
    except Exception:
        return payload


def arabert_analyze(text: str) -> Dict[str, Any]:  # type: ignore[override]
    payload = _backend_arabert_tool.arabert_analyze(text)

    if isinstance(payload, dict):
        payload = _postprocess_arabert_payload(payload)
    return payload


# keep original imports accessible
from backend.analyzers import arabert_tool as _backend_arabert_tool

backend_analyze = _backend_arabert_tool.arabert_analyze


def get_arabert_status() -> str:  # type: ignore[override]
    return _backend_arabert_tool.get_arabert_status()


def get_arabert_status_detail() -> Dict[str, Any]:  # type: ignore[override]
    return _backend_arabert_tool.get_arabert_status_detail()


def load_arabert() -> bool:  # type: ignore[override]
    return _backend_arabert_tool.load_arabert()


__all__ = [
    "arabert_analyze",
    "get_arabert_status",
    "get_arabert_status_detail",
    "load_arabert",
]


