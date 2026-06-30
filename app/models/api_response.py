from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from backend.schemas.unified_schema import AnalysisEnvelope


STANDARD_RESPONSE_KEYS = {"status", "message", "data", "metadata", "errors"}


def dump_envelope(payload: AnalysisEnvelope) -> Dict[str, Any]:
    return payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()


def success_response(
    data: Optional[Mapping[str, Any]] = None,
    *,
    message: str = "OK",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return the platform-wide API response shape while preserving legacy top-level
    fields for the existing frontend screens.
    """
    payload: Dict[str, Any] = dict(data or {})
    response: Dict[str, Any] = {
        "status": "success",
        "message": message,
        "data": payload,
        "metadata": dict(metadata or payload.get("meta") or {}),
        "errors": [],
    }

    for key, value in payload.items():
        if key not in STANDARD_RESPONSE_KEYS:
            response[key] = value

    return response
