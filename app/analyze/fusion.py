from __future__ import annotations

from typing import Any, Dict

from app.services.merger_service import fusion_system


def fusion_for_text(text: str, camel_res: Dict[str, Any], stanza_res: Dict[str, Any], farasa_res: Dict[str, Any]) -> Dict[str, Any]:
    return fusion_system(text, camel_res, stanza_res, farasa_res)

