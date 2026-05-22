from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolStatus(BaseModel):
    status: str
    note: Optional[str] = None


class ToolStatuses(BaseModel):
    camel: ToolStatus
    farasa: ToolStatus
    stanza: ToolStatus
    qalsadi: ToolStatus
    sinatools: ToolStatus
    udpipe: ToolStatus
    alkhalil: ToolStatus
    arabert: ToolStatus


class StatusResponse(BaseModel):
    platform: str
    version: str
    tools: Dict[str, Any] = Field(default_factory=dict)


class EvaluateDatasetResponse(BaseModel):
    total_sentences: int
    average_f1: float
    average_f1_pct: str
    results: List[Dict[str, Any]]

