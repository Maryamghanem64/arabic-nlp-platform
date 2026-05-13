from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MatchStatus(str, Enum):
    FULL_MATCH = "FULL_MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    CONFLICT = "CONFLICT"


class ToolTokenAnalysis(BaseModel):
    surface: str

    # optional fields depending on tool
    lemma: Optional[str] = None
    pos: Optional[str] = None
    segmentation: Optional[List[str]] = None

    # additional tool-specific fields
    raw: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: Literal["camel", "stanza", "qalsadi", "farasa"]
    status: Literal["ok", "error", "failed"]
    error: Optional[str] = None

    tokens: List[ToolTokenAnalysis] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class TokenComparison(BaseModel):
    pos_status: MatchStatus = MatchStatus.CONFLICT
    lemma_status: MatchStatus = MatchStatus.CONFLICT
    segmentation_status: MatchStatus = MatchStatus.CONFLICT

    details: Dict[str, Any] = Field(default_factory=dict)


class FusionResult(BaseModel):
    final_pos: Optional[str] = None
    final_lemma: Optional[str] = None
    final_segmentation: Optional[List[str]] = None

    confidence: float = 0.0
    confidence_level: Literal["high", "medium", "low"] = "low"

    chosen_sources: Dict[str, str] = Field(default_factory=dict)


class TokenFusion(BaseModel):
    token: str
    tools: Dict[str, ToolTokenAnalysis] = Field(default_factory=dict)
    comparison: TokenComparison = Field(default_factory=TokenComparison)
    fusion: FusionResult = Field(default_factory=FusionResult)
    flags: List[str] = Field(default_factory=list)


class FusionResponse(BaseModel):
    text: str
    fusion_result: List[TokenFusion]
    meta: Dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    total_words: int
    pos_agreement_pct: str
    lemma_match_pct: str
    segmentation_coverage: float

