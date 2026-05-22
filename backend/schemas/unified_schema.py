from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class DependencyTD(TypedDict):
    head: Optional[int]
    head_text: Optional[str]
    deprel: Optional[str]


class FeaturesTD(TypedDict):
    gender: Optional[str]
    number: Optional[str]
    tense: Optional[str]
    case: Optional[str]
    definite: Optional[str]
    voice: Optional[str]
    person: Optional[str]


class ConfidenceTD(TypedDict):
    score: float
    level: Literal["low", "medium", "high"]


class TokenMetaTD(TypedDict, total=False):
    source_tool: Optional[str]
    root_type: Optional[str]
    corrections: List[str]
    notes: List[str]


class UnifiedTokenTD(TypedDict):
    surface: Optional[str]
    lemma: Optional[str]
    root: Optional[str]
    pos: Optional[str]
    gloss: Optional[str]
    segmentation: Optional[List[str]]
    features: FeaturesTD
    dependency: DependencyTD
    confidence: ConfidenceTD
    meta: TokenMetaTD


class UnifiedToolErrorTD(TypedDict, total=False):
    tool: str
    status: Literal["error"]
    error: str
    tokens: List[UnifiedTokenTD]


class UnifiedToolOkTD(TypedDict, total=False):
    tool: str
    status: Literal["ok", "failed"]
    input: Optional[str]
    word_count: int
    tokens: List[UnifiedTokenTD]


UnifiedToolResponseTD = Dict[str, Any]

