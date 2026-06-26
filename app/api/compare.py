from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core.startup import run_all_registered_tools
from app.utils.helpers import normalize_lemma_for_compare, normalize_pos_for_compare
from backend.services.alignment_engine import align_tools
from backend.services.comparison_service import build_conflicts
from backend.services.normalizer import normalize_tool_output
from backend.schemas.unified_schema import AnalysisEnvelope

router = APIRouter()

VALID_COMPARE_TOOLS = {"camel", "farasa", "stanza", "qalsadi", "alkhalil", "udpipe"}


def _dump_envelope(payload: AnalysisEnvelope) -> dict:
    return payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()


def _parse_tools(tools: str) -> list[str]:
    return [t.strip().lower() for t in tools.split(",") if t.strip()]


def _token_value(token: dict | None, key: str) -> str | None:
    if not token:
        return None
    value = token.get(key)
    if value is not None and str(value).strip():
        return str(value).strip()
    analyses = token.get("analyses") or []
    if analyses and isinstance(analyses[0], dict):
        nested = analyses[0].get(key)
        if nested is not None and str(nested).strip():
            return str(nested).strip()
    return None


def _group_feature(row_tools: dict, feature: str) -> dict:
    values_by_tool = {
        tool: _token_value(token, "upos" if feature == "pos" and not _token_value(token, "pos") else feature)
        for tool, token in row_tools.items()
        if token
    }
    normalizer = normalize_pos_for_compare if feature == "pos" else normalize_lemma_for_compare if feature == "lemma" else lambda v: str(v).strip()
    groups: dict[str, dict] = {}
    for tool, value in values_by_tool.items():
        if not value:
            continue
        normalized = normalizer(value)
        if not normalized:
            continue
        groups.setdefault(normalized, {"value": value, "tools": []})["tools"].append(tool)

    status = "missing"
    if len(groups) == 1 and groups:
        status = "agreement"
    elif len(groups) > 1:
        status = "conflict"
    elif values_by_tool:
        status = "partial"

    return {"status": status, "groups": list(groups.values()), "values_by_tool": values_by_tool}


def _educational_notes(row_tools: dict, conflicts: list[dict]) -> list[str]:
    notes: list[str] = []
    lemma = _group_feature(row_tools, "lemma")
    pos = _group_feature(row_tools, "pos")

    if lemma["status"] == "conflict":
        notes.append("Lemma disagreement often reflects different normalization choices: some analyzers return a dictionary lemma, while others preserve gender, number, or diacritics from the surface token.")
    if pos["status"] == "conflict":
        notes.append("POS disagreement usually comes from different tagsets and context models: morphology-first tools may prefer lexical category, while UD parsers prefer syntactic function in context.")
    if any(conflict.get("feature") == "pos" for conflict in conflicts):
        notes.append("Treat POS conflicts as high-value evidence: they change downstream syntax, agreement metrics, and fusion confidence.")
    if not notes and row_tools.get("farasa") and _token_value(row_tools.get("farasa"), "segmentation"):
        notes.append("Segmentation evidence is mainly clitic evidence; it should be compared with lemma/POS before trusting a fused analysis.")
    return notes


@router.get("/compare")
def compare(text: str, tools: str = Query("camel,farasa,stanza,qalsadi,alkhalil,udpipe")):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    requested = [tool for tool in _parse_tools(tools) if tool in VALID_COMPARE_TOOLS]
    if not requested:
        raise HTTPException(400, "No supported tools requested")

    all_results = run_all_registered_tools(text)
    normalized = {name: normalize_tool_output(name, payload) for name, payload in all_results.items() if name in VALID_COMPARE_TOOLS}

    present_tools = [tool for tool in requested if tool in normalized and (normalized[tool].get("tokens") or [])]
    base_tool = "farasa" if "farasa" in present_tools else next((tool for tool in requested if tool in present_tools), None)
    if base_tool is None:
        base_tool = "farasa" if "farasa" in normalized else next((tool for tool in requested if tool in normalized), None)
    if base_tool is None:
        raise HTTPException(503, "No compare-capable tool returned tokens")

    base_tokens = normalized.get(base_tool, {}).get("tokens", []) or []
    tools_tokens = {tool: normalized.get(tool, {}).get("tokens", []) or [] for tool in requested if tool in normalized}

    aligned, _meta = align_tools(base_tokens=base_tokens, tools_tokens=tools_tokens)

    comparison = []
    for index, row in enumerate(aligned):
        row_conflicts = build_conflicts(
            camel_tok=row.tools.get("camel"),
            stanza_tok=row.tools.get("stanza"),
            qalsadi_tok=row.tools.get("qalsadi"),
            alkhalil_tok=row.tools.get("alkhalil"),
            udpipe_tok=row.tools.get("udpipe"),
        )
        row_tools = {tool: row.tools.get(tool) or {} for tool in tools_tokens.keys()}
        comparison.append(
            {
                "index": index,
                "word": row.base.get("surface") or f"#{index + 1}",
                "tools": row_tools,
                "conflicts": row_conflicts,
                "agreement": {
                    "lemma": _group_feature(row_tools, "lemma"),
                    "root": _group_feature(row_tools, "root"),
                    "pos": _group_feature(row_tools, "pos"),
                },
                "educational_notes": _educational_notes(row_tools, row_conflicts),
            }
        )

    envelope = AnalysisEnvelope(
        input=text,
        tools={tool: normalized.get(tool, {}) for tool in requested},
        comparison=comparison,
        active_tools=present_tools,
        meta={
            "active_tools": present_tools,
            "degraded_tools": [tool for tool, payload in all_results.items() if isinstance(payload, dict) and payload.get("status") not in {"ok"}],
            "base_tool": base_tool,
            "requested_tools": requested,
        },
    )
    return _dump_envelope(envelope)
