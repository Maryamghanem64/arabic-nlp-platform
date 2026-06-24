from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core.startup import run_all_tools

router = APIRouter()


def _parse_tools(tools: str) -> list[str]:
    return [t.strip().lower() for t in tools.split(",") if t.strip()]


@router.get("/compare")
def compare(text: str, tools: str = Query("camel,farasa,stanza,qalsadi")):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    tool_list = set(_parse_tools(tools))
    if not tool_list:
        raise HTTPException(400, "No tools requested")

    # Run core tools (do not change tool execution logic—reuse existing run_all_tools)
    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)

    all_results = {
        "camel": camel_res,
        "farasa": farasa_res,
        "stanza": stanza_res,
        "qalsadi": qalsadi_res,
    }

    # UI CompareView expects { comparison: [ {word, tools:{toolKey: {available, entries}} } ] }
    # The UI normalizes shapes itself; we return a minimal backend contract compatible with its normalization.
    # We return per-tool token arrays under each tool key.

    def extract_tokens(payload):
        if isinstance(payload, dict):
            if isinstance(payload.get("tokens"), list):
                return payload.get("tokens")
            if isinstance(payload.get("analyses"), list) and payload["analyses"]:
                # fallback if wrapped
                return payload.get("analyses", [])
        return []

    tokens_by_tool = {k: extract_tokens(all_results.get(k, {})) for k in tool_list if k in all_results}

    # Simple alignment-free fallback: index by tool token positions.
    # The frontend only requires that tools[toolKey] exists; its normalization handles missing cells.
    max_len = max((len(v) for v in tokens_by_tool.values()), default=0)
    comparison = []
    for i in range(max_len):
        row = {"index": i, "word": None, "tools": {}}
        for tool_key in tokens_by_tool.keys():
            tok = tokens_by_tool[tool_key][i] if i < len(tokens_by_tool[tool_key]) else None
            row["tools"][tool_key] = tok if tok is not None else {}
        # word fallback
        for tool_key in tokens_by_tool.keys():
            tok = tokens_by_tool[tool_key][i] if i < len(tokens_by_tool[tool_key]) else None
            if tok and (tok.get("surface") or tok.get("word")):
                row["word"] = tok.get("surface") or tok.get("word")
                break
        if not row["word"]:
            row["word"] = f"#{i+1}"
        comparison.append(row)

    return {"input": text, "comparison": comparison, "active_tools": sorted(tool_list)}

