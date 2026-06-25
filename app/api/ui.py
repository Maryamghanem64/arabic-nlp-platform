from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.startup import cached_analyze, camel_analyze, farasa_analyze, stanza_analyze, qalsadi_analyze
from app.tools.alkhalil_tool import alkhalil_analyze
from app.tools.udpipe_tool import udpipe_analyze

router = APIRouter()


def _tool_names_from_query(tools: str) -> List[str]:
    return [t.strip().lower() for t in tools.split(",") if t.strip()]


def _normalize_for_ui(tool_name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from backend.services.normalizer import normalize_tool_output

        return normalize_tool_output(tool_name, raw)
    except Exception:
        return {"tool": tool_name, "status": "error", "input": None, "word_count": 0, "tokens": []}


def _ui_token_from_aligned(*, base_token: Dict[str, Any], tool_token: Optional[Dict[str, Any]], tool_name: str) -> Dict[str, Any]:
    if tool_name == "farasa":
        seg = tool_token.get("segmentation") if tool_token else None
        return {"segments": seg if isinstance(seg, list) and seg else [base_token.get("surface")]}

    lemma = tool_token.get("lemma") if tool_token else None
    root = tool_token.get("root") if tool_token else None
    pos = tool_token.get("pos") if tool_token else None

    from backend.services.ui_contracts import placeholder, safe_pos, pos_badge

    pos_std = safe_pos(pos)

    return {"lemma": placeholder(lemma), "root": placeholder(root), "pos": pos_std, "badge": pos_badge(pos_std)}


def _agreement_for_row(*, aligned_row: Any) -> Dict[str, Any]:
    tools = aligned_row.tools

    camel = tools.get("camel")
    stanza = tools.get("stanza")
    qalsadi = tools.get("qalsadi")

    def eq(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        import re

        def stripd(x: Any) -> str:
            return re.sub(r"[\u064B-\u065F\u0670]", "", str(x)).strip()

        sa = stripd(a)
        sb = stripd(b)
        return sa != "" and sa == sb

    pos_ok = True
    lemma_ok = True
    root_ok = True

    pos_vals = [t.get("pos") for t in [camel, stanza, qalsadi] if t and t.get("pos")]
    if len(pos_vals) >= 2:
        pos_ok = all(v == pos_vals[0] for v in pos_vals)

    lemma_vals = [t.get("lemma") for t in [camel, stanza, qalsadi] if t and t.get("lemma")]
    if len(lemma_vals) >= 2:
        lemma_ok = all(eq(v, lemma_vals[0]) for v in lemma_vals)

    root_vals = [t.get("root") for t in [camel, stanza, qalsadi] if t and t.get("root")]
    if len(root_vals) >= 2:
        root_ok = all(eq(v, root_vals[0]) for v in root_vals)

    status, color = ("none", "red")
    if pos_ok and lemma_ok and root_ok:
        status, color = "full", "green"
    elif pos_ok or lemma_ok or root_ok:
        status, color = "partial", "yellow"

    return {
        "pos": pos_ok,
        "lemma": lemma_ok,
        "root": root_ok,
        "status": status,
        "status_color": color,
        "agreement_state": status,
    }


@router.get("/ui/analyze/{tool}")
def ui_analyze(tool: str, text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    tool_l = tool.strip().lower()
    if tool_l not in {"camel", "farasa", "stanza", "qalsadi", "alkhalil", "udpipe"}:
        raise HTTPException(404, "Tool not supported for ui/analyze")

    raw = cached_analyze(
        {
            "camel": camel_analyze,
            "farasa": farasa_analyze,
            "stanza": stanza_analyze,
            "qalsadi": qalsadi_analyze,
            "alkhalil": alkhalil_analyze,
            "udpipe": udpipe_analyze,
        }[tool_l],
        text,
    )

    normalized = _normalize_for_ui(tool_l, raw)
    tokens = normalized.get("tokens", []) or []

    from backend.services.ui_contracts import placeholder, safe_pos

    rows = []
    for t in tokens:
        if tool_l == "farasa":
            seg = t.get("segmentation")
            segs = seg if isinstance(seg, list) and seg else [t.get("surface")]
            rows.append({"word": t.get("surface"), "segments": segs})
        else:
            rows.append({"word": t.get("surface"), "lemma": placeholder(t.get("lemma")), "root": placeholder(t.get("root")), "pos": safe_pos(t.get("pos"))})

    return {"tool": tool_l, "rows": rows}


@router.get("/ui/compare")
def ui_compare(text: str, tools: str = Query("camel,stanza,qalsadi,farasa")):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    tool_list = _tool_names_from_query(tools)

    raw_results: Dict[str, Dict[str, Any]] = {}
    if "camel" in tool_list:
        raw_results["camel"] = cached_analyze(camel_analyze, text)
    if "farasa" in tool_list:
        raw_results["farasa"] = cached_analyze(farasa_analyze, text)
    if "stanza" in tool_list:
        raw_results["stanza"] = cached_analyze(stanza_analyze, text)
    if "qalsadi" in tool_list:
        raw_results["qalsadi"] = cached_analyze(qalsadi_analyze, text)
    if "alkhalil" in tool_list:
        raw_results["alkhalil"] = cached_analyze(alkhalil_analyze, text)
    if "udpipe" in tool_list:
        raw_results["udpipe"] = cached_analyze(udpipe_analyze, text)

    norm = {t: _normalize_for_ui(t, raw_results[t]) for t in raw_results.keys()}

    base_tool = "farasa" if "farasa" in norm else ("camel" if "camel" in norm else "stanza" if "stanza" in norm else "qalsadi")
    base_tokens = norm.get(base_tool, {}).get("tokens", []) or []
    tools_tokens = {t: norm[t].get("tokens", []) or [] for t in norm.keys()}

    from backend.services.alignment_engine import align_tools, compute_agreements

    aligned, _meta = align_tools(base_tokens=base_tokens, tools_tokens=tools_tokens)
    agreements = compute_agreements(aligned_tokens=aligned)

    rows = []
    for at in aligned:
        row = {"word": at.base.get("surface"), "camel": None, "stanza": None, "qalsadi": None, "alkhalil": None, "udpipe": None, "farasa": None, "agreement": None}

        if "camel" in norm:
            row["camel"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("camel"), tool_name="camel")
        if "stanza" in norm:
            row["stanza"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("stanza"), tool_name="stanza")
        if "qalsadi" in norm:
            row["qalsadi"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("qalsadi"), tool_name="qalsadi")
        if "alkhalil" in norm:
            row["alkhalil"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("alkhalil"), tool_name="alkhalil")
        if "udpipe" in norm:
            row["udpipe"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("udpipe"), tool_name="udpipe")
        if "farasa" in norm:
            row["farasa"] = _ui_token_from_aligned(base_token=at.base, tool_token=at.tools.get("farasa"), tool_name="farasa")

        row["agreement"] = _agreement_for_row(aligned_row=at)
        rows.append(row)

    return {"summary": {"pos_agreement": agreements.get("pos_agreement", 0), "lemma_agreement": agreements.get("lemma_agreement", 0), "root_agreement": agreements.get("root_agreement", 0), "token_count": agreements.get("token_count", len(rows))}, "rows": rows}


@router.get("/ui/fusion")
def ui_fusion(text: str):
    if not text or not text.strip():
        raise HTTPException(400, "Empty text")

    camel_raw = cached_analyze(camel_analyze, text)
    stanza_raw = cached_analyze(stanza_analyze, text)
    qalsadi_raw = cached_analyze(qalsadi_analyze, text)
    alkhalil_raw = cached_analyze(alkhalil_analyze, text)
    udpipe_raw = cached_analyze(udpipe_analyze, text)
    farasa_raw = cached_analyze(farasa_analyze, text)

    camel_n = _normalize_for_ui("camel", camel_raw)
    stanza_n = _normalize_for_ui("stanza", stanza_raw)
    qalsadi_n = _normalize_for_ui("qalsadi", qalsadi_raw)
    alkhalil_n = _normalize_for_ui("alkhalil", alkhalil_raw)
    udpipe_n = _normalize_for_ui("udpipe", udpipe_raw)
    farasa_n = _normalize_for_ui("farasa", farasa_raw)

    base_tokens = farasa_n.get("tokens", []) or []
    tools_tokens = {
        "camel": camel_n.get("tokens", []) or [],
        "stanza": stanza_n.get("tokens", []) or [],
        "qalsadi": qalsadi_n.get("tokens", []) or [],
        "alkhalil": alkhalil_n.get("tokens", []) or [],
        "udpipe": udpipe_n.get("tokens", []) or [],
        "farasa": farasa_n.get("tokens", []) or [],
    }

    from backend.services.alignment_engine import align_tools

    aligned, _ = align_tools(base_tokens=base_tokens, tools_tokens=tools_tokens)

    rows = []
    for at in aligned:
        camel = at.tools.get("camel")
        stanza = at.tools.get("stanza")
        qalsadi = at.tools.get("qalsadi")

        agreement = _agreement_for_row(aligned_row=at)

        alkhalil = at.tools.get("alkhalil")
        udpipe = at.tools.get("udpipe")

        def pick(feature: str):
            for t, src in [(camel, "camel"), (stanza, "stanza"), (qalsadi, "qalsadi"), (alkhalil, "alkhalil"), (udpipe, "udpipe")]:
                if t and t.get(feature):
                    return t.get(feature), src
            return None, "-"

        lemma, lemma_src = pick("lemma")
        root, root_src = pick("root")
        pos, pos_src = pick("pos")

        from backend.services.ui_contracts import placeholder, safe_pos

        confs = []
        for t in [camel, stanza, qalsadi, alkhalil, udpipe]:
            if t and isinstance(t.get("confidence"), dict):
                confs.append(float(t["confidence"].get("score") or 0.0))
        confidence = round(sum(confs) / len(confs), 2) if confs else 0.0

        rows.append(
            {
                "word": at.base.get("surface"),
                "lemma": placeholder(lemma),
                "root": placeholder(root),
                "pos": safe_pos(pos),
                "gloss": placeholder(camel.get("gloss") if camel else None, default="-"),
                "confidence": confidence,
                "source": {
                    "lemma": lemma_src if lemma_src else "camel",
                    "root": root_src if root_src else "camel",
                    "pos": "agreement" if agreement.get("status") in ("full", "partial") else (pos_src if pos_src else "camel"),
                },
            }
        )

    return {"rows": rows}

