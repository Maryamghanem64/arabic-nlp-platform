from __future__ import annotations

import io
import json
import csv
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.startup import run_all_tools, fusion_system, evaluate_tools
from app.utils.constants import GOLD_DATASET

router = APIRouter()


_inflight_eval: Dict[str, Any] = {}


@router.get("/evaluate")
def evaluate(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")

    # Bug 1: ensure per-request execution doesn't duplicate tool calls.
    # Key includes endpoint + text; evaluation logic/metrics remain unchanged.
    key = f"evaluate::{text}"
    cached = _inflight_eval.get(key)
    if cached is not None:
        camel_res, farasa_res, stanza_res = cached
        from app.core.tool_registry import detect_tool_status
        statuses = detect_tool_status()
        all_tool_results = {
            "camel": camel_res,
            "farasa": farasa_res,
            "stanza": stanza_res,
        }
        for tool_name in statuses.keys():
            if tool_name not in all_tool_results:
                all_tool_results[tool_name] = {"status": statuses[tool_name].get("status"), "reason": statuses[tool_name].get("reason")}

        return {
            "input": text,
            "evaluation": evaluate_tools(text, camel_res, stanza_res, farasa_res, qalsadi_res=all_tool_results.get("qalsadi"), all_tool_results=all_tool_results),
        }


    # Core tools used by fusion/evaluation.
    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)

    # Tool participation for metrics/excluded_tools must include qalsadi.
    all_tool_results = {
        "camel": camel_res,
        "farasa": farasa_res,
        "stanza": stanza_res,
        "qalsadi": qalsadi_res,
    }

    # We don't run additional analysis; statuses come from tool_registry detect.
    # evaluate_tools expects all_tool_results to contain each tool result.
    # For inactive tools we construct lightweight dicts with the computed status.
    from app.core.tool_registry import detect_tool_status

    statuses = detect_tool_status()
    for tool_name in statuses.keys():
        if tool_name not in all_tool_results:
            all_tool_results[tool_name] = {"status": statuses[tool_name].get("status"), "reason": statuses[tool_name].get("reason")}

    _inflight_eval[key] = (camel_res, farasa_res, stanza_res)
    return {
        "input": text,
        "evaluation": evaluate_tools(
            text,
            camel_res,
            stanza_res,
            farasa_res,
            qalsadi_res=all_tool_results.get("qalsadi"),
            all_tool_results=all_tool_results,
        ),
    }




@router.get("/evaluate/dataset")
def evaluate_dataset():
    """Evaluate tools against gold dataset — 10 sentences"""
    results = []
    for item in GOLD_DATASET:
        camel_res, farasa_res, stanza_res, _ = run_all_tools(item["text"])
        from app.core.tool_registry import detect_tool_status
        statuses = detect_tool_status()
        all_tool_results = {
            "camel": camel_res,
            "farasa": farasa_res,
            "stanza": stanza_res,
        }
        for tool_name in statuses.keys():
            if tool_name not in all_tool_results:
                all_tool_results[tool_name] = {"status": statuses[tool_name].get("status"), "reason": statuses[tool_name].get("reason")}

        eval_result = evaluate_tools(
            item["text"],
            camel_res,
            stanza_res,
            farasa_res,
            qalsadi_res=all_tool_results.get("qalsadi"),
            all_tool_results=all_tool_results,
        )

        results.append(
            {
                "text": item["text"],
                "pos_agreement_pct": eval_result["pos_agreement_pct"],
                "lemma_match_pct": eval_result["lemma_match_pct"],
                "pos_f1": eval_result["pos_f1"],
                "total_words": eval_result["total_words"],
            }
        )

    avg_f1 = sum(r["pos_f1"] for r in results) / len(results)
    return {
        "total_sentences": len(results),
        "average_f1": round(avg_f1, 3),
        "average_f1_pct": f"{round(avg_f1 * 100, 1)}%",
        "results": results,
    }


@router.get("/export")
def export_results(text: str, format: str = Query("json", description="json or csv")):
    """Export full analysis as downloadable JSON or CSV."""
    if not text.strip():
        raise HTTPException(400, "Empty text")

    camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools(text)
    fused = fusion_system(text, camel_res, stanza_res, farasa_res, qalsadi_res)
    all_tool_results = {
        "camel": camel_res,
        "farasa": farasa_res,
        "stanza": stanza_res,
        "qalsadi": qalsadi_res,
    }
    from app.core.tool_registry import detect_tool_status

    statuses = detect_tool_status()
    for tool_name in statuses.keys():
        if tool_name not in all_tool_results:
            all_tool_results[tool_name] = {"status": statuses[tool_name].get("status"), "reason": statuses[tool_name].get("reason")}

    evaln = evaluate_tools(text, camel_res, stanza_res, farasa_res, qalsadi_res=qalsadi_res, all_tool_results=all_tool_results)

    if format == "json":
        payload: Dict[str, Any] = {
            "input": text,
            "combined": {
                "camel": camel_res,
                "farasa": farasa_res,
                "stanza": stanza_res,
                "qalsadi": qalsadi_res,
            },
            "fusion": fused,
            "evaluation": evaln,
        }
        return StreamingResponse(
            io.StringIO(json.dumps(payload, ensure_ascii=False, indent=2)),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=analysis.json"},
        )

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "word",
                "lemma",
                "root",
                "root_type",
                "pos",
                "pos_source",
                "gender",
                "number",
                "tense",
                "gloss",
                "case",
                "definite",
                "confidence_score",
                "confidence_level",
                "notes",
            ]
        )
        for tok in fused.get("fusion", []):
            f = tok.get("final", {})
            writer.writerow(
                [
                    tok.get("word", ""),
                    f.get("lemma", ""),
                    f.get("root", ""),
                    f.get("root_type", ""),
                    f.get("pos", ""),
                    tok.get("sources", {}).get("pos", ""),
                    f.get("gender", ""),
                    f.get("number", ""),
                    f.get("tense", ""),
                    f.get("gloss", ""),
                    f.get("case", ""),
                    f.get("definite", ""),
                    f.get("confidence_score", ""),
                    f.get("confidence_level", ""),
                    "; ".join(tok.get("notes", [])),
                ]
            )

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=analysis.csv"},
        )

    raise HTTPException(400, "format must be 'json' or 'csv'")


@router.post("/cache/clear")
def cache_clear():
    from app.core.startup import clear_cache

    clear_cache()
    return {"status": "cache cleared"}

