from __future__ import annotations

import io
import json
import csv
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.startup import run_all_registered_tools, fusion_system, evaluate_tools
from app.utils.constants import GOLD_DATASET

router = APIRouter()


_inflight_eval: Dict[str, Any] = {}


@router.get("/evaluate")
def evaluate(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")

    key = f"evaluate::{text}"
    cached = _inflight_eval.get(key)
    if cached is not None:
        return cached

    all_tool_results = run_all_registered_tools(text)
    camel_res = all_tool_results.get("camel", {})
    stanza_res = all_tool_results.get("stanza", {})
    farasa_res = all_tool_results.get("farasa", {})
    qalsadi_res = all_tool_results.get("qalsadi", {})

    payload = {
        "input": text,
        "tools": all_tool_results,
        "evaluation": evaluate_tools(
            text,
            camel_res,
            stanza_res,
            farasa_res,
            qalsadi_res=qalsadi_res,
            all_tool_results=all_tool_results,
        ),
    }
    _inflight_eval[key] = payload
    return {
        "input": text,
        "tools": all_tool_results,
        "evaluation": payload["evaluation"],
    }




@router.get("/evaluate/dataset")
def evaluate_dataset():
    """Evaluate tools against gold dataset — 10 sentences"""
    results = []
    for item in GOLD_DATASET:
        all_tool_results = run_all_registered_tools(item["text"])
        camel_res = all_tool_results.get("camel", {})
        stanza_res = all_tool_results.get("stanza", {})
        farasa_res = all_tool_results.get("farasa", {})
        qalsadi_res = all_tool_results.get("qalsadi", {})

        eval_result = evaluate_tools(
            item["text"],
            camel_res,
            stanza_res,
            farasa_res,
            qalsadi_res=qalsadi_res,
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

    all_tool_results = run_all_registered_tools(text)
    camel_res = all_tool_results.get("camel", {})
    stanza_res = all_tool_results.get("stanza", {})
    farasa_res = all_tool_results.get("farasa", {})
    qalsadi_res = all_tool_results.get("qalsadi", {})
    fused = fusion_system(text, camel_res, stanza_res, farasa_res, qalsadi_res=qalsadi_res, all_tool_results=all_tool_results)
    evaln = evaluate_tools(text, camel_res, stanza_res, farasa_res, qalsadi_res=qalsadi_res, all_tool_results=all_tool_results)

    if format == "json":
        payload: Dict[str, Any] = {
            "input": text,
            "combined": all_tool_results,
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

