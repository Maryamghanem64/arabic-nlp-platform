from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.services.merger_service import evaluate_tools, run_all_tools
from app.utils.constants import GOLD_DATASET
from app.utils.helpers import normalize_pos_for_compare


def run_all_tools_for_dataset(text: str):
    return run_all_tools(text)


def evaluate_dataset() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    for item in GOLD_DATASET:
        camel_res, farasa_res, stanza_res, qalsadi_res = run_all_tools_for_dataset(item["text"])

        eval_result = evaluate_tools(item["text"], camel_res, stanza_res, farasa_res)

        gold_pos = {g["word"]: g["pos"] for g in item["gold"]}
        camel_tokens = camel_res.get("tokens", [])

        camel_correct = sum(
            1
            for t in camel_tokens
            if normalize_pos_for_compare(t.get("analyses", [{}])[0].get("pos", ""))
            == gold_pos.get(t["surface"], "")
        )

        results.append(
            {
                "text": item["text"],
                "pos_agreement_pct": eval_result["pos_agreement_pct"],
                "lemma_match_pct": eval_result["lemma_match_pct"],
                "pos_f1": eval_result["pos_f1"],
                "camel_vs_gold": f"{camel_correct}/{len(item['gold'])}",
            }
        )

    avg_f1 = sum(r["pos_f1"] for r in results) / len(results) if results else 0.0

    return {
        "total_sentences": len(results),
        "average_f1": round(avg_f1, 3),
        "average_f1_pct": f"{round(avg_f1 * 100, 1)}%",
        "results": results,
    }

