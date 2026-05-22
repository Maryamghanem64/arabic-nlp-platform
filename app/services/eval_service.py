from __future__ import annotations

from typing import Any, Dict

from app.utils.helpers import classify_conflict, normalize_pos_for_compare, strip_diacritics


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def evaluate_tools(text: str, camel_res, stanza_res, farasa_res):
    # SIMPLE VERSION — no modular imports, always works (kept identical to main.py behavior)
    words = [t["surface"] for t in farasa_res.get("tokens", [])]
    camel_tokens = camel_res.get("tokens", [])
    stanza_tokens = stanza_res.get("tokens", [])
    farasa_tokens = farasa_res.get("tokens", [])
    total = len(words)

    pos_tp = pos_fp = pos_fn = 0
    lemma_match = 0
    seg_coverage = 0
    conflicts = []
    all_conflicts = []

    for i in range(total):
        camel_ana = (
            camel_tokens[i]["analyses"][0]
            if i < len(camel_tokens) and camel_tokens[i].get("analyses")
            else None
        )
        stanza_tok = stanza_tokens[i] if i < len(stanza_tokens) else None
        farasa_tok = farasa_tokens[i] if i < len(farasa_tokens) else None

        if camel_ana and stanza_tok:
            camel_pos = normalize_pos_for_compare(camel_ana.get("pos"))
            stanza_pos = stanza_tok.get("upos", "").upper()

            if camel_pos and stanza_pos:
                if camel_pos == stanza_pos:
                    pos_tp += 1
                else:
                    pos_fp += 1
                    pos_fn += 1
                    conflicts.append({"word": words[i], "camel_pos": camel_pos, "stanza_pos": stanza_pos})
                    all_conflicts.append(classify_conflict("pos", camel_pos, stanza_pos))

            c_lemma = strip_diacritics(camel_ana.get("lemma"))
            s_lemma = strip_diacritics(stanza_tok.get("lemma"))
            if c_lemma and s_lemma:
                if c_lemma == s_lemma:
                    lemma_match += 1
                else:
                    all_conflicts.append(classify_conflict("lemma", c_lemma, s_lemma))

        if farasa_tok and farasa_tok.get("segmentation"):
            seg_coverage += 1

    pos_agreement = pos_tp / total if total else 0
    pos_prf = compute_prf(pos_tp, pos_fp, pos_fn)

    return {
        "total_words": total,
        "pos_agreement": round(pos_agreement, 2),
        "pos_agreement_pct": f"{round(pos_agreement * 100, 1)}%",
        "pos_precision": pos_prf["precision"],
        "pos_recall": pos_prf["recall"],
        "pos_f1": pos_prf["f1"],
        "lemma_match": round(lemma_match / total, 2) if total else 0,
        "lemma_match_pct": f"{round(lemma_match / total * 100, 1)}%" if total else "0%",
        "segmentation_coverage": round(seg_coverage / total, 2) if total else 0,
        "pos_conflicts": conflicts,
        "all_conflicts": all_conflicts,
    }

