from __future__ import annotations

from typing import Any, Dict

from app.utils.helpers import classify_conflict, normalize_pos_for_compare, strip_diacritics


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def evaluate_tools(text: str, camel_res, stanza_res, farasa_res):
    """Evaluate CAMeL vs Stanza using surface-string alignment (not index).

    Index-based alignment results in systematic 0% scores when tools tokenize
    into different token counts/order.
    """

    farasa_tokens = farasa_res.get("tokens", []) or []
    camel_tokens = camel_res.get("tokens", []) or []
    stanza_tokens = stanza_res.get("tokens", []) or []

    # Use Farasa surfaces as the UI token anchor
    words = [t.get("surface") for t in farasa_tokens if isinstance(t, dict) and t.get("surface")]
    total = len(words)

    # Build surface -> first token maps
    camel_map = {}
    for t in camel_tokens:
        if not isinstance(t, dict):
            continue
        s = t.get("surface")
        if s and s not in camel_map:
            camel_map[s] = t

    stanza_map = {}
    for t in stanza_tokens:
        if not isinstance(t, dict):
            continue
        s = t.get("surface")
        if s and s not in stanza_map:
            stanza_map[s] = t

    pos_tp = pos_fp = pos_fn = 0
    lemma_match = 0
    seg_coverage = 0
    conflicts = []
    all_conflicts = []

    for w in words:
        camel_tok = camel_map.get(w)
        stanza_tok = stanza_map.get(w)

        # segmentation coverage
        f_tok = next((t for t in farasa_tokens if isinstance(t, dict) and t.get("surface") == w), None)
        if f_tok and f_tok.get("segmentation"):
            seg_coverage += 1

        camel_ana = None
        if camel_tok and isinstance(camel_tok, dict):
            analyses = camel_tok.get("analyses") or []
            if analyses and isinstance(analyses[0], dict):
                camel_ana = analyses[0]

        if camel_ana and stanza_tok:
            camel_pos = normalize_pos_for_compare(camel_ana.get("pos"))
            stanza_pos = str(stanza_tok.get("upos", "")).upper() if stanza_tok.get("upos") else ""

            if camel_pos and stanza_pos:
                if camel_pos == stanza_pos:
                    pos_tp += 1
                else:
                    pos_fp += 1
                    pos_fn += 1
                    conflicts.append({"word": w, "camel_pos": camel_pos, "stanza_pos": stanza_pos})
                    all_conflicts.append(classify_conflict("pos", camel_pos, stanza_pos))

            c_lemma = strip_diacritics(camel_ana.get("lemma"))
            s_lemma = strip_diacritics(stanza_tok.get("lemma"))
            if c_lemma and s_lemma:
                if c_lemma == s_lemma:
                    lemma_match += 1
                else:
                    all_conflicts.append(classify_conflict("lemma", c_lemma, s_lemma))

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


