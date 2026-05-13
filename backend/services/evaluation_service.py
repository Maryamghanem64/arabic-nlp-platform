from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.services.comparison_service import _lemma_equal, _pos_equal


def evaluate_agreement(text: str, *, camel_res: Dict[str, Any], stanza_res: Dict[str, Any], farasa_res: Dict[str, Any]) -> Dict[str, Any]:
    # Uses farasa tokens for length as previously.
    farasa_tokens: List[Dict[str, Any]] = farasa_res.get("tokens", []) if farasa_res else []
    total = len(farasa_tokens)

    camel_tokens = camel_res.get("tokens", []) if camel_res else []
    stanza_tokens = stanza_res.get("tokens", []) if stanza_res else []

    pos_match = 0
    lemma_match = 0
    seg_cov = 0

    for i in range(total):
        camel_tok = camel_tokens[i] if i < len(camel_tokens) else None
        stanza_tok = stanza_tokens[i] if i < len(stanza_tokens) else None
        farasa_tok = farasa_tokens[i]

        if camel_tok and stanza_tok:
            if _pos_equal(camel_tok.get("pos"), stanza_tok.get("pos") or stanza_tok.get("upos")):
                pos_match += 1
            if _lemma_equal(camel_tok.get("lemma"), stanza_tok.get("lemma")):
                lemma_match += 1

        if farasa_tok and farasa_tok.get("segmentation"):
            seg_cov += 1

    return {
        "total_words": total,
        "pos_agreement_pct": f"{(pos_match / total * 100) if total else 0:.1f}%",
        "lemma_match_pct": f"{(lemma_match / total * 100) if total else 0:.1f}%",
        "segmentation_coverage": (seg_cov / total) if total else 0.0,
    }

