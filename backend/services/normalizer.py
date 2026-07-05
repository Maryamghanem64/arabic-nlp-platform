from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


UNIFIED_TOKEN_TEMPLATE: Dict[str, Any] = {
    "surface": None,
    "lemma": None,
    "root": None,
    "pos": None,
    "gloss": None,
    "features": {
        "gender": None,
        "number": None,
        "tense": None,
        "person": None,
        "case": None,
        "definite": None,
        "voice": None,
    },
    "segmentation": None,
    "dependency": {"head": None, "head_text": None, "deprel": None},
    "confidence": {"score": 0.0, "level": "low"},
}


POS_STANDARD = {
    "VERB": "VERB", "فعل": "VERB", "V": "VERB",
    "NOUN": "NOUN", "اسم": "NOUN", "N": "NOUN",
    "ADJ": "ADJ", "ADJECTIVE": "ADJ", "adjective": "ADJ", "صفة": "ADJ",
    "ADV": "ADV", "ADVERB": "ADV", "adv": "ADV", "ظرف": "ADV",
    "PRON": "PRON", "PRONOUN": "PRON", "pronoun": "PRON", "ضمير": "PRON",
    "ADP": "ADP", "ADPOSITION": "ADP",
    "PART": "PART", "PARTICLE": "PART", "حرف": "PART",
    "CCONJ": "CCONJ", "CONJ": "CCONJ", "CONJUNCTION": "CCONJ", "و": "CCONJ", "أو": "CCONJ",
    "SCONJ": "SCONJ", "SUBORDINATING_CONJ": "SCONJ",
    "PUNCT": "PUNCT", "PUNCTUATION": "PUNCT",
    "NUM": "NUM", "NUMBER": "NUM", "NUMERAL": "NUM",
}

_VALID_POS = {"VERB", "NOUN", "ADJ", "ADV", "PRON", "ADP", "PART", "CCONJ", "SCONJ", "PUNCT", "NUM", "X"}


def _pos_standardize(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    key = str(pos).strip()
    if not key:
        return None

    up = key.upper()
    if up in POS_STANDARD:
        return POS_STANDARD[up]
    if key in POS_STANDARD:
        return POS_STANDARD[key]
    if up in {"PROPN", "NOUN_PROP", "NOUNPROPER"}:
        return "NOUN"
    if up in _VALID_POS:
        return up
    return "X"


def _normalize_pos_text(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    text = str(pos).strip()
    if not text:
        return None
    return text.replace("ـ", "").replace("ﻻ", "لا")


def normalize_alkhalil_pos(value: Optional[str]) -> Optional[str]:
    """Map AlKhalil Arabic grammatical descriptions to UD-style POS labels."""
    if value is None:
        return None

    text = _normalize_pos_text(value)
    if not text or text in {"#", "X", "x", "UNK", "unknown", "None", "null", "0"}:
        return None

    compact = text.replace(" ", "")
    up = text.upper()
    if up in _VALID_POS:
        return up

    if "حرف جر" in text:
        return "ADP"
    if "حرف عطف" in text:
        return "CCONJ"
    if "ضمير" in text:
        return "PRON"
    if "ظرف" in text:
        return "ADV"
    if "صفة" in text:
        return "ADJ"
    if "فعل" in text or "ماض" in text or "مضارع" in text or "أمر" in text or "امر" in compact:
        return "VERB"
    if any(k in text for k in ("اسم", "مفرد", "مثنى", "جمع", "فاعل", "مفعول", "مصدر")):
        return "NOUN"
    if "حرف" in text:
        return "PART"

    mapped = _pos_standardize(text)
    if mapped and mapped != "X":
        return mapped
    return None


def extract_alkhalil_canonical_pos(
    token: Dict[str, Any] | None,
    context_pos_votes: Dict[str, str] | None = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (normalized_pos, raw_selected_value) for an AlKhalil token.

    It inspects all analyses and prefers the candidate that matches the majority
    POS vote from other tools when provided. This prevents false conflicts such
    as selecting a verbal analysis for "في" when other tools agree on ADP and
    AlKhalil has a "حرف جر" candidate among its analyses.
    """
    if not isinstance(token, dict):
        return None, None

    candidates: List[Dict[str, Any]] = []

    def add_candidate(raw: Any, index: int) -> None:
        if raw is None:
            return
        raw_text = str(raw).strip()
        if not raw_text or raw_text in {"#", "X", "x", "UNK", "unknown", "None", "null", "0"}:
            return
        normalized = normalize_alkhalil_pos(raw_text)
        if normalized:
            candidates.append({"normalized_pos": normalized, "raw_value": raw_text, "analysis_index": index})

    # Top-level fields are kept for traceability, but analyses are usually richer.
    for raw in (token.get("pos"), token.get("upos"), token.get("pos_raw"), token.get("raw_pos"), token.get("gloss")):
        add_candidate(raw, -1)

    analyses = token.get("analyses") or []
    if isinstance(analyses, list):
        for i, analysis in enumerate(analyses):
            if not isinstance(analysis, dict):
                continue
            # Required priority: type, gloss, pos.
            for field in ("type", "gloss", "pos"):
                add_candidate(analysis.get(field), i)

    if not candidates:
        raw = token.get("pos_raw") or token.get("raw_pos") or token.get("pos") or token.get("gloss")
        return None, str(raw).strip() if raw else None

    if context_pos_votes:
        counts: Dict[str, int] = {}
        for voted in context_pos_votes.values():
            if voted:
                counts[str(voted).strip().upper()] = counts.get(str(voted).strip().upper(), 0) + 1
        if counts:
            majority_pos = max(counts.items(), key=lambda kv: kv[1])[0]
            for candidate in candidates:
                if candidate["normalized_pos"] == majority_pos:
                    return candidate["normalized_pos"], candidate["raw_value"]

    first = candidates[0]
    return first["normalized_pos"], first["raw_value"]


def _confidence(score: Optional[float]) -> Dict[str, Any]:
    score_f = float(score) if score is not None else 0.0
    level = "low"
    if score_f >= 0.9:
        level = "high"
    elif score_f >= 0.6:
        level = "medium"
    return {"score": score_f, "level": level}


def _analysis_payload(
    *,
    lemma: Any = None,
    root: Any = None,
    pos: Any = None,
    gender: Any = None,
    number: Any = None,
    tense: Any = None,
    gloss: Any = None,
) -> List[Dict[str, Any]]:
    return [{"lemma": lemma, "root": root, "pos": pos, "gender": gender, "number": number, "tense": tense, "gloss": gloss}]


def _deep_copy_template() -> Dict[str, Any]:
    import copy
    return copy.deepcopy(UNIFIED_TOKEN_TEMPLATE)


def _unified_token(
    *,
    source_tool: str,
    surface: str,
    lemma: Optional[str],
    root: Optional[str],
    pos: Optional[str],
    gloss: Optional[str],
    segmentation: Optional[List[str]],
    features: Optional[Dict[str, Any]] = None,
    dependency: Optional[Dict[str, Any]] = None,
    confidence: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t = _deep_copy_template()
    t["surface"] = surface
    t["lemma"] = lemma
    t["root"] = root
    t["pos"] = pos
    t["gloss"] = gloss
    t["segmentation"] = segmentation
    if features:
        for k, v in features.items():
            t["features"][k] = v
    if dependency:
        for k, v in dependency.items():
            t["dependency"][k] = v
    t["confidence"] = confidence or {"score": 0.0, "level": "low"}
    _ = source_tool
    _ = meta
    return t


def normalize_camel_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    tokens_out: List[Dict[str, Any]] = []
    for tok in raw_result.get("tokens", []) or []:
        surface = tok.get("surface")
        analyses = tok.get("analyses") or []
        best = analyses[0] if analyses else {}
        segmentation = tok.get("segmentation")
        features = {
            "gender": best.get("gender"), "number": best.get("number"), "tense": best.get("tense"),
            "case": None, "definite": None, "voice": None, "person": None,
        }
        meta = {"root_type": best.get("root_type"), "corrections": best.get("corrections") or [], "notes": []}
        pos = _pos_standardize(best.get("pos"))
        tokens_out.append(
            _unified_token(
                source_tool="camel", surface=surface, lemma=best.get("lemma"), root=best.get("root"), pos=pos,
                gloss=best.get("gloss"), segmentation=segmentation if isinstance(segmentation, list) else [surface],
                features=features, dependency={"head": None, "head_text": None, "deprel": None},
                confidence=_confidence(best.get("confidence")), meta=meta,
            ) | {"meta": meta, "analyses": _analysis_payload(lemma=best.get("lemma"), root=best.get("root"), pos=pos, gender=best.get("gender"), number=best.get("number"), tense=best.get("tense"), gloss=best.get("gloss"))}
        )
    return {"tool": "camel", "status": raw_result.get("status", "ok"), "input": raw_result.get("input"), "word_count": len(tokens_out), "tokens": tokens_out}


def normalize_farasa_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    tokens_out: List[Dict[str, Any]] = []
    for tok in raw_result.get("tokens", []) or []:
        surface = tok.get("surface")
        segmentation = tok.get("segmentation") or [surface]
        tokens_out.append(
            _unified_token(source_tool="farasa", surface=surface, lemma=None, root=None, pos=None, gloss=None, segmentation=segmentation, features=None, dependency=None, confidence=_confidence(None), meta={})
            | {"meta": {"root_type": None, "corrections": [], "notes": []}, "analyses": _analysis_payload()}
        )
    return {"tool": "farasa", "status": raw_result.get("status", "ok"), "input": raw_result.get("input"), "word_count": len(tokens_out), "tokens": tokens_out}


def normalize_stanza_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    tokens_out: List[Dict[str, Any]] = []
    for tok in raw_result.get("tokens", []) or []:
        surface = tok.get("surface")
        feats = {"gender": tok.get("gender"), "number": tok.get("number"), "tense": tok.get("tense"), "case": tok.get("case"), "definite": tok.get("definite"), "voice": tok.get("voice"), "person": tok.get("person")}
        dep_raw = tok.get("dependency") or {}
        pos = tok.get("pos") or tok.get("upos")
        tokens_out.append(
            _unified_token(source_tool="stanza", surface=surface, lemma=tok.get("lemma"), root=None, pos=pos, gloss=None, segmentation=[surface], features=feats, dependency={"head": dep_raw.get("head"), "head_text": dep_raw.get("head_text"), "deprel": dep_raw.get("deprel")}, confidence=_confidence(None), meta={})
            | {"meta": {"root_type": None, "corrections": [], "notes": []}, "analyses": _analysis_payload(lemma=tok.get("lemma"), root=None, pos=pos, gender=feats.get("gender"), number=feats.get("number"), tense=feats.get("tense"), gloss=None)}
        )
    return {"tool": "stanza", "status": raw_result.get("status", "ok"), "input": raw_result.get("input"), "word_count": len(tokens_out), "tokens": tokens_out}


QALSADI_POS_MAP = {"فعل": "VERB", "اسم": "NOUN", "صفة": "ADJ", "حرف": "PART", "ضمير": "PRON", "ظرف": "ADV", "ADVERB": "ADV", "ADJECTIVE": "ADJ", "NOUN": "NOUN", "VERB": "VERB", "PART": "PART", "PRON": "PRON"}


def normalize_qalsadi_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    tokens_out: List[Dict[str, Any]] = []
    for tok in raw_result.get("tokens", []) or []:
        surface = tok.get("surface")
        lemma = tok.get("lemma")
        root = tok.get("root")
        pos = QALSADI_POS_MAP.get(tok.get("pos"), tok.get("pos"))
        normalized_flag = bool(tok.get("normalized"))
        tokens_out.append(
            _unified_token(source_tool="qalsadi", surface=surface, lemma=lemma, root=root, pos=pos, gloss=None, segmentation=[surface], features=None, dependency=None, confidence=_confidence(tok.get("freq")), meta={})
            | {"original_surface": tok.get("original_surface"), "normalized": normalized_flag, "note": tok.get("note") if normalized_flag else None, "analyses": _analysis_payload(lemma=lemma, root=root, pos=pos)}
        )
    return {"tool": "qalsadi", "status": raw_result.get("status", "ok"), "input": raw_result.get("input"), "word_count": len(tokens_out), "tokens": tokens_out}


def normalize_tool_output(tool_name: str, raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize any supported tool output into the unified schema."""
    if not isinstance(raw_result, dict):
        return {"tool": tool_name, "status": "error", "input": None, "word_count": 0, "tokens": []}

    if isinstance(raw_result.get("tokens"), list):
        sample = next((t for t in raw_result["tokens"] if isinstance(t, dict)), None)
        if sample and "features" in sample and "dependency" in sample and "confidence" in sample and "meta" in sample:
            return raw_result

    if raw_result.get("status") == "error" and not raw_result.get("tokens"):
        return {"tool": tool_name, "status": "error", "input": raw_result.get("input"), "word_count": 0, "tokens": [], "error": raw_result.get("error")}

    if tool_name == "camel":
        return normalize_camel_output(raw_result)
    if tool_name == "farasa":
        return normalize_farasa_output(raw_result)
    if tool_name == "stanza":
        return normalize_stanza_output(raw_result)
    if tool_name == "qalsadi":
        return normalize_qalsadi_output(raw_result)

    if tool_name == "alkhalil":
        tokens_out: List[Dict[str, Any]] = []
        for tok in raw_result.get("tokens", []) or []:
            if isinstance(tok, dict):
                surface = tok.get("surface")
                lemma = tok.get("lemma")
                root = tok.get("root")
                pos, raw_pos = extract_alkhalil_canonical_pos(tok)
                gloss = tok.get("gloss")
                original_surface = tok.get("original_surface")
                normalized_flag = bool(tok.get("normalized"))
                note = tok.get("note")
                analyses = tok.get("analyses") if isinstance(tok.get("analyses"), list) else _analysis_payload(lemma=lemma, root=root, pos=pos, gloss=gloss)
            else:
                surface = str(tok)
                lemma = surface
                root = pos = raw_pos = gloss = original_surface = note = None
                normalized_flag = False
                analyses = _analysis_payload(lemma=lemma)
            tokens_out.append(
                _unified_token(source_tool="alkhalil", surface=surface, lemma=lemma, root=root, pos=pos, gloss=gloss, segmentation=[surface], features=None, dependency=None, confidence=_confidence(None), meta={})
                | {"meta": {"root_type": None, "corrections": [], "notes": []}, "original_surface": original_surface, "normalized": normalized_flag, "note": note, "pos_raw": raw_pos, "analyses": analyses}
            )
        return {"tool": "alkhalil", "status": raw_result.get("status", "ok"), "input": raw_result.get("input"), "word_count": len(tokens_out), "tokens": tokens_out}

    if tool_name == "udpipe":
        tokens_out: List[Dict[str, Any]] = []
        for tok in raw_result.get("tokens", []) or []:
            surface = tok.get("surface")
            lemma = tok.get("lemma")
            pos = _pos_standardize(tok.get("upos") or tok.get("pos"))
            dep_raw = tok.get("dependency") or {}
            tokens_out.append(
                _unified_token(source_tool="udpipe", surface=surface, lemma=lemma, root=None, pos=pos, gloss=None, segmentation=[surface], features={"case": tok.get("case")}, dependency={"head": dep_raw.get("head"), "head_text": dep_raw.get("head_text"), "deprel": dep_raw.get("deprel")}, confidence=_confidence(None), meta={})
                | {"meta": {"root_type": None, "corrections": [], "notes": []}, "analyses": _analysis_payload(lemma=lemma, root=None, pos=pos)}
            )
        return {"tool": "udpipe", "status": raw_result.get("status", "ok"), "input": raw_result.get("input"), "word_count": len(tokens_out), "tokens": tokens_out}

    return {"tool": tool_name, "status": raw_result.get("status", "ok"), "input": raw_result.get("input"), "word_count": raw_result.get("word_count", 0), "tokens": raw_result.get("tokens", []) or []}
