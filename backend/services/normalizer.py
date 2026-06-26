from __future__ import annotations

from typing import Any, Dict, List, Optional


# Required frontend-friendly normalized token schema (no extra nested metadata).
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
    # VERB
    "VERB": "VERB",
    "فعل": "VERB",
    "V": "VERB",

    # NOUN
    "NOUN": "NOUN",
    "اسم": "NOUN",
    "N": "NOUN",

    # ADJ
    "ADJ": "ADJ",
    "adjective": "ADJ",
    "ADJECTIVE": "ADJ",
    "صفة": "ADJ",

    # ADV
    "ADV": "ADV",
    "ADVERB": "ADV",
    "adv": "ADV",
    "ظرف": "ADV",

    # PRON
    "PRON": "PRON",
    "PRONOUN": "PRON",
    "pronoun": "PRON",
    "ضمير": "PRON",

    # ADP
    "ADP": "ADP",
    "ADPOSITION": "ADP",
    "ADPOSITION": "ADP",

    # PART
    "PART": "PART",
    "PARTICLE": "PART",
    "حرف": "PART",
    "حرف جر": "PART",

    # CCONJ
    "CCONJ": "CCONJ",
    "CONJ": "CCONJ",
    "CONJUNCTION": "CCONJ",
    "و": "CCONJ",
    "أو": "CCONJ",

    # SCONJ
    "SCONJ": "SCONJ",
    "SUBORDINATING_CONJ": "SCONJ",

    # PUNCT
    "PUNCT": "PUNCT",
    "PUNCTUATION": "PUNCT",
    "PUNCTUATION": "PUNCT",

    # NUM
    "NUM": "NUM",
    "NUMBER": "NUM",
    "NUMERAL": "NUM",

    # fallback
}


def _pos_standardize(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    key = str(pos).strip()
    if not key:
        return None

    # Normalize a few common cases
    up = key.upper()
    if up in POS_STANDARD:
        return POS_STANDARD[up]

    # Try Arabic direct mapping and other known keys as-is
    if key in POS_STANDARD:
        return POS_STANDARD[key]

    # Handle common UD pos values
    if up in {"PROPN", "NOUN_PROP", "NOUNPROPER"}:
        return "NOUN"

    if up in {"ADP"}:
        return "ADP"

    if up in {"X"}:
        return "X"

    # If already one of the standardized values, keep it.
    if up in {"VERB", "NOUN", "ADJ", "ADV", "PRON", "ADP", "PART", "CCONJ", "SCONJ", "PUNCT", "NUM", "X"}:
        return up

    # unknown -> X
    return "X"


def _normalize_pos_text(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    text = str(pos).strip()
    if not text:
        return None
    return text.replace("ـ", "").replace("ﻻ", "لا")


def normalize_alkhalil_pos(pos: Optional[str]) -> Optional[str]:
    """Map AlKhalil grammatical descriptions to Universal POS."""
    text = _normalize_pos_text(pos)
    if not text:
        return None

    lowered = text.lower()
    compact = lowered.replace(" ", "")

    direct_map = {
        "فعل": "VERB",
        "ماض": "VERB",
        "مضارع": "VERB",
        "أمر": "VERB",
        "امر": "VERB",
        "اسم": "NOUN",
        "فاعل": "NOUN",
        "مفعول": "NOUN",
        "مصدر": "NOUN",
        "صفة": "ADJ",
        "مشبه": "ADJ",
        "منسوب": "ADJ",
        "ظرف": "ADV",
        "حرف جر": "ADP",
        "حرف عطف": "CCONJ",
        "ضمير": "PRON",
    }
    for needle, mapped in direct_map.items():
        if needle in text:
            return mapped

    if any(needle in compact for needle in ("فعل", "ماض", "مضارع", "امر", "past", "present", "imperative")):
        return "VERB"
    if any(needle in compact for needle in ("اسم", "فاعل", "مفعول", "مصدر", "noun")):
        return "NOUN"
    if any(needle in compact for needle in ("صفة", "مشبه", "منسوب", "adjective")):
        return "ADJ"
    if "ظرف" in compact or "adverb" in compact:
        return "ADV"
    if "حرفجر" in compact or "adposition" in compact:
        return "ADP"
    if "حرفعطف" in compact or "conjunction" in compact:
        return "CCONJ"
    if "ضمير" in compact or "pronoun" in compact:
        return "PRON"

    return None



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
    return [
        {
            "lemma": lemma,
            "root": root,
            "pos": pos,
            "gender": gender,
            "number": number,
            "tense": tense,
            "gloss": gloss,
        }
    ]


def _deep_copy_template() -> Dict[str, Any]:
    # Avoid shared mutable state
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
    if confidence:
        t["confidence"] = confidence
    else:
        t["confidence"] = {"score": 0.0, "level": "low"}

    # Backward/forward compatibility: historically the project used extra meta.
    # For the required frontend schema we DO NOT include meta.
    # (We keep `source_tool` out of the token payload for payload size.)
    _ = source_tool
    _ = meta
    return t



def normalize_camel_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    tokens_out: List[Dict[str, Any]] = []
    for tok in raw_result.get("tokens", []) or []:
        surface = tok.get("surface")
        segmentation = tok.get("segmentation")
        analyses = tok.get("analyses") or []
        best = analyses[0] if analyses else {}

        features = {
            "gender": best.get("gender"),
            "number": best.get("number"),
            "tense": best.get("tense"),
            "case": None,
            "definite": None,
            "voice": None,
            "person": None,
        }

        dependency = {
            "head": None,
            "head_text": None,
            "deprel": None,
        }

        score = best.get("confidence")
        conf = _confidence(score)

        meta = {
            "root_type": best.get("root_type"),
            "corrections": best.get("corrections") or [],
            "notes": [],
        }

        tokens_out.append(
            _unified_token(
                source_tool="camel",
                surface=surface,
                lemma=best.get("lemma"),
                root=best.get("root"),
                pos=_pos_standardize(best.get("pos")),
                gloss=best.get("gloss"),

                segmentation=segmentation if isinstance(segmentation, list) else [surface],
                features=features,
                dependency=dependency,
                confidence=conf,
                meta=meta,
            ) | {
                "meta": meta,
                "analyses": _analysis_payload(
                    lemma=best.get("lemma"),
                    root=best.get("root"),
                    pos=_pos_standardize(best.get("pos")),
                    gender=best.get("gender"),
                    number=best.get("number"),
                    tense=best.get("tense"),
                    gloss=best.get("gloss"),
                )
            }
        )

    return {
        "tool": "camel",
        "status": raw_result.get("status", "ok"),
        "input": raw_result.get("input"),
        "word_count": len(tokens_out),
        "tokens": tokens_out,
    }


def normalize_farasa_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    tokens_out: List[Dict[str, Any]] = []
    for tok in raw_result.get("tokens", []) or []:
        surface = tok.get("surface")
        segmentation = tok.get("segmentation") or [surface]
        tokens_out.append(
            _unified_token(
                source_tool="farasa",
                surface=surface,
                lemma=None,
                root=None,
                pos=None,
                gloss=None,
                segmentation=segmentation,
                features=None,
                dependency=None,
                confidence=_confidence(None),
                meta={"root_type": None, "corrections": [], "notes": []},
            ) | {
                "meta": {"root_type": None, "corrections": [], "notes": []},
                "analyses": _analysis_payload(),
            }
        )

    return {
        "tool": "farasa",
        "status": raw_result.get("status", "ok"),
        "input": raw_result.get("input"),
        "word_count": len(tokens_out),
        "tokens": tokens_out,
    }


def normalize_stanza_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    tokens_out: List[Dict[str, Any]] = []
    for tok in raw_result.get("tokens", []) or []:
        surface = tok.get("surface")
        feats: Dict[str, Any] = {}
        # existing stanza wrapper uses explicit keys
        feats["gender"] = tok.get("gender")
        feats["number"] = tok.get("number")
        feats["tense"] = tok.get("tense")
        feats["case"] = tok.get("case")
        feats["definite"] = tok.get("definite")
        feats["voice"] = tok.get("voice")
        feats["person"] = tok.get("person")

        dependency_raw = tok.get("dependency") or {}
        dependency = {
            "head": dependency_raw.get("head"),
            "head_text": dependency_raw.get("head_text"),
            "deprel": dependency_raw.get("deprel"),
        }

        segmentation = [surface]
        tokens_out.append(
            _unified_token(
                source_tool="stanza",
                surface=surface,
                lemma=tok.get("lemma"),
                root=None,
                pos=tok.get("pos") or tok.get("upos"),
                gloss=None,
                segmentation=segmentation,
                features=feats,
                dependency=dependency,
                confidence=_confidence(None),
                meta={"root_type": None, "corrections": [], "notes": []},
            ) | {
                "meta": {"root_type": None, "corrections": [], "notes": []},
                "analyses": _analysis_payload(
                    lemma=tok.get("lemma"),
                    root=None,
                    pos=tok.get("pos") or tok.get("upos"),
                    gender=feats.get("gender"),
                    number=feats.get("number"),
                    tense=feats.get("tense"),
                    gloss=None,
                )
            }
        )

    return {
        "tool": "stanza",
        "status": raw_result.get("status", "ok"),
        "input": raw_result.get("input"),
        "word_count": len(tokens_out),
        "tokens": tokens_out,
    }


QALSADI_POS_MAP = {
    "فعل": "VERB",
    "اسم": "NOUN",
    "صفة": "ADJ",
    "حرف": "PART",
    "ضمير": "PRON",
    "ظرف": "ADV",
    # fallbacks already in wrappers
    "ADVERB": "ADV",
    "ADJECTIVE": "ADJ",
    "NOUN": "NOUN",
    "VERB": "VERB",
    "PART": "PART",
    "PRON": "PRON",
}


def normalize_qalsadi_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    tokens_out: List[Dict[str, Any]] = []
    for tok in raw_result.get("tokens", []) or []:
        surface = tok.get("surface")
        lemma = tok.get("lemma")
        root = tok.get("root")
        pos_raw = tok.get("pos")
        gloss = None
        pos = QALSADI_POS_MAP.get(pos_raw, pos_raw)
        normalized_flag = bool(tok.get("normalized"))

        tokens_out.append(
            _unified_token(
                source_tool="qalsadi",
                surface=surface,
                lemma=lemma,
                root=root,
                pos=pos,
                gloss=gloss,
                segmentation=[surface],
                features=None,
                dependency=None,
                confidence=_confidence(tok.get("freq")),
                meta={"root_type": None, "corrections": [], "notes": []},
            ) | {
                "original_surface": tok.get("original_surface"),
                "normalized": normalized_flag,
                "note": tok.get("note") if normalized_flag else None,
                "analyses": _analysis_payload(
                    lemma=lemma,
                    root=root,
                    pos=pos,
                    gender=None,
                    number=None,
                    tense=None,
                    gloss=None,
                )
            }
        )

    return {
        "tool": "qalsadi",
        "status": raw_result.get("status", "ok"),
        "input": raw_result.get("input"),
        "word_count": len(tokens_out),
        "tokens": tokens_out,
    }


def normalize_tool_output(tool_name: str, raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize any supported tool output into the unified schema."""

    # NOTE: If raw_result is already normalized (tokens include `features` + `dependency`),
    # pass it through to remain migration-safe.
    if isinstance(raw_result, dict) and isinstance(raw_result.get("tokens"), list):
        sample = next((t for t in raw_result["tokens"] if isinstance(t, dict)), None)
        if sample and "features" in sample and "dependency" in sample and "confidence" in sample and "meta" in sample:
            return raw_result

    if not isinstance(raw_result, dict):
        return {
            "tool": tool_name,
            "status": "error",
            "input": None,
            "word_count": 0,
            "tokens": [],
        }


    if raw_result.get("status") == "error" and not raw_result.get("tokens"):
        return {
            "tool": tool_name,
            "status": "error",
            "input": raw_result.get("input"),
            "word_count": 0,
            "tokens": [],
            "error": raw_result.get("error"),
        }

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
        raw_tokens = raw_result.get("tokens", []) or []
        raw_lemmas = raw_result.get("lemmas", []) or []
        for idx, tok in enumerate(raw_tokens):
            original_surface = None
            normalized_flag = False
            note = None
            gloss_value = None
            if isinstance(tok, dict):
                surface = tok.get("surface")
                lemma = tok.get("lemma")
                root = tok.get("root")
                raw_pos = tok.get("pos_raw") or tok.get("raw_pos") or tok.get("pos")
                pos = normalize_alkhalil_pos(raw_pos or tok.get("upos"))
                gloss = tok.get("gloss")
                original_surface = tok.get("original_surface")
                normalized_flag = bool(tok.get("normalized"))
                note = tok.get("note")
                gloss_value = tok.get("gloss")
            else:
                surface = str(tok)
                lemma = raw_lemmas[idx] if idx < len(raw_lemmas) else surface
                root = None
                pos = None
                gloss = None
                raw_pos = None
            tokens_out.append(
                _unified_token(
                    source_tool="alkhalil",
                    surface=surface,
                    lemma=lemma,
                    root=root,
                    pos=pos,
                    gloss=gloss,
                    segmentation=[surface],
                    features=None,
                    dependency=None,
                    confidence=_confidence(None),
                    meta={"root_type": None, "corrections": [], "notes": []},
                ) | {
                    "meta": {"root_type": None, "corrections": [], "notes": []},
                    "original_surface": original_surface,
                    "normalized": normalized_flag,
                    "note": note,
                    "pos_raw": raw_pos if isinstance(tok, dict) else None,
                    "analyses": _analysis_payload(
                        lemma=lemma,
                        root=root,
                        pos=pos,
                        gender=None,
                        number=None,
                        tense=None,
                        gloss=gloss_value,
                    )
                }
            )
        return {
            "tool": "alkhalil",
            "status": raw_result.get("status", "ok"),
            "input": raw_result.get("input"),
            "word_count": len(tokens_out),
            "tokens": tokens_out,
        }
    if tool_name == "udpipe":
        tokens_out: List[Dict[str, Any]] = []
        for tok in raw_result.get("tokens", []) or []:
            surface = tok.get("surface")
            lemma = tok.get("lemma")
            pos = _pos_standardize(tok.get("upos") or tok.get("pos"))
            dependency_raw = tok.get("dependency") or {}
            tokens_out.append(
                _unified_token(
                    source_tool="udpipe",
                    surface=surface,
                    lemma=lemma,
                    root=None,
                    pos=pos,
                    gloss=None,
                    segmentation=[surface],
                    features={"case": tok.get("case")},
                    dependency={
                        "head": dependency_raw.get("head"),
                        "head_text": dependency_raw.get("head_text"),
                        "deprel": dependency_raw.get("deprel"),
                    },
                    confidence=_confidence(None),
                    meta={"root_type": None, "corrections": [], "notes": []},
                ) | {
                    "meta": {"root_type": None, "corrections": [], "notes": []},
                    "analyses": _analysis_payload(
                        lemma=lemma,
                        root=None,
                        pos=pos,
                        gender=None,
                        number=None,
                        tense=None,
                        gloss=None,
                    )
                }
            )
        return {
            "tool": "udpipe",
            "status": raw_result.get("status", "ok"),
            "input": raw_result.get("input"),
            "word_count": len(tokens_out),
            "tokens": tokens_out,
        }

    # Partner/unknown tools: return empty normalized structure.
    return {
        "tool": tool_name,
        "status": "ok",
        "input": raw_result.get("input"),
        "word_count": 0,
        "tokens": [],
    }

