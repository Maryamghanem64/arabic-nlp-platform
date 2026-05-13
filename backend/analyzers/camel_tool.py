from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize

from backend.analyzers.base import Analyzer
from backend.utils.text_norm import normalize_whitespace

# --- normalization maps (duplicated to keep module self-contained) ---
ASPECT_MAP = {"p": "past", "i": "present", "c": "imperative", "na": None}
GENDER_MAP = {"m": "masculine", "f": "feminine", "na": None}
NUMBER_MAP = {"s": "singular", "d": "dual", "p": "plural", "na": None}

POS_MAP = {
    "noun": "NOUN",
    "verb": "VERB",
    "adj": "ADJECTIVE",
    "prep": "ADPOSITION",
    "pron": "PRONOUN",
    "adv": "ADVERB",
    "conj": "CONJUNCTION",
    "part": "PARTICLE",
    "punc": "PUNCTUATION",
}


POS_UNIFIED = {
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJECTIVE": "ADJ",
    "ADPOSITION": "ADP",
    "PRONOUN": "PRON",
    "ADVERB": "ADV",
    "CONJUNCTION": "CCONJ",
    "PARTICLE": "PART",
    "PUNCTUATION": "PUNCT",
    "CONJ_SUB": "SCONJ",
}


WEAK_VERB_ROOTS = {
    "ق.قل": "ق.وَل",
    "بع": "ب.ي.ع",
    "نم": "ن.و.م",
    "صم": "ص.و.م",
    "خ.ف": "خ.و.ف",
    "زر": "ز.و.ر",
    "ط.ر": "ط.ي.ر",
    "سر": "س.ي.ر",
    "عد": "ع.و.د",
    "جلأ": "ج.ي.أ",
    "شعل": "ش.ي.أ",
    "كل": "أ.كل",
}

SINGLE_LETTER_PARTICLES = {
    "ب": {"root": "ب", "gloss": "with/by", "pos": "ADPOSITION"},
    "ل": {"root": "ل", "gloss": "to/for", "pos": "ADPOSITION"},
    "و": {"root": "و", "gloss": "and", "pos": "CONJUNCTION"},
    "ف": {"root": "ف", "gloss": "then/so", "pos": "CONJUNCTION"},
    "ك": {"root": "ك", "gloss": "like/as", "pos": "ADPOSITION"},
}

GLOSS_NOISE = {
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "i",
    "me",
    "you",
    "he",
    "him",
    "she",
    "it",
    "us",
    "them",
    "we",
    "the",
    "a",
    "an",
    "of",
    "for",
    "with",
    "that",
    "which",
    "who",
    "whose",
    "what",
    "defgen",
    "defnom",
    "defacc",
    "indef",
    "def",
    "one",
    "two",
    "three",
    "fempl",
    "mascpl",
    "femsg",
    "mascsg",
    "masc",
    "fem",
}


def map_pos(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    # pos from CAMeL is like noun/verb/adj...
    key = pos.strip().lower()
    mapped = POS_MAP.get(key)
    if not mapped:
        mapped = pos.upper()
    return POS_UNIFIED.get(mapped, mapped)


def clean_root(root: Optional[str]) -> Optional[str]:
    return root.replace("#.", "").replace(".#", "").strip() if root else None


def confidence_bucket(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def strip_diacritics(text: Optional[str]) -> str:
    if not text:
        return ""
    import re

    return re.sub(r"[\u064B-\u065F\u0670]", "", text)


def simplify_gloss(gloss: Optional[str]) -> Optional[str]:
    import re

    if not gloss:
        return None
    simplified = re.sub(r"[\[\]().;]", "", gloss.split(";")[0]).strip()
    simplified = simplified.replace("the+", "").replace("+", " ").replace("_", " ")
    words = simplified.split()
    clean = [w for w in words if w.lower() not in GLOSS_NOISE]
    result = " ".join(clean).strip()
    return result if result else None


def augment_root(root: str, lemma: str, pos: str, surface: str = "") -> tuple[str, str, Optional[str]]:
    if not root:
        return root, "unknown", None
    if surface in SINGLE_LETTER_PARTICLES:
        p = SINGLE_LETTER_PARTICLES[surface]
        return p["root"], "monoliteral", p["gloss"]
    parts = root.split(".")
    if len(parts) >= 3:
        return root, "triliteral", None
    if len(parts) == 2 and pos == "verb" and root in WEAK_VERB_ROOTS:
        return WEAK_VERB_ROOTS[root], "triliteral_weak", None
    if len(parts) == 2:
        return root, "biliteral", None
    if len(parts) == 1:
        return root, "monoliteral", None
    return root, "unknown", None


def correct_number(surface: str, number: str, segmentation: List[str], pos: str) -> tuple[str, bool]:
    if not number or pos != "NOUN":
        return number, False
    if (
        number == "dual"
        and surface.endswith("تي")
        and len(segmentation) >= 2
        and segmentation[-2] == "ت"
        and segmentation[-1] == "ي"
    ):
        return "singular", True
    return number, False


class CamelTool(Analyzer):
    tool_name = "camel"

    def __init__(self):
        self._db = MorphologyDB.builtin_db()
        self._disambiguator = MLEDisambiguator.pretrained()

    def analyze(self, text: str) -> Dict[str, Any]:
        t0 = time.time()
        tokens = simple_word_tokenize(normalize_whitespace(text))
        results = self._disambiguator.disambiguate(tokens)

        token_outputs: List[Dict[str, Any]] = []
        for token, disambig in zip(tokens, results):
            analyses = []
            segs = [token]
            for a in disambig.analyses[:3]:
                features = a.analysis
                score = round(float(a.score), 4)

                raw_root = clean_root(features.get("root"))
                raw_pos = features.get("pos")
                raw_lemma = features.get("lex")
                raw_gloss = features.get("gloss")

                aug_root, root_type, part_gloss = augment_root(
                    raw_root or "", raw_lemma or "", raw_pos or "", token
                )
                clean_gloss = part_gloss or simplify_gloss(raw_gloss)

                corrected_num, num_fixed = correct_number(
                    token,
                    NUMBER_MAP.get(features.get("num")),
                    segs,
                    map_pos(raw_pos),
                )

                analyses.append(
                    {
                        "surface": token,
                        "lemma": raw_lemma,
                        "root": aug_root,
                        "root_type": root_type,
                        "pos": map_pos(raw_pos),
                        "gender": GENDER_MAP.get(features.get("gen")),
                        "number": corrected_num,
                        "tense": ASPECT_MAP.get(features.get("asp")),
                        "gloss": clean_gloss,
                        "confidence": score,
                        "confidence_level": confidence_bucket(score),
                        "corrections": ["number"] if num_fixed else [],
                    }
                )

            token_outputs.append({"surface": token, "analyses": analyses, "segmentation": segs})

        return {
            "tool": "camel",
            "status": "ok",
            "input": text,
            "word_count": len(token_outputs),
            "tokens": token_outputs,
            "elapsed": time.time() - t0,
        }

