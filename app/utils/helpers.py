from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.utils.constants import (
    GLOSS_NOISE,
    KNOWN_FIXES,
    SINGLE_LETTER_PARTICLES,
    POS_MAP,
    POS_UNIFIED,
    WEAK_VERB_ROOTS,
)


def map_pos(pos: Optional[str]) -> Optional[str]:
    return POS_MAP.get(pos, pos.upper()) if pos else None


def normalize_pos_for_compare(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    return POS_UNIFIED.get(pos.upper(), pos.upper())


def clean_root(root: Optional[str]) -> Optional[str]:
    return root.replace("#.", "").replace(".#", "").strip() if root else None


def strip_diacritics(text: Optional[str]) -> str:
    if not text:
        return ""
    # Arabic diacritics + tatweel
    return re.sub(r"[\u064B-\u065F\u0670]", "", text)


def simplify_gloss(gloss: Optional[str]) -> Optional[str]:
    if not gloss:
        return None
    simplified = re.sub(r"[\[\]().;]", "", gloss.split(";")[0]).strip()
    simplified = simplified.replace("the+", "").replace("+", " ").replace("_", " ")
    words = simplified.split()
    clean = [w for w in words if w.lower() not in GLOSS_NOISE]
    result = " ".join(clean).strip()
    return result if result else None


def confidence_bucket(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def augment_root(root: str, lemma: str, pos: str, surface: str = "") -> Tuple[str, str, Optional[str]]:
    if not root:
        return root, "unknown", None

    if surface in SINGLE_LETTER_PARTICLES:
        p = SINGLE_LETTER_PARTICLES[surface]
        return p["root"], "monoliteral", p["gloss"]

    parts = root.split(".")
    if len(parts) >= 3:
        return root, "triliteral", None

    if len(parts) == 2 and pos == "verb" and root in WEAK_VERB_ROOTS:
        aug = WEAK_VERB_ROOTS[root]
        return aug, "triliteral_weak", None

    if len(parts) == 2:
        return root, "biliteral", None

    if len(parts) == 1:
        return root, "monoliteral", None

    return root, "unknown", None


def correct_number(surface: str, number: str, segmentation: List[str], pos: str) -> Tuple[str, bool]:
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


def parse_feats(feats: Optional[str]) -> Dict[str, str]:
    if not feats:
        return {}

    result: Dict[str, str] = {}
    for pair in feats.split("|"):
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.lower()
        val_lower = val.lower()

        if "gender" in key:
            val = "masc" if "masc" in val_lower else "fem" if "fem" in val_lower else val_lower
        elif "number" in key:
            val = "sing" if "sing" in val_lower else "dual" if "dual" in val_lower else "plur" if "plur" in val_lower else val_lower
        elif "aspect" in key:
            val = "perf" if "perf" in val_lower else "impf" if "impf" in val_lower else val_lower
            result["tense"] = val
        elif "voice" in key:
            val = "act" if "act" in val_lower else "pass" if "pass" in val_lower else val_lower
        elif "case" in key:
            val = val_lower[:3]
        elif "definite" in key:
            val = "yes" if val_lower in ("def", "yes") else "no"
        else:
            val = val_lower

        result[key] = val

    return result


def classify_conflict(feature: str, val_a: Any, val_b: Any) -> Dict[str, str]:
    severity_map = {
        "pos": "high",
        "lemma": "medium",
        "root": "medium",
        "tense": "low",
        "gender": "low",
        "number": "low",
    }
    return {
        "feature": feature,
        "tool_a": str(val_a),
        "tool_b": str(val_b),
        "severity": severity_map.get(feature, "low"),
        "type": f"{feature}_mismatch",
    }

