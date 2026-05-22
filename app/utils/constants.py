from __future__ import annotations

from typing import Dict, List, Optional, Any

# --- POS + unified tags ---
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

# --- Future reference ---
SINATOOLS_POS_MAP = {}

# --- Qalsadi POS labels kept for future ---
QALSADI_POS_MAP = {
    "فعل": "VERB",
    "اسم": "NOUN",
    "صفة": "ADJ",
    "حرف": "PART",
    "ضمير": "PRON",
    "ظرف": "ADV",

    "اسم فاعل": "NOUN",
    "اسم مفعول": "NOUN",
    "مصدر": "NOUN",

    "STOPWORD": "STOP",
}

# --- Fusion / weighting ---
FUSION_WEIGHTS = {
    "lemma": {"camel": 3, "stanza": 1},
    "pos": {"camel": 2, "stanza": 2},
    "morphology": {"camel": 3, "stanza": 1},
    "segmentation": {"farasa": 3},
    "syntax": {"stanza": 3},
}

# --- Morphology maps ---
ASPECT_MAP = {"p": "past", "i": "present", "c": "imperative", "na": None}
GENDER_MAP = {"m": "masculine", "f": "feminine", "na": None}
NUMBER_MAP = {"s": "singular", "d": "dual", "p": "plural", "na": None}

# --- Root helpers ---
WEAK_VERB_ROOTS = {
    "ق.ل": "ق.و.ل",
    "ب.ع": "ب.ي.ع",
    "ن.م": "ن.و.م",
    "ص.م": "ص.و.م",
    "خ.ف": "خ.و.ف",
    "ز.ر": "ز.و.ر",
    "ط.ر": "ط.ي.ر",
    "س.ر": "س.ي.ر",
    "ع.د": "ع.و.د",
    "ج.ء": "ج.ي.ء",
    "ش.ء": "ش.ي.ء",
    "ك.ل": "أ.ك.ل",
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

KNOWN_FIXES: Dict[str, Dict[str, Any]] = {
    # keep map for future fixes
}


GOLD_DATASET = [
    {
        "text": "كتب الطالب الدرس",
        "gold": [
            {"word": "كتب", "pos": "VERB", "lemma": "كتب"},
            {"word": "الطالب", "pos": "NOUN", "lemma": "طالب"},
            {"word": "الدرس", "pos": "NOUN", "lemma": "درس"},
        ],
    },
    {
        "text": "ذهب محمد إلى المدرسة",
        "gold": [
            {"word": "ذهب", "pos": "VERB", "lemma": "ذهب"},
            {"word": "محمد", "pos": "NOUN", "lemma": "محمد"},
            {"word": "إلى", "pos": "ADP", "lemma": "إلى"},
            {"word": "المدرسة", "pos": "NOUN", "lemma": "مدرسة"},
        ],
    },
    {
        "text": "سيعمل العمال بكفاءة",
        "gold": [
            {"word": "سيعمل", "pos": "VERB", "lemma": "عمل"},
            {"word": "العمال", "pos": "NOUN", "lemma": "عامل"},
            {"word": "بكفاءة", "pos": "NOUN", "lemma": "كفاءة"},
        ],
    },
    {
        "text": "العين جميلة",
        "gold": [
            {"word": "العين", "pos": "NOUN", "lemma": "عين"},
            {"word": "جميلة", "pos": "ADJ", "lemma": "جميل"},
        ],
    },
    {
        "text": "قرأ الطلاب الكتب في المكتبة",
        "gold": [
            {"word": "قرأ", "pos": "VERB", "lemma": "قرأ"},
            {"word": "الطلاب", "pos": "NOUN", "lemma": "طالب"},
            {"word": "الكتب", "pos": "NOUN", "lemma": "كتاب"},
            {"word": "في", "pos": "ADP", "lemma": "في"},
            {"word": "المكتبة", "pos": "NOUN", "lemma": "مكتبة"},
        ],
    },
    {
        "text": "لم يذهب محمد إلى المدرسة أمس",
        "gold": [
            {"word": "لم", "pos": "PART", "lemma": "لم"},
            {"word": "يذهب", "pos": "VERB", "lemma": "ذهب"},
            {"word": "محمد", "pos": "NOUN", "lemma": "محمد"},
            {"word": "إلى", "pos": "ADP", "lemma": "إلى"},
            {"word": "المدرسة", "pos": "NOUN", "lemma": "مدرسة"},
            {"word": "أمس", "pos": "ADV", "lemma": "أمس"},
        ],
    },
    {
        "text": "باع التاجر بضاعته بسعر مرتفع",
        "gold": [
            {"word": "باع", "pos": "VERB", "lemma": "باع"},
            {"word": "التاجر", "pos": "NOUN", "lemma": "تاجر"},
            {"word": "بضاعته", "pos": "NOUN", "lemma": "بضاعة"},
            {"word": "بسعر", "pos": "NOUN", "lemma": "سعر"},
            {"word": "مرتفع", "pos": "ADJ", "lemma": "مرتفع"},
        ],
    },
    {
        "text": "إن البنات يأكلن المثلجات",
        "gold": [
            {"word": "إن", "pos": "PART", "lemma": "إن"},
            {"word": "البنات", "pos": "NOUN", "lemma": "بنت"},
            {"word": "يأكلن", "pos": "VERB", "lemma": "أكل"},
            {"word": "المثلجات", "pos": "NOUN", "lemma": "مثلجة"},
        ],
    },
    {
        "text": "الكتاب الذي قرأته مفيد جداً",
        "gold": [
            {"word": "الكتاب", "pos": "NOUN", "lemma": "كتاب"},
            {"word": "الذي", "pos": "PRON", "lemma": "الذي"},
            {"word": "قرأته", "pos": "VERB", "lemma": "قرأ"},
            {"word": "مفيد", "pos": "ADJ", "lemma": "مفيد"},
            {"word": "جداً", "pos": "ADV", "lemma": "جداً"},
        ],
    },
    {
        "text": "وجدت المعلمة طالبة مجتهدة في الفصل",
        "gold": [
            {"word": "وجدت", "pos": "VERB", "lemma": "وجد"},
            {"word": "المعلمة", "pos": "NOUN", "lemma": "معلمة"},
            {"word": "طالبة", "pos": "NOUN", "lemma": "طالب"},
            {"word": "مجتهدة", "pos": "ADJ", "lemma": "مجتهد"},
            {"word": "في", "pos": "ADP", "lemma": "في"},
            {"word": "الفصل", "pos": "NOUN", "lemma": "فصل"},
        ],
    },
]

