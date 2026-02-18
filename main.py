# ============================================================
# Arabic NLP Comparative Platform (v5.2 - Final)
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import re

from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from farasa.segmenter import FarasaSegmenter

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Arabic NLP Comparative Platform",
    version="5.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load Resources
# ============================================================

logger.info("Loading NLP resources...")

db = MorphologyDB.builtin_db()
disambiguator = MLEDisambiguator.pretrained()
farasa_segmenter = FarasaSegmenter(interactive=False)

logger.info("Resources loaded successfully")

# ============================================================
# Maps
# ============================================================

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
    "punc": "PUNCTUATION"
}

# ============================================================
# Weak Verb Root Augmentation Table
# ============================================================

WEAK_VERB_ROOTS = {
    # قال - قول (to say)
    "ق.ل": "ق.و.ل",
    
    # باع - بيع (to sell)
    "ب.ع": "ب.ي.ع",
    
    # نام - نوم (to sleep)
    "ن.م": "ن.و.م",
    
    # صام - صوم (to fast)
    "ص.م": "ص.و.م",
    
    # خاف - خوف (to fear)
    "خ.ف": "خ.و.ف",
    
    # زار - زور (to visit)
    "ز.ر": "ز.و.ر",
    
    # طار - طير (to fly)
    "ط.ر": "ط.ي.ر",
    
    # سار - سير (to walk)
    "س.ر": "س.ي.ر",
    
    # عاد - عود (to return)
    "ع.د": "ع.و.د",
    
    # جاء - جيء (to come)
    "ج.ء": "ج.ي.ء",
    
    # شاء - شيء (to wish)
    "ش.ء": "ش.ي.ء",
}

# ============================================================
# Single-letter particles (common prepositions/conjunctions)
# ============================================================

SINGLE_LETTER_PARTICLES = {
    "ب": {"root": "ب", "gloss": "with/by", "pos": "ADPOSITION"},
    "ل": {"root": "ل", "gloss": "to/for", "pos": "ADPOSITION"},
    "و": {"root": "و", "gloss": "and", "pos": "CONJUNCTION"},
    "ف": {"root": "ف", "gloss": "then/so", "pos": "CONJUNCTION"},
    "ك": {"root": "ك", "gloss": "like/as", "pos": "ADPOSITION"},
}

# ============================================================
# Schemas
# ============================================================

class MorphAnalysis(BaseModel):
    lemma: Optional[str]
    root: Optional[str]
    root_type: Optional[str]
    part_of_speech: Optional[str]
    gender: Optional[str]
    number: Optional[str]
    tense: Optional[str]
    english_gloss: Optional[str]
    confidence_score: float
    confidence_level: str
    corrections: List[str]


class TokenOutput(BaseModel):
    surface_form: str
    morphology: List[MorphAnalysis]
    segmentation: List[str]
    analysis_count: int


class CombinedResponse(BaseModel):
    input_sentence: str
    word_count: int
    tokens: List[TokenOutput]

# ============================================================
# Helpers
# ============================================================

def map_pos(pos: Optional[str]) -> Optional[str]:
    """Map CAMeL POS to unified format"""
    if not pos:
        return None
    return POS_MAP.get(pos, pos.upper())


def clean_root(root: Optional[str]) -> Optional[str]:
    """Remove CAMeL internal markers"""
    if not root:
        return None
    return root.replace("#.", "").replace(".#", "").strip()


def confidence_bucket(score: float) -> str:
    """Categorize confidence score"""
    if score >= 0.9:
        return "high"
    elif score >= 0.6:
        return "medium"
    return "low"


def simplify_gloss(gloss: Optional[str]) -> Optional[str]:
    """Clean English gloss aggressively"""
    if not gloss:
        return None

    simplified = gloss.split(";")[0].strip()
    simplified = re.sub(r'\[.*?\]', '', simplified)
    simplified = re.sub(r'\(.*?\)', '', simplified)
    simplified = simplified.replace("the+", "")
    simplified = simplified.replace("+", " ")
    simplified = simplified.replace("_", " ")

    noise_words = {
        "my", "your", "his", "her", "its", "our", "their",
        "i", "me", "you", "he", "him", "she", "it", "us", "them", "we",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "the", "a", "an",
        "of", "for", "with",
        "that", "which", "who", "whose", "what"
    }

    words = simplified.split()
    
    # Keep single-word prepositions
    if len(words) == 1:
        return simplified.strip() if simplified.strip() else None
    
    clean_words = [w for w in words if w.lower() not in noise_words]
    result = " ".join(clean_words)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result if result else None

# ============================================================
# Post-Processing - Root Augmentation
# ============================================================

def augment_root(root: str, lemma: str, pos: str, surface: str = "") -> tuple[str, str, Optional[str]]:
    """
    Augment roots and handle special cases.
    Returns: (augmented_root, root_type, corrected_gloss)
    """
    if not root:
        return root, "unknown", None
    
    # Special case: single-letter particles
    if surface in SINGLE_LETTER_PARTICLES:
        particle = SINGLE_LETTER_PARTICLES[surface]
        return particle["root"], "monoliteral", particle["gloss"]
    
    parts = root.split(".")
    
    # Already triliteral+
    if len(parts) >= 3:
        return root, "triliteral", None
    
    # Biliteral verb → lookup table
    if len(parts) == 2 and pos == "verb":
        if root in WEAK_VERB_ROOTS:
            augmented = WEAK_VERB_ROOTS[root]
            logger.info(f"Root augmented: {root} → {augmented}")
            return augmented, "triliteral_weak", None
        else:
            return root, "biliteral_weak", None
    
    # Biliteral particle
    if len(parts) == 2:
        return root, "biliteral", None
    
    # Monoliteral
    if len(parts) == 1:
        return root, "monoliteral", None
    
    return root, "unknown", None

# ============================================================
# Post-Processing - Number Correction
# ============================================================

def correct_number(surface: str, number: str, segmentation: List[str], pos: str) -> tuple[str, bool]:
    """Fix dual misclassification for possessives"""
    if not number or pos != "NOUN":
        return number, False
    
    if number == "dual" and surface.endswith("تي"):
        if len(segmentation) >= 2 and segmentation[-2] == "ت" and segmentation[-1] == "ي":
            logger.info(f"Number corrected: {surface} dual → singular")
            return "singular", True
    
    return number, False

# ============================================================
# Core Logic
# ============================================================

def run_camel_analysis(text: str, segmentation_map: dict) -> List[List[MorphAnalysis]]:
    """Run CAMeL with post-processing"""
    tokens = simple_word_tokenize(text)
    disambig_results = disambiguator.disambiguate(tokens)

    all_results = []

    for token, disambig in zip(tokens, disambig_results):
        token_analyses = []
        token_segments = segmentation_map.get(token, [token])

        for analysis_obj in disambig.analyses[:2]:
            features = analysis_obj.analysis
            score = round(analysis_obj.score, 4)
            
            raw_root = clean_root(features.get("root"))
            raw_number = NUMBER_MAP.get(features.get("num"))
            raw_pos = features.get("pos")
            raw_lemma = features.get("lex")
            raw_gloss = features.get("gloss")
            
            corrections = []
            
            # Root augmentation (with particle handling)
            augmented_root, root_type, particle_gloss = augment_root(
                raw_root, 
                raw_lemma, 
                raw_pos,
                token
            )
            
            # Gloss handling
            if particle_gloss:
                # Use special particle gloss
                clean_gloss = particle_gloss
                corrections.append("gloss")
                if augmented_root != raw_root:
                    corrections.append("root")
            else:
                # Normal gloss cleaning
                clean_gloss = simplify_gloss(raw_gloss)
                if clean_gloss != raw_gloss:
                    corrections.append("gloss")
                if augmented_root != raw_root:
                    corrections.append("root")
            
            # Number correction
            corrected_number, num_fixed = correct_number(
                token, raw_number, token_segments, map_pos(raw_pos)
            )
            if num_fixed:
                corrections.append("number")

            token_analyses.append(
                MorphAnalysis(
                    lemma=raw_lemma,
                    root=augmented_root,
                    root_type=root_type,
                    part_of_speech=map_pos(raw_pos),
                    gender=GENDER_MAP.get(features.get("gen")),
                    number=corrected_number,
                    tense=ASPECT_MAP.get(features.get("asp")),
                    english_gloss=clean_gloss,
                    confidence_score=score,
                    confidence_level=confidence_bucket(score),
                    corrections=corrections
                )
            )

        all_results.append(token_analyses)

    return all_results


def run_farasa_segmentation(text: str) -> tuple[List[List[str]], dict]:
    """Run Farasa segmentation"""
    segmented_text = farasa_segmenter.segment(text)
    tokens = simple_word_tokenize(text)
    raw_segments = segmented_text.split()

    aligned_segments = []
    segment_map = {}

    for token, raw_seg in zip(tokens, raw_segments):
        parts = [p for p in raw_seg.split("+") if p]
        aligned_segments.append(parts if parts else [raw_seg])
        segment_map[token] = parts if parts else [raw_seg]

    return aligned_segments, segment_map

# ============================================================
# Endpoints
# ============================================================

@app.get("/analyze-combined", response_model=CombinedResponse)
def analyze_combined(text: str):
    """Combined analysis with intelligent post-processing"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    try:
        farasa_segments, segment_map = run_farasa_segmentation(text)
        camel_results = run_camel_analysis(text, segment_map)

        tokens_output = []
        surface_tokens = simple_word_tokenize(text)

        for surface, morph_list, segs in zip(surface_tokens, camel_results, farasa_segments):
            tokens_output.append(
                TokenOutput(
                    surface_form=surface,
                    morphology=morph_list,
                    segmentation=segs,
                    analysis_count=len(morph_list)
                )
            )

        return CombinedResponse(
            input_sentence=text,
            word_count=len(tokens_output),
            tokens=tokens_output
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {
        "message": "Arabic NLP Platform v5.2 - Production Ready",
        "features": [
            "Weak verb root augmentation via lookup table",
            "Single-letter particle handling",
            "Number correction for possessives",
            "Aggressive gloss cleaning",
            "Comprehensive correction tracking"
        ],
        "stats": {
            "weak_verbs_supported": len(WEAK_VERB_ROOTS),
            "particles_supported": len(SINGLE_LETTER_PARTICLES)
        }
    }