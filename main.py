from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import re
import os

from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from farasa.segmenter import FarasaSegmenter
import stanza

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App
app = FastAPI(title="Arabic NLP Platform", version="6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Resources with Graceful Fallbacks
logger.info("Loading NLP resources...")

camel_db = None
camel_disambiguator = None
try:
    camel_db = MorphologyDB.builtin_db()
    camel_disambiguator = MLEDisambiguator.pretrained()
    logger.info("✅ CAMeL Tools loaded")
except Exception as e:
    logger.error(f"❌ CAMeL failed: {e}")

farasa_segmenter = None
try:
   farasa_segmenter = FarasaSegmenter(interactive=False)
   logger.info("✅ Farasa loaded from ./Farasa_bin")
except Exception as e:
    logger.error(f"❌ Farasa failed: {e}")
    
stanza_pipeline = None
try:
    stanza_pipeline = stanza.Pipeline(
        'ar',
        processors='tokenize,mwt,pos,lemma,depparse',
        verbose=False
    )
    logger.info("✅ Stanza loaded")
except Exception as e:
    logger.error(f"❌ Stanza failed: {e}")
    logger.info("Resource loading complete")

# Post-Processing Tables (Exact from Original)
WEAK_VERB_ROOTS = {
    "ق.ل": "ق.و.ل", "ب.ع": "ب.ي.ع", "ن.م": "ن.و.م", "ص.م": "ص.و.م",
    "خ.ف": "خ.و.ف", "ز.ر": "ز.و.ر", "ط.ر": "ط.ي.ر", "س.ر": "س.ي.ر",
    "ع.د": "ع.و.د", "ج.ء": "ج.ي.ء", "ش.ء": "ش.ي.ء"
}

SINGLE_LETTER_PARTICLES = {
    "ب": {"root": "ب", "gloss": "with/by", "pos": "ADPOSITION"},
    "ل": {"root": "ل", "gloss": "to/for", "pos": "ADPOSITION"},
    "و": {"root": "و", "gloss": "and", "pos": "CONJUNCTION"},
    "ف": {"root": "ف", "gloss": "then/so", "pos": "CONJUNCTION"},
    "ك": {"root": "ك", "gloss": "like/as", "pos": "ADPOSITION"},
}

ASPECT_MAP = {"p": "past", "i": "present", "c": "imperative", "na": None}
GENDER_MAP = {"m": "masculine", "f": "feminine", "na": None}
NUMBER_MAP = {"s": "singular", "d": "dual", "p": "plural", "na": None}
POS_MAP = {
    "noun": "NOUN", "verb": "VERB", "adj": "ADJECTIVE", "prep": "ADPOSITION",
    "pron": "PRONOUN", "adv": "ADVERB", "conj": "CONJUNCTION",
    "part": "PARTICLE", "punc": "PUNCTUATION"
}

# Post-Processing Functions (Exact Logic)
def map_pos(pos: Optional[str]) -> Optional[str]:
    return POS_MAP.get(pos, pos.upper()) if pos else None

def clean_root(root: Optional[str]) -> Optional[str]:
    return root.replace("#.", "").replace(".#", "").strip() if root else None

def confidence_bucket(score: float) -> str:
    if score >= 0.9: return "high"
    elif score >= 0.6: return "medium"
    return "low"

def simplify_gloss(gloss: Optional[str]) -> Optional[str]:
    if not gloss: return None
    simplified = re.sub(r'[\[\]().;]', '', gloss.split(";")[0]).strip()
    simplified = simplified.replace("the+", "").replace("+", " ").replace("_", " ")
    words = simplified.split()
    noise = {"my", "your", "his", "her", "its", "our", "their", "i", "me", "you", "he", "him", "she", "it", "us", "them", "we", "the", "a", "an", "of", "for", "with", "that", "which", "who", "whose", "what"}
    clean_words = [w for w in words if w.lower() not in noise]
    result = " ".join(clean_words).strip()
    return result if result else None

def augment_root(root: str, lemma: str, pos: str, surface: str = "") -> tuple[str, str, Optional[str]]:
    if not root: return root, "unknown", None
    if surface in SINGLE_LETTER_PARTICLES:
        p = SINGLE_LETTER_PARTICLES[surface]
        return p["root"], "monoliteral", p["gloss"]
    parts = root.split(".")
    if len(parts) >= 3: return root, "triliteral", None
    if len(parts) == 2 and pos == "verb" and root in WEAK_VERB_ROOTS:
        return WEAK_VERB_ROOTS[root], "triliteral_weak", None
    if len(parts) == 2: return root, "biliteral", None
    if len(parts) == 1: return root, "monoliteral", None
    return root, "unknown", None

def correct_number(surface: str, number: str, segmentation: List[str], pos: str) -> tuple[str, bool]:
    if not number or pos != "NOUN": return number, False
    if number == "dual" and surface.endswith("تي") and len(segmentation) >= 2 and segmentation[-2] == "ت" and segmentation[-1] == "ي":
        return "singular", True
    return number, False

# Pydantic Models
class MorphAnalysis(BaseModel):
    lemma: Optional[str]
    root: Optional[str]
    root_type: Optional[str]
    pos: Optional[str]
    gender: Optional[str]
    number: Optional[str]
    tense: Optional[str]
    gloss: Optional[str]
    confidence: float
    confidence_level: str
    corrections: List[str]

class TokenOutput(BaseModel):
    surface: str
    analyses: List[MorphAnalysis]
    segmentation: List[str]

class AnalysisResult(BaseModel):
    tool: str
    tokens: List[TokenOutput]
    status: str

class CompareResult(BaseModel):
    text: str
    results: Dict[str, Dict[str, Any]]

class Status(BaseModel):
    camel: Dict[str, Any]
    farasa: Dict[str, Any]
    stanza: Dict[str, Any]

# Tool Functions
def camel_analyze(text: str) -> Dict[str, Any]:
    if not camel_disambiguator or not camel_db:
        return {"status": "failed", "error": "CAMeL not loaded", "tokens": []}
    try:
        tokens = simple_word_tokenize(text)
        results = camel_disambiguator.disambiguate(tokens)
        token_outputs = []
        for token, disambig in zip(tokens, results):
            analyses = []
            segs = [token]
            for a in disambig.analyses[:3]:
                features = a.analysis
                score = round(a.score, 4)
                raw_root = clean_root(features.get("root"))
                raw_pos = features.get("pos")
                raw_lemma = features.get("lex")
                raw_gloss = features.get("gloss")
                aug_root, root_type, part_gloss = augment_root(raw_root or "", raw_lemma or "", raw_pos or "", token)
                clean_gloss = part_gloss or simplify_gloss(raw_gloss)
                corrections = []
                if aug_root != raw_root: corrections.append("root")
                if clean_gloss != raw_gloss: corrections.append("gloss")
                corrected_num, num_fixed = correct_number(token, NUMBER_MAP.get(features.get("num")), segs, map_pos(raw_pos))
                if num_fixed: corrections.append("number")
                analyses.append(MorphAnalysis(
                    lemma=raw_lemma, root=aug_root, root_type=root_type, pos=map_pos(raw_pos),
                    gender=GENDER_MAP.get(features.get("gen")), number=corrected_num,
                    tense=ASPECT_MAP.get(features.get("asp")), gloss=clean_gloss,
                    confidence=score, confidence_level=confidence_bucket(score), corrections=corrections
                ))
            token_outputs.append(TokenOutput(surface=token, analyses=analyses, segmentation=segs))
        return {"status": "ok", "tokens": token_outputs}
    except Exception as e:
        return {"status": "error", "error": str(e), "tokens": []}

def farasa_analyze(text: str) -> Dict[str, Any]:
    if not farasa_segmenter:
        return {"status": "failed", "error": "Farasa not loaded", "tokens": []}
    try:
        segmented = farasa_segmenter.segment(text)
        raw_tokens = simple_word_tokenize(text)
        raw_segs = segmented.split()
        token_outputs = []
        for token, seg in zip(raw_tokens, raw_segs):
            parts = seg.split('+')
            token_outputs.append(TokenOutput(surface=token, analyses=[], segmentation=parts))
        return {"status": "ok", "tokens": token_outputs, "segmented_text": segmented}
    except Exception as e:
        return {"status": "error", "error": str(e), "tokens": []}

def stanza_analyze(text: str) -> Dict[str, Any]:
    if not stanza_pipeline:
        return {"status": "failed", "error": "Stanza not loaded", "tokens": []}
    try:
        doc = stanza_pipeline(text)
        token_outputs = []
        for sentence in doc.sentences:
            for word in sentence.words:
                analyses = [MorphAnalysis(
    lemma=word.lemma,
    root=None,
    root_type=None,
    pos=word.xpos or word.upos,
    gender=None,
    number=None,
    tense=None,
    gloss=word.feats,
    confidence=1.0,
    confidence_level="high",
    corrections=[]
)]
                token_outputs.append(TokenOutput(surface=word.text, analyses=analyses, segmentation=[word.text]))
                logger.info(f"Stanza: {word.text} -> lemma:{word.lemma} pos:{word.upos} head:{word.head} deprel:{word.deprel}")
        return {"status": "ok", "tokens": token_outputs}
    except Exception as e:
        return {"status": "error", "error": str(e), "tokens": []}

# Endpoints
@app.get("/", response_model=Status)
def root():
    return Status(
        camel={"status": "ok" if camel_disambiguator else "failed"},
        farasa={"status": "ok" if farasa_segmenter else "failed"},
        stanza={"status": "ok" if stanza_pipeline else "failed"}
    )

@app.get("/analyze-combined", response_model=Dict[str, Any])
def analyze_combined(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    camel_res = camel_analyze(text)
    farasa_res = farasa_analyze(text)
    return {"camel": camel_res, "farasa": farasa_res}

@app.get("/analyze-stanza", response_model=Dict[str, Any])
def analyze_stanza_endpoint(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    return stanza_analyze(text)

@app.get("/compare", response_model=Dict[str, Any])
def compare(
    text: str,
    tools: str = Query("camel,farasa,stanza")
):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    tool_list = [t.strip() for t in tools.split(",")]
    results = {}
    if "camel" in tool_list:
        results["camel"] = camel_analyze(text)
    if "farasa" in tool_list:
        results["farasa"] = farasa_analyze(text)
    if "stanza" in tool_list:
        results["stanza"] = stanza_analyze(text)
    return {"text": text, "tools": tool_list, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
