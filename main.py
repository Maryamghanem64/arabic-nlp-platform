# ============================================================
# Arabic NLP Comparative Platform (v7.0)
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import re

from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from farasa.segmenter import FarasaSegmenter
import stanza

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
    version="7.0",
    description="Compare Arabic NLP tools: CAMeL Tools, Farasa, Stanza"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load Resources
# ============================================================

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
    logger.info("✅ Farasa loaded")
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

# ============================================================
# Maps
# ============================================================

ASPECT_MAP = {"p": "past", "i": "present", "c": "imperative", "na": None}
GENDER_MAP = {"m": "masculine", "f": "feminine", "na": None}
NUMBER_MAP = {"s": "singular", "d": "dual", "p": "plural", "na": None}

POS_MAP = {
    "noun": "NOUN", "verb": "VERB", "adj": "ADJECTIVE",
    "prep": "ADPOSITION", "pron": "PRONOUN", "adv": "ADVERB",
    "conj": "CONJUNCTION", "part": "PARTICLE", "punc": "PUNCTUATION"
}

WEAK_VERB_ROOTS = {
    "ق.ل": "ق.و.ل", "ب.ع": "ب.ي.ع", "ن.م": "ن.و.م", "ص.م": "ص.و.م",
    "خ.ف": "خ.و.ف", "ز.ر": "ز.و.ر", "ط.ر": "ط.ي.ر", "س.ر": "س.ي.ر",
    "ع.د": "ع.و.د", "ج.ء": "ج.ي.ء", "ش.ء": "ش.ي.ء"
}

SINGLE_LETTER_PARTICLES = {
    "ب": {"root": "ب", "gloss": "with/by",  "pos": "ADPOSITION"},
    "ل": {"root": "ل", "gloss": "to/for",   "pos": "ADPOSITION"},
    "و": {"root": "و", "gloss": "and",       "pos": "CONJUNCTION"},
    "ف": {"root": "ف", "gloss": "then/so",  "pos": "CONJUNCTION"},
    "ك": {"root": "ك", "gloss": "like/as",  "pos": "ADPOSITION"},
}

# ============================================================
# Schemas
# ============================================================

class MorphAnalysis(BaseModel):
    lemma:            Optional[str] = None
    root:             Optional[str] = None
    root_type:        Optional[str] = None
    pos:              Optional[str] = None
    gender:           Optional[str] = None
    number:           Optional[str] = None
    tense:            Optional[str] = None
    gloss:            Optional[str] = None
    confidence:       float = 0.0
    confidence_level: str = "low"
    corrections:      List[str] = []

class TokenOutput(BaseModel):
    surface:    str
    analyses:   List[MorphAnalysis] = []
    segmentation: List[str] = []

class Dependency(BaseModel):
    head:      Optional[int] = None
    head_text: Optional[str] = None
    deprel:    Optional[str] = None

class StanzaToken(BaseModel):
    surface:    str
    lemma:      Optional[str] = None
    upos:       Optional[str] = None
    xpos:       Optional[str] = None
    gender:     Optional[str] = None
    number:     Optional[str] = None
    tense:      Optional[str] = None
    person:     Optional[str] = None
    voice:      Optional[str] = None
    case:       Optional[str] = None
    definite:   Optional[str] = None
    aspect:     Optional[str] = None
    dependency: Optional[Dependency] = None

class StanzaResponse(BaseModel):
    tool:       str = "stanza"
    status:     str = "ok"
    input:      str
    word_count: int
    tokens:     List[StanzaToken]

class CombinedResponse(BaseModel):
    tool:       str = "camel+farasa"
    status:     str = "ok"
    input:      str
    word_count: int
    tokens:     List[TokenOutput]

# ============================================================
# Helpers
# ============================================================

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
    noise = {
        "my","your","his","her","its","our","their",
        "i","me","you","he","him","she","it","us","them","we",
        "the","a","an","of","for","with","that","which","who","whose","what"
    }
    words = simplified.split()
    clean_words = [w for w in words if w.lower() not in noise]
    result = " ".join(clean_words).strip()
    return result if result else None

def augment_root(root: str, lemma: str, pos: str, surface: str = "") -> tuple:
    if not root: return root, "unknown", None
    if surface in SINGLE_LETTER_PARTICLES:
        p = SINGLE_LETTER_PARTICLES[surface]
        return p["root"], "monoliteral", p["gloss"]
    parts = root.split(".")
    if len(parts) >= 3: return root, "triliteral", None
    if len(parts) == 2 and pos == "verb" and root in WEAK_VERB_ROOTS:
        augmented = WEAK_VERB_ROOTS[root]
        logger.info(f"Root augmented: {root} → {augmented}")
        return augmented, "triliteral_weak", None
    if len(parts) == 2: return root, "biliteral", None
    if len(parts) == 1: return root, "monoliteral", None
    return root, "unknown", None

def correct_number(surface: str, number: str, segmentation: List[str], pos: str) -> tuple:
    if not number or pos != "NOUN": return number, False
    if (number == "dual" and surface.endswith("تي") and
            len(segmentation) >= 2 and
            segmentation[-2] == "ت" and segmentation[-1] == "ي"):
        logger.info(f"Number corrected: {surface} dual → singular")
        return "singular", True
    return number, False

def parse_feats(feats: Optional[str]) -> Dict[str, str]:
    if not feats: return {}
    result = {}
    for pair in feats.split("|"):
        if "=" in pair:
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

# ============================================================
# Tool Functions
# ============================================================

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
                features    = a.analysis
                score       = round(a.score, 4)
                raw_root    = clean_root(features.get("root"))
                raw_pos     = features.get("pos")
                raw_lemma   = features.get("lex")
                raw_gloss   = features.get("gloss")
                aug_root, root_type, part_gloss = augment_root(
                    raw_root or "", raw_lemma or "", raw_pos or "", token
                )
                clean_gloss = part_gloss or simplify_gloss(raw_gloss)
                corrections = []
                if aug_root != raw_root: corrections.append("root")
                if clean_gloss != raw_gloss: corrections.append("gloss")
                corrected_num, num_fixed = correct_number(
                    token, NUMBER_MAP.get(features.get("num")), segs, map_pos(raw_pos)
                )
                if num_fixed: corrections.append("number")
                analyses.append(MorphAnalysis(
                    lemma=raw_lemma, root=aug_root, root_type=root_type,
                    pos=map_pos(raw_pos),
                    gender=GENDER_MAP.get(features.get("gen")),
                    number=corrected_num,
                    tense=ASPECT_MAP.get(features.get("asp")),
                    gloss=clean_gloss,
                    confidence=score,
                    confidence_level=confidence_bucket(score),
                    corrections=corrections
                ))
            token_outputs.append(TokenOutput(surface=token, analyses=analyses, segmentation=segs))
        return {"tool": "camel", "status": "ok", "input": text,
                "word_count": len(token_outputs), "tokens": token_outputs}
    except Exception as e:
        return {"tool": "camel", "status": "error", "error": str(e), "tokens": []}

def farasa_analyze(text: str) -> Dict[str, Any]:
    if not farasa_segmenter:
        return {"tool": "farasa", "status": "failed", "error": "Farasa not loaded", "tokens": []}
    try:
        segmented   = farasa_segmenter.segment(text)
        raw_tokens  = simple_word_tokenize(text)
        raw_segs    = segmented.split()
        token_outputs = []
        for token, seg in zip(raw_tokens, raw_segs):
            parts = [p for p in seg.split("+") if p]
            token_outputs.append(TokenOutput(surface=token, analyses=[], segmentation=parts))
        return {"tool": "farasa", "status": "ok", "input": text,
                "word_count": len(token_outputs),
                "segmented_text": segmented, "tokens": token_outputs}
    except Exception as e:
        return {"tool": "farasa", "status": "error", "error": str(e), "tokens": []}

def stanza_analyze(text: str) -> Dict[str, Any]:
    if not stanza_pipeline:
        return {"tool": "stanza", "status": "failed", "error": "Stanza not loaded", "tokens": []}
    try:
        doc    = stanza_pipeline(text)
        tokens = []
        for sentence in doc.sentences:
            for word in sentence.words:
                feats     = parse_feats(word.feats)
                head      = int(word.head) if word.head and str(word.head) != "0" else None
                head_text = None
                if head and 1 <= head <= len(sentence.words):
                    head_text = sentence.words[head - 1].text
                elif str(word.head) == "0":
                    head_text = "root"
                tokens.append(StanzaToken(
                    surface=word.text, lemma=word.lemma,
                    upos=word.upos, xpos=word.xpos,
                    gender=feats.get("gender"),
                    number=feats.get("number"),
                    tense=feats.get("tense"),
                    person=feats.get("person"),
                    voice=feats.get("voice"),
                    case=feats.get("case"),
                    definite=feats.get("definite"),
                    aspect=feats.get("aspect"),
                    dependency=Dependency(head=head, head_text=head_text, deprel=word.deprel)
                ))
        return {"tool": "stanza", "status": "ok", "input": text,
                "word_count": len(tokens), "tokens": tokens}
    except Exception as e:
        return {"tool": "stanza", "status": "error", "error": str(e), "tokens": []}

# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "platform": "Arabic NLP Comparative Platform",
        "version":  "7.0",
        "tools": {
            "camel": {
                "status":       "ok" if camel_disambiguator else "failed",
                "capabilities": ["lemma","root","pos","gender","number","tense","gloss","confidence"]
            },
            "farasa": {
                "status":       "ok" if farasa_segmenter else "failed",
                "capabilities": ["segmentation"]
            },
            "stanza": {
                "status":       "ok" if stanza_pipeline else "failed",
                "capabilities": ["lemma","upos","xpos","gender","number","tense","case","definite","dependency"]
            },
        },
        "endpoints": [
            "GET /analyze/camel?text=...",
            "GET /analyze/farasa?text=...",
            "GET /analyze/stanza?text=...",
            "GET /analyze-combined?text=...",
            "GET /compare?text=...&tools=camel,farasa,stanza"
        ]
    }

@app.get("/analyze/{tool}")
def analyze_by_tool(tool: str, text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    if tool == "camel":
        return camel_analyze(text)
    elif tool == "farasa":
        return farasa_analyze(text)
    elif tool == "stanza":
        return stanza_analyze(text)
    else:
        raise HTTPException(404, f"Tool '{tool}' not found. Available: camel, farasa, stanza")
@app.get("/analyze-combined")
def analyze_combined(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    return {
        "input":  text,
        "camel":  camel_analyze(text),
        "farasa": farasa_analyze(text),
        "stanza": stanza_analyze(text),
    }

@app.get("/compare")
def compare(
    text:  str,
    tools: str = Query("camel,farasa,stanza")
):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    tool_list = [t.strip() for t in tools.split(",")]
    results   = {}
    if "camel"   in tool_list: results["camel"]   = camel_analyze(text)
    if "farasa"  in tool_list: results["farasa"]  = farasa_analyze(text)
    if "stanza"  in tool_list: results["stanza"]  = stanza_analyze(text)
    return {
        "input": text,
        "tools": tool_list,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)