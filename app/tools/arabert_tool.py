import threading
import logging

logger = logging.getLogger(__name__)

arabert_pipeline = None
arabert_loaded   = False
arabert_loading  = False
arabert_lock     = threading.Lock()

def load_arabert():
    global arabert_pipeline, arabert_loaded, arabert_loading

    if arabert_loaded:
        return True
    if arabert_loading:
        return False

    arabert_loading = True
    try:
        from transformers import pipeline
        arabert_pipeline = pipeline(
            "fill-mask",
            model="aubmindlab/bert-base-arabertv2",
            device=-1
        )
        arabert_loaded = True
        logger.info("✅ AraBERT lazy-loaded")
        return True
    except Exception as e:
        arabert_pipeline = None
        arabert_loaded   = False
        logger.warning(f"⚠️ AraBERT failed: {e}")
        return False
    finally:
        arabert_loading = False

def get_arabert_status():
    if arabert_loaded:   return "ok"
    if arabert_loading:  return "loading"
    return "lazy"

def arabert_analyze(text: str) -> dict:
    global arabert_pipeline

    # Lazy load on first request
    if arabert_pipeline is None:
        loaded = load_arabert()
        if not loaded or arabert_pipeline is None:
            return {
                "tool":       "arabert",
                "status":     "unavailable",
                "reason":     "AraBERT model not loaded. First request triggers download (~700MB).",
                "input":      text,
                "word_count": 0,
                "tokens":     []
            }

    try:
        from camel_tools.tokenizers.word import simple_word_tokenize
        tokens_text = simple_word_tokenize(text)
        tokens = []

        with arabert_lock:
            for word in tokens_text:
                # Use fill-mask to get confidence of this word
                masked = text.replace(word, "[MASK]", 1)
                try:
                    results = arabert_pipeline(masked, top_k=1)
                    confidence = results[0]["score"] if results else 0.0
                except Exception:
                    confidence = 0.0

                tokens.append({
                    "surface":    word,
                    "lemma":      None,
                    "pos":        None,
                    "root":       None,
                    "gloss":      None,
                    "confidence": round(confidence, 4),
                })

        return {
            "tool":       "arabert",
            "status":     "ok",
            "approach":   "contextual fill-mask (BERT)",
            "input":      text,
            "word_count": len(tokens),
            "tokens":     tokens,
            "lemmas":     [],
            "pos":        [],
            "reason":     "",
        }
    except Exception as e:
        logger.error(f"[ARABERT] error: {e}")
        return {
            "tool":       "arabert",
            "status":     "error",
            "reason":     str(e),
            "input":      text,
            "word_count": 0,
            "tokens":     []
        }