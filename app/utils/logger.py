import logging
import time
from typing import Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("arabic-nlp-platform")


def log_time(tool: str, text: str, elapsed: float, *, prefix: str = "") -> None:
    """Structured timing log used by tools."""
    snippet = (text or "").strip()
    logger.info(f"{prefix}[{tool.upper()}] '{snippet[:30]}' -> {elapsed:.3f}s")


def now_s() -> float:
    return time.time()

