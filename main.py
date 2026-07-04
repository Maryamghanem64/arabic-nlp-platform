# Version: 8.3.1
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router
from app.api.compare import router as compare_router
from app.api.evaluate import router as evaluate_router
from app.api.fusion import router as fusion_router
from app.api.ui import router as ui_router
from app.api.tools_sinatools import router as tools_sinatools_router

from app.core.startup import start_heavy_tool_warmup
from app.core.tool_registry import log_startup_report

from app.tools.alkhalil_tool import load_alkhalil
from app.tools.farasa_tool import load_farasa
from app.tools.udpipe_tool import load_udpipe

app = FastAPI(title="Arabic NLP Comparative Platform", version="8.3")


def _cors_origins() -> list[str]:
    configured = os.environ.get("ARABIC_NLP_CORS_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(fusion_router)
app.include_router(evaluate_router)
app.include_router(compare_router)
app.include_router(ui_router)
app.include_router(tools_sinatools_router)



@app.on_event("startup")
def validate_tools_on_startup():
    # Load optional heavy models best-effort â€” must never crash.
    # AraBERT is LAZY LOADED on first request (do not load here).
    preload_core_enabled = os.environ.get("ARABIC_NLP_PRELOAD_CORE", "false").lower() in {"1", "true", "yes", "on"}
    if preload_core_enabled:
        try:
            load_farasa()
        except Exception:
            pass
        try:
            load_alkhalil()
        except Exception:
            pass
        try:
            load_udpipe()
        except Exception:
            pass
    startup_report_enabled = os.environ.get("ARABIC_NLP_STARTUP_REPORT", "false").lower() in {"1", "true", "yes", "on"}
    if startup_report_enabled:
        log_startup_report()
    warmup_enabled = os.environ.get("ARABIC_NLP_STARTUP_WARMUP", "false").lower() in {"1", "true", "yes", "on"}
    warmup_disabled = os.environ.get("ARABIC_NLP_DISABLE_STARTUP_WARMUP", "").lower() in {"1", "true", "yes", "on"}
    if warmup_enabled and not warmup_disabled:
        start_heavy_tool_warmup()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
