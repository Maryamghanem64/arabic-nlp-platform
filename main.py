# Version: 8.3.1
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router
from app.api.compare import router as compare_router
from app.api.evaluate import router as evaluate_router
from app.api.fusion import router as fusion_router
from app.api.ui import router as ui_router
from app.core.tool_registry import log_startup_report

from app.tools.alkhalil_tool import load_alkhalil
from app.tools.arabert_tool import arabert_analyze, get_arabert_status
from app.tools.farasa_tool import load_farasa
from app.tools.madamira_tool import load_madamira
from app.tools.udpipe_tool import load_udpipe

app = FastAPI(title="Arabic NLP Comparative Platform", version="8.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(fusion_router)
app.include_router(evaluate_router)
app.include_router(compare_router)
app.include_router(ui_router)


@app.on_event("startup")
def validate_tools_on_startup():
    # Load optional heavy models best-effort â€” must never crash.
    # AraBERT is LAZY LOADED on first request (do not load here).
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
    try:
        load_madamira()
    except Exception:
        pass

    log_startup_report()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

