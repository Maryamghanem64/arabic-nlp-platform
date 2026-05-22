from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router
from app.api.fusion import router as fusion_router
from app.api.evaluate import router as evaluate_router
from app.api.ui import router as ui_router

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
app.include_router(ui_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

