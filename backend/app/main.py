import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.explain import router as explain_router
from .api.explanations import router as explanation_router
from .api.health import router as health_router
from .api.model_info import router as model_info_router
from .api.prediction import router as prediction_router
from .services.prediction_service import get_prediction_service


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_prediction_service()
    yield


def _frontend_origins() -> list[str]:
    configured = os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        origin if origin.startswith(("http://", "https://")) else f"https://{origin}"
        for origin in origins
    ]

app = FastAPI(
    title="Stroke Risk Assessment API",
    description=(
        "Educational stroke-risk model API with SHAP explanations. Outputs are not "
        "medical diagnoses or clinical advice."
    ),
    version="0.8.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(model_info_router)
app.include_router(explain_router)
app.include_router(explanation_router)
