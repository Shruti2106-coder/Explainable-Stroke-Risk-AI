from fastapi import APIRouter

from ..services.prediction_service import is_prediction_service_loaded


router = APIRouter(tags=["system"])


@router.get("/health")
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": is_prediction_service_loaded(),
    }