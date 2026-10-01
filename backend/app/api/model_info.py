from typing import Any

from fastapi import APIRouter

from ..services.prediction_service import get_prediction_service


router = APIRouter(tags=["model"])


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    return get_prediction_service().model_info()