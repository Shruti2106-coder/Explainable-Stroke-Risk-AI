from typing import Any

from fastapi import APIRouter

from ..schemas.patient import PatientInput
from ..services.shap_explanation import get_prediction_explanation


router = APIRouter(tags=["explainable-ai"])


@router.post("/explain")
def explain_prediction(patient: PatientInput) -> dict[str, Any]:
    return get_prediction_explanation(patient.model_dump())