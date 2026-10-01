"""API routes for model-level and single-patient SHAP explanations."""

from typing import Any

from fastapi import APIRouter, HTTPException

from ..schemas.patient import PatientInput
from ..services.shap_explanation import (
    get_global_explanation,
    get_prediction_explanation,
)


router = APIRouter(prefix="/explanations", tags=["explainable-ai"])


@router.get("/global")
def global_explanation() -> dict[str, Any]:
    """Return global SHAP rankings and paths to saved figures/data."""
    try:
        return get_global_explanation()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/prediction")
def prediction_explanation(patient_data: PatientInput) -> dict[str, Any]:
    """Return a patient's predicted class, class probabilities, and local SHAP values."""
    try:
        return get_prediction_explanation(patient_data.model_dump())
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error