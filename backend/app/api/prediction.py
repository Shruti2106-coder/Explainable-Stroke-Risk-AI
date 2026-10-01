from fastapi import APIRouter

from ..schemas.patient import PatientInput, PredictionResponse
from ..services.prediction_service import get_prediction_service


router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_risk(patient: PatientInput) -> PredictionResponse:
    result = get_prediction_service().predict(patient)
    return PredictionResponse.model_validate(result)