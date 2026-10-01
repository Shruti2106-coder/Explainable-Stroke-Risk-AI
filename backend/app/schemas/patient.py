"""Pydantic request schemas derived from the approved feature configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURE_CONFIG_PATH = PROJECT_ROOT / "models" / "final_feature_config.json"

# These are the observed prepared-dataset bounds, not clinical reference ranges.
OBSERVED_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "Age": (18.0, 90.0),
    "Height_cm": (145.0, 195.0),
    "Weight_kg": (45.0, 130.0),
    "BMI": (11.9, 61.5),
    "Blood_Pressure_Systolic": (100.0, 170.0),
    "Blood_Pressure_Diastolic": (60.0, 110.0),
    "Heart_Rate": (55.0, 110.0),
    "Blood_Glucose": (70.0, 220.0),
    "HbA1c": (4.5, 10.5),
    "Total_Cholesterol": (130.0, 320.0),
    "HDL": (25.0, 90.0),
    "LDL": (60.0, 220.0),
    "Triglycerides": (60.0, 350.0),
    "Sleep_Hours": (4.0, 10.0),
    "Exercise_Hours_Per_Week": (0.0, 10.0),
    "Daily_Walking_Minutes": (0.0, 120.0),
}
OPTIONAL_FEATURES = {
    "Age",
    "Height_cm",
    "Weight_kg",
    "Blood_Glucose",
    "HbA1c",
    "Total_Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "Sleep_Hours",
    "Medication_Adherence",
    "Exercise_Hours_Per_Week",
    "Daily_Walking_Minutes",
}


class PatientInputBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_blood_pressure_order(self) -> PatientInputBase:
        systolic = getattr(self, "Blood_Pressure_Systolic", None)
        diastolic = getattr(self, "Blood_Pressure_Diastolic", None)
        if systolic is not None and diastolic is not None and systolic <= diastolic:
            raise ValueError(
                "Blood_Pressure_Systolic must be greater than Blood_Pressure_Diastolic."
            )
        return self


def _load_feature_config() -> dict[str, Any]:
    if not FEATURE_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Final feature configuration not found: {FEATURE_CONFIG_PATH}")
    return json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))


def _build_patient_input_model() -> type[BaseModel]:
    config = _load_feature_config()
    fields: dict[str, tuple[Any, Any]] = {}
    for feature in config["features_safe_for_prediction"]:
        if feature in config["numeric_features"]:
            lower, upper = OBSERVED_NUMERIC_RANGES[feature]
            value_type: Any = Annotated[
                float,
                Field(
                    ge=lower,
                    le=upper,
                    description=(
                        f"Observed prepared-dataset range [{lower}, {upper}]. "
                        "This is a model-support bound, not a clinical threshold."
                    ),
                ),
            ]
        else:
            allowed_values = config["observed_categories"].get(feature)
            if not allowed_values:
                raise ValueError(f"No observed category values are configured for {feature!r}.")
            value_type = Literal[tuple(allowed_values)]

        if feature in OPTIONAL_FEATURES:
            value_type = Optional[value_type]
            default_value = None
        else:
            default_value = ...
        fields[feature] = (value_type, default_value)

    return create_model(
        "PatientInput",
        __base__=PatientInputBase,
        **fields,
    )


PatientInput = _build_patient_input_model()


class ProbabilityResponse(BaseModel):
    low: float = Field(ge=0.0, le=1.0)
    moderate: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_risk: Literal["Low", "Moderate", "High"]
    probability_low: float = Field(ge=0.0, le=1.0)
    probability_moderate: float = Field(ge=0.0, le=1.0)
    probability_high: float = Field(ge=0.0, le=1.0)
    probabilities: ProbabilityResponse
    disclaimer: str