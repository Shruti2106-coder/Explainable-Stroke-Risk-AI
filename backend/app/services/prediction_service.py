"""Load and run the finalized pipeline without retraining."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd

try:
    from backend.ml.preprocessing import select_model_features
except ImportError:
    from ml.preprocessing import select_model_features

from ..schemas.patient import PatientInput
from ..schemas.patient import OBSERVED_NUMERIC_RANGES, OPTIONAL_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_PATH = PROJECT_ROOT / "models" / "final_stroke_risk_pipeline.joblib"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "models" / "final_feature_config.json"
METADATA_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"
MEDICAL_DISCLAIMER = (
    "Educational output only. This is not a medical diagnosis or a substitute for "
    "professional medical advice."
)


class PredictionService:
    def __init__(self) -> None:
        for artifact_path in (MODEL_PATH, FEATURE_CONFIG_PATH, METADATA_PATH):
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Required model artifact not found: {artifact_path}")
        self.pipeline = joblib.load(MODEL_PATH)
        self.config = json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
        self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.classes = [str(value) for value in self.pipeline.named_steps["model"].classes_]
        expected_classes = set(self.config["target_classes"])
        if set(self.classes) != expected_classes:
            raise RuntimeError("Saved model classes do not match the feature configuration.")
        approved = set(self.config["features_safe_for_prediction"])
        if approved != set(self.metadata["approved_features"]):
            raise RuntimeError("Saved model and metadata use different approved feature sets.")

    def predict(self, patient: PatientInput | dict[str, Any]) -> dict[str, Any]:
        payload = patient.model_dump() if isinstance(patient, PatientInput) else patient
        frame = pd.DataFrame([payload])
        features = select_model_features(frame, self.config)
        predicted_risk = str(self.pipeline.predict(features)[0])
        raw_probabilities = self.pipeline.predict_proba(features)[0]
        class_to_probability = {
            risk_class: float(raw_probabilities[index])
            for index, risk_class in enumerate(self.classes)
        }
        probabilities = {
            risk_class: class_to_probability[risk_class]
            for risk_class in ("Low", "Moderate", "High")
        }
        return {
            "predicted_risk": predicted_risk,
            "probability_low": probabilities["Low"],
            "probability_moderate": probabilities["Moderate"],
            "probability_high": probabilities["High"],
            "probabilities": {
                "low": probabilities["Low"],
                "moderate": probabilities["Moderate"],
                "high": probabilities["High"],
            },
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def model_info(self) -> dict[str, Any]:
        comparison_path = PROJECT_ROOT / "reports" / "baseline_vs_tuned_validation.csv"
        model_comparison = []
        if comparison_path.is_file():
            comparison_frame = pd.read_csv(comparison_path)
            model_comparison = json.loads(comparison_frame.to_json(orient="records"))
        return {
            "model_name": self.metadata["model_name"],
            "model_type": (
                f"{type(self.pipeline.named_steps['model']).__module__}."
                f"{type(self.pipeline.named_steps['model']).__name__}"
            ),
            "target_column": self.config["target_column"],
            "target_classes": ["Low", "Moderate", "High"],
            "approved_features": self.config["features_safe_for_prediction"],
            "numeric_features": self.config["numeric_features"],
            "categorical_features": self.config["categorical_features"],
            "categorical_options": self.config["observed_categories"],
            "numeric_ranges": {
                feature: {"min": bounds[0], "max": bounds[1]}
                for feature, bounds in OBSERVED_NUMERIC_RANGES.items()
            },
            "optional_features": sorted(OPTIONAL_FEATURES),
            "excluded_leakage_features": self.config["features_excluded_due_to_leakage"],
            "identifier_columns": self.config["identifier_columns"],
            "fit_rows": self.metadata["fit_rows"],
            "validation_macro_f1": self.metadata["validation_metrics"]["macro_f1"],
            "final_test_metrics": self.metadata["final_test_metrics"],
            "final_test_class_order": self.metadata["class_order"],
            "model_comparison_validation": model_comparison,
            "disclaimer": MEDICAL_DISCLAIMER,
        }


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Load the serialized model/preprocessor once; never fit or retrain at startup."""
    return PredictionService()


def is_prediction_service_loaded() -> bool:
    return get_prediction_service.cache_info().currsize > 0