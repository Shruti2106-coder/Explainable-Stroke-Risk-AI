import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.services.shap_explanation import ShapExplanationService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def explanation_service(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("shap-artifacts")
    return ShapExplanationService(
        report_dir=output_dir / "reports",
        image_dir=output_dir / "images",
    )


def test_local_shap_explanation_uses_real_training_sample(explanation_service):
    training_sample = pd.read_csv(PROJECT_ROOT / "data/splits/train.csv", nrows=1)
    patient = training_sample.loc[0, explanation_service.safe_features].to_dict()

    result = explanation_service.get_prediction_explanation(patient)

    assert result["predicted_risk_class"] in explanation_service.classes
    assert set(result["probabilities"]) == {"Low", "Moderate", "High"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
    assert result["explanation_output"] == "raw_class_margin"
    assert result["features_contributing_toward_prediction"]
    assert all(
        entry["feature"] in explanation_service.safe_features
        for entry in result["features_contributing_toward_prediction"]
        + result["features_contributing_away_from_prediction"]
    )
    assert "not medical causation" in result["disclaimer"]
    assert not any(
        "Patient_ID" in entry["feature"]
        or entry["feature"] in explanation_service.config["features_excluded_due_to_leakage"]
        for entry in result["features_contributing_toward_prediction"]
        + result["features_contributing_away_from_prediction"]
    )


def test_global_shap_creates_frontend_artifacts_from_validation_rows(explanation_service):
    result = explanation_service.get_global_explanation(max_rows=6)

    assert result["explainer"] == "TreeExplainer"
    assert result["explained_split"] == "validation"
    assert result["rows_explained"] == 6
    assert len(result["feature_importance"]) == len(explanation_service.safe_features)
    assert result["feature_importance"][0]["mean_abs_shap"] >= result["feature_importance"][-1]["mean_abs_shap"]
    for artifact_path in result["artifacts"].values():
        assert Path(artifact_path).is_file()
    for summary_path in result["class_specific_summary_plots"].values():
        assert Path(summary_path).is_file()

    saved_result = json.loads(Path(result["artifacts"]["global_explanation_json"]).read_text())
    assert saved_result["feature_importance"][0]["feature"] == result["feature_importance"][0]["feature"]


def test_local_explanation_rejects_multiple_patients(explanation_service):
    with pytest.raises(ValueError, match="exactly one patient"):
        explanation_service.get_prediction_explanation(
            pd.DataFrame([{}, {}])
        )