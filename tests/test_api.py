import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.main import app
from app.schemas.patient import PatientInput


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def patient_payload():
    sample = pd.read_csv(PROJECT_ROOT / "data/splits/train.csv", nrows=1).iloc[0]
    payload = {}
    for feature in PatientInput.model_fields:
        value = sample[feature]
        payload[feature] = None if pd.isna(value) else value.item() if hasattr(value, "item") else value
    return payload


def test_health_endpoint_loads_saved_model(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_endpoint_returns_multiclass_probabilities(api_client, patient_payload):
    response = api_client.post("/predict", json=patient_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_risk"] in {"Low", "Moderate", "High"}
    assert set(body["probabilities"]) == {"low", "moderate", "high"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert body["disclaimer"].startswith("Educational output only")


def test_model_info_endpoint_returns_feature_policy(api_client):
    response = api_client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "XGBoost"
    assert body["target_classes"] == ["Low", "Moderate", "High"]
    assert "Patient_ID" not in body["approved_features"]
    assert "Stroke_Risk_Score" in body["excluded_leakage_features"]


@pytest.mark.parametrize("path", ["/explain", "/explanations/prediction"])
def test_explain_endpoint_returns_local_shap_data(api_client, patient_payload, path):
    response = api_client.post(path, json=patient_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_risk_class"] in {"Low", "Moderate", "High"}
    assert set(body["probabilities"]) == {"Low", "Moderate", "High"}
    assert body["features_contributing_toward_prediction"]
    assert "not medical causation" in body["disclaimer"]


def test_global_explanation_endpoint_returns_artifacts(api_client):
    response = api_client.get("/explanations/global")
    assert response.status_code == 200
    body = response.json()
    assert body["explainer"] == "TreeExplainer"
    assert body["rows_explained"] > 0
    assert Path(body["artifacts"]["feature_importance_csv"]).is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Age", 1000),
        ("Country", "Atlantis"),
        ("Patient_ID", 5),
        ("Blood_Pressure_Systolic", 70),
    ],
)
def test_predict_rejects_out_of_domain_or_unapproved_inputs(
    api_client, patient_payload, field, value
):
    invalid_payload = {**patient_payload, field: value}
    response = api_client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_accepts_training_supported_missing_values(api_client, patient_payload):
    patient_payload["Age"] = None
    response = api_client.post("/predict", json=patient_payload)
    assert response.status_code == 200


def test_predict_rejects_missing_required_field_and_malformed_body(api_client, patient_payload):
    missing_required = {key: value for key, value in patient_payload.items() if key != "Gender"}
    assert api_client.post("/predict", json=missing_required).status_code == 422
    assert api_client.post("/predict", json=["not", "an", "object"]).status_code == 422


def test_cors_allows_configured_vite_origin(api_client, patient_payload):
    response = api_client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"