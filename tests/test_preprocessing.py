import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from backend.ml.preprocessing import (
    build_preprocessor,
    build_training_pipeline,
    fit_preprocessor,
    get_transformed_feature_metadata,
    select_model_features,
    transform_features,
    validate_feature_config,
)


@pytest.fixture
def feature_config():
    return {
        "features_safe_for_prediction": ["Age", "BMI", "Gender"],
        "features_excluded_due_to_leakage": ["Stroke_Risk_Score"],
        "identifier_columns": ["Patient_ID"],
        "target_column": "Stroke_Risk",
        "target_classes": ["Low", "Moderate", "High"],
        "numeric_features": ["Age", "BMI"],
        "categorical_features": ["Gender"],
    }


@pytest.fixture
def patient_rows():
    return pd.DataFrame(
        {
            "Patient_ID": [11, 12, 13],
            "Age": [20.0, np.nan, 40.0],
            "BMI": [22.0, 28.0, 34.0],
            "Gender": ["Female", None, "Unknown-at-inference"],
            "Stroke_Risk_Score": [25, 45, 85],
            "Stroke_Risk": ["Low", "Moderate", "High"],
        }
    )


def test_feature_selection_drops_identifiers_leakage_and_target(feature_config, patient_rows):
    selected = select_model_features(patient_rows, feature_config)
    assert selected.columns.tolist() == ["Age", "BMI", "Gender"]


def test_config_rejects_leakage_feature_in_safe_list(feature_config):
    invalid_config = {**feature_config}
    invalid_config["features_safe_for_prediction"] = [
        *feature_config["features_safe_for_prediction"],
        "Stroke_Risk_Score",
    ]
    with pytest.raises(ValueError, match="forbidden columns"):
        validate_feature_config(invalid_config)


def test_preprocessor_fits_training_rows_and_handles_unseen_categories(
    feature_config, patient_rows
):
    training_rows = patient_rows.iloc[:2]
    preprocessor = fit_preprocessor(training_rows, feature_config)
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    assert numeric_imputer.statistics_[0] == 20.0

    validation_rows = patient_rows.iloc[2:]
    transformed = transform_features(preprocessor, validation_rows, feature_config)
    assert transformed.shape[0] == 1
    dense_output = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert np.isfinite(dense_output).all()


def test_transformed_names_map_back_to_source_columns(feature_config, patient_rows):
    preprocessor = fit_preprocessor(patient_rows.iloc[:2], feature_config)
    metadata = get_transformed_feature_metadata(preprocessor, feature_config)
    assert metadata
    assert all("transformed_name" in item and "source_feature" in item for item in metadata)
    assert {item["source_feature"] for item in metadata} == {"Age", "BMI", "Gender"}
    assert all("Patient_ID" not in item["transformed_name"] for item in metadata)
    assert all("Stroke_Risk_Score" not in item["transformed_name"] for item in metadata)


def test_builder_returns_unfitted_transformer(feature_config):
    preprocessor = build_preprocessor(feature_config)
    assert not hasattr(preprocessor, "transformers_")


def test_training_pipeline_keeps_preprocessing_inside_unfitted_pipeline(feature_config):
    pipeline = build_training_pipeline(feature_config, LogisticRegression())
    assert pipeline.named_steps["preprocessing"] is not None
    assert not hasattr(pipeline.named_steps["preprocessing"], "transformers_")
    assert not hasattr(pipeline.named_steps["model"], "classes_")