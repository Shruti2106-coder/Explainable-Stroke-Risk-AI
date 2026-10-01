import pandas as pd
from sklearn.linear_model import LogisticRegression

from backend.ml.class_imbalance import balanced_class_weights, oversample_training_partition
from backend.ml.preprocessing import build_random_oversampling_pipeline


def feature_config():
    return {
        "features_safe_for_prediction": ["Age", "Gender"],
        "features_excluded_due_to_leakage": ["Stroke_Risk_Score"],
        "identifier_columns": ["Patient_ID"],
        "target_column": "Stroke_Risk",
        "target_classes": ["Low", "Moderate", "High"],
        "numeric_features": ["Age"],
        "categorical_features": ["Gender"],
    }


def training_rows():
    return pd.DataFrame(
        {
            "Patient_ID": range(1, 11),
            "Age": range(20, 30),
            "Gender": ["A"] * 5 + ["B"] * 3 + ["C"] * 2,
            "Stroke_Risk_Score": [20] * 5 + [50] * 3 + [80] * 2,
            "Stroke_Risk": ["Low"] * 5 + ["Moderate"] * 3 + ["High"] * 2,
        }
    )


def test_balanced_weights_use_training_class_counts_only():
    labels = training_rows()["Stroke_Risk"]
    weights = balanced_class_weights(labels, feature_config()["target_classes"])
    assert weights["High"] > weights["Moderate"]
    assert weights["Low"] < weights["Moderate"]
    assert abs(sum(weights[risk] * int(labels.eq(risk).sum()) for risk in weights) - len(labels)) < 1e-8


def test_random_oversampling_only_duplicates_training_rows():
    config = feature_config()
    train = training_rows()
    balanced_x, balanced_y = oversample_training_partition(train, config, random_state=5)
    original = train[config["features_safe_for_prediction"]].copy()
    original[config["target_column"]] = train[config["target_column"]].astype(str)
    resampled = balanced_x.copy()
    resampled[config["target_column"]] = balanced_y

    assert balanced_y.value_counts().nunique() == 1
    for _, sample in resampled.iterrows():
        matches = original.eq(sample).all(axis=1) | (original.isna() & sample.isna()).all(axis=1)
        assert matches.any()
    assert "Patient_ID" not in balanced_x
    assert "Stroke_Risk_Score" not in balanced_x


def test_random_oversampling_pipeline_is_unfitted_and_contains_sampler():
    pipeline = build_random_oversampling_pipeline(feature_config(), LogisticRegression())
    assert pipeline.named_steps["sampling"].__class__.__name__ == "RandomOverSampler"
    assert not hasattr(pipeline.named_steps["preprocessing"], "transformers_")
    assert not hasattr(pipeline.named_steps["model"], "classes_")