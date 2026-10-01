"""Leakage-safe preprocessing shared by training, validation, and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONFIG_PATH = Path(__file__).with_name("feature_config.json")
REQUIRED_CONFIG_KEYS = {
    "features_safe_for_prediction",
    "features_excluded_due_to_leakage",
    "identifier_columns",
    "target_column",
    "numeric_features",
    "categorical_features",
    "target_classes",
}


def validate_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    missing_keys = REQUIRED_CONFIG_KEYS.difference(config)
    if missing_keys:
        raise ValueError(f"Feature config is missing keys: {', '.join(sorted(missing_keys))}")

    safe_features = config["features_safe_for_prediction"]
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    target_column = config["target_column"]
    forbidden = (
        set(config["features_excluded_due_to_leakage"])
        | set(config["identifier_columns"])
        | {target_column}
    )
    if len(safe_features) != len(set(safe_features)):
        raise ValueError("Safe feature list contains duplicate columns.")
    if len(numeric_features) != len(set(numeric_features)):
        raise ValueError("Numeric feature list contains duplicate columns.")
    if len(categorical_features) != len(set(categorical_features)):
        raise ValueError("Categorical feature list contains duplicate columns.")
    overlap = set(safe_features) & forbidden
    if overlap:
        raise ValueError(f"Safe features include forbidden columns: {sorted(overlap)}")
    if set(numeric_features) & set(categorical_features):
        raise ValueError("Numeric and categorical feature lists overlap.")
    if set(numeric_features) | set(categorical_features) != set(safe_features):
        raise ValueError("Numeric and categorical lists must partition the safe features.")
    if not safe_features:
        raise ValueError("At least one safe feature is required.")
    return config


def load_feature_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return validate_feature_config(config)


def select_model_features(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    include_target: bool = False,
) -> pd.DataFrame:
    """Select exactly the approved columns; unrelated or forbidden inputs are ignored."""
    validate_feature_config(config)
    safe_features = config["features_safe_for_prediction"]
    target_column = config["target_column"]
    required = set(safe_features)
    if include_target:
        required.add(target_column)
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Input is missing configured columns: {', '.join(sorted(missing_columns))}")

    selected = data.loc[:, safe_features].copy()
    for column in config["categorical_features"]:
        values = selected[column].astype("string").str.strip()
        values = values.mask(values.eq(""), pd.NA)
        selected[column] = pd.Series(
            values.to_numpy(dtype=object, na_value=np.nan),
            index=selected.index,
            dtype=object,
        )
    for column in config["numeric_features"]:
        selected[column] = pd.to_numeric(selected[column], errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )

    if include_target:
        target_values = data[target_column]
        if target_values.isna().any():
            raise ValueError(f"Target column {target_column!r} contains missing values.")
        observed = set(target_values.astype(str).unique())
        unexpected = observed.difference(config["target_classes"])
        if unexpected:
            raise ValueError(f"Unexpected target classes: {sorted(unexpected)}")
        selected[target_column] = target_values
    return selected


def build_preprocessor(
    config: dict[str, Any],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build an unfitted transformer; fitting statistics must come from training rows only."""
    validate_feature_config(config)
    numeric_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config["numeric_features"]),
            ("categorical", categorical_pipeline, config["categorical_features"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_training_pipeline(
    config: dict[str, Any],
    estimator: Any,
    *,
    scale_numeric: bool = True,
) -> Pipeline:
    """Wrap preprocessing and an estimator so each fit/CV fold learns its own transforms."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor(config, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def build_random_oversampling_pipeline(
    config: dict[str, Any],
    estimator: Any,
    *,
    random_state: int = 42,
    scale_numeric: bool = True,
) -> ImbalancedPipeline:
    """Build a pipeline that resamples only inside fit, after train-fold preprocessing."""
    validate_feature_config(config)
    return ImbalancedPipeline(
        steps=[
            ("preprocessing", build_preprocessor(config, scale_numeric=scale_numeric)),
            ("sampling", RandomOverSampler(random_state=random_state)),
            ("model", estimator),
        ]
    )


def fit_preprocessor(
    training_frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Fit preprocessing on a training partition only; does not persist the result."""
    training_features = select_model_features(training_frame, config)
    return build_preprocessor(config, scale_numeric=scale_numeric).fit(training_features)


def transform_features(
    fitted_preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> Any:
    """Apply a fitted training transformer to validation, test, or API input rows."""
    if not hasattr(fitted_preprocessor, "transformers_"):
        raise ValueError("Preprocessor must be fitted on training data before transform.")
    features = select_model_features(frame, config)
    return fitted_preprocessor.transform(features)


def get_transformed_feature_metadata(
    fitted_preprocessor: ColumnTransformer,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """Return each transformed name with its source column for SHAP attribution."""
    if not hasattr(fitted_preprocessor, "transformers_"):
        raise ValueError("Preprocessor must be fitted before feature metadata is available.")
    validate_feature_config(config)
    transformed_names = fitted_preprocessor.get_feature_names_out().tolist()
    source_columns: list[str] = []

    numeric_pipeline = fitted_preprocessor.named_transformers_["numeric"]
    numeric_names = numeric_pipeline.get_feature_names_out(config["numeric_features"])
    for feature_name in numeric_names:
        if feature_name.startswith("missingindicator_"):
            source_columns.append(feature_name.removeprefix("missingindicator_"))
        else:
            source_columns.append(feature_name)

    categorical_pipeline = fitted_preprocessor.named_transformers_["categorical"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["one_hot"]
    imputed_categorical_names = categorical_imputer.get_feature_names_out(
        config["categorical_features"]
    )
    for feature_name, categories in zip(
        imputed_categorical_names, encoder.categories_, strict=True
    ):
        source_feature = (
            feature_name.removeprefix("missingindicator_")
            if feature_name.startswith("missingindicator_")
            else feature_name
        )
        source_columns.extend([source_feature] * len(categories))
    if len(transformed_names) != len(source_columns):
        raise RuntimeError(
            "Transformed feature names could not be aligned with source columns; "
            "inspect the fitted transformer configuration."
        )
    return [
        {"transformed_name": transformed_name, "source_feature": source_feature}
        for transformed_name, source_feature in zip(transformed_names, source_columns, strict=True)
    ]


def get_transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Return the exact transformed names in the transformer's output order."""
    if not hasattr(fitted_preprocessor, "transformers_"):
        raise ValueError("Preprocessor must be fitted before output names are available.")
    return fitted_preprocessor.get_feature_names_out().tolist()