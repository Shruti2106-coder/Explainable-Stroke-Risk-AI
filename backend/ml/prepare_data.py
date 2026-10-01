"""Prepare a model-input copy using the persisted feature configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

try:
    from .preprocessing import build_preprocessor as _build_preprocessor
    from .preprocessing import load_feature_config
except ImportError:
    from preprocessing import build_preprocessor as _build_preprocessor
    from preprocessing import load_feature_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "stroke_risk_prediction_dataset.csv"
DEFAULT_CONFIG = Path(__file__).with_name("feature_config.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "stroke_risk_prepared.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Source CSV path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Feature config JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Prepared CSV output path")
    return parser.parse_args()


def prepare_frame(
    data: pd.DataFrame,
    config: dict[str, Any],
    require_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean rows and select configured columns without fitting any statistics."""
    safe_features = config["features_safe_for_prediction"]
    target_column = config["target_column"]
    required_columns = set(safe_features)
    if require_target:
        required_columns.add(target_column)
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {', '.join(sorted(missing_columns))}")

    prepared = data.copy()
    for column in [*config["categorical_features"], target_column]:
        if column in prepared.columns:
            values = prepared[column].astype("string").str.strip()
            prepared[column] = values.mask(values.eq(""), pd.NA)

    if require_target:
        if prepared[target_column].isna().any():
            raise ValueError(f"Target column {target_column!r} contains missing values.")
        observed_targets = set(prepared[target_column].astype(str).unique())
        unexpected_targets = observed_targets.difference(config["target_classes"])
        if unexpected_targets:
            raise ValueError(f"Unexpected target classes: {sorted(unexpected_targets)}")

    input_rows = len(prepared)
    duplicate_mask = prepared.duplicated()
    duplicates_removed = int(duplicate_mask.sum())
    prepared = prepared.loc[~duplicate_mask].copy()

    numeric_features = config["numeric_features"]
    for column in numeric_features:
        prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)

    systolic_column = "Blood_Pressure_Systolic"
    diastolic_column = "Blood_Pressure_Diastolic"
    invalid_pressure_rows = 0
    if systolic_column in prepared and diastolic_column in prepared:
        comparable = prepared[systolic_column].notna() & prepared[diastolic_column].notna()
        invalid_pressure = comparable & prepared[systolic_column].le(prepared[diastolic_column])
        invalid_pressure_rows = int(invalid_pressure.sum())
        prepared = prepared.loc[~invalid_pressure].copy()

    selected_columns = list(safe_features)
    if require_target:
        selected_columns.append(target_column)
    model_frame = prepared.loc[:, selected_columns].copy()
    summary = {
        "input_rows": input_rows,
        "exact_duplicates_removed": duplicates_removed,
        "invalid_blood_pressure_rows_removed": invalid_pressure_rows,
        "output_rows": len(model_frame),
        "missing_feature_cells_retained": int(model_frame[safe_features].isna().sum().sum()),
    }
    return model_frame, summary


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    """Compatibility wrapper around the shared Module 4 preprocessor builder."""
    return _build_preprocessor(config)


def prepare_dataset(csv_path: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV was not found: {csv_path}")
    if csv_path.resolve() == output_path.resolve():
        raise ValueError("Prepared output must not overwrite the original CSV.")

    config = load_feature_config(config_path)
    source = pd.read_csv(csv_path)
    model_frame, summary = prepare_frame(source, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_frame.to_csv(output_path, index=False)
    summary["output_columns"] = model_frame.columns.tolist()
    summary["safe_feature_count"] = len(config["features_safe_for_prediction"])
    summary["target_column"] = config["target_column"]
    summary["output_path"] = str(output_path)

    print(json.dumps(summary, indent=2))
    print("Prepared class distribution:")
    print(model_frame[config["target_column"]].value_counts().to_string())
    print("Missing values retained for train-only imputation:")
    print(model_frame[config["features_safe_for_prediction"]].isna().sum().to_string())
    print(f"Prepared data written to: {output_path}")
    print("Preprocessor created but not fitted; fit it on training data only.")
    return summary


def main() -> None:
    args = parse_args()
    prepare_dataset(args.csv.resolve(), args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()