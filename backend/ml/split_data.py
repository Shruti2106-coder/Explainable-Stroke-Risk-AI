"""Create deterministic stratified train, validation, and locked final-test partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .preprocessing import load_feature_config
except ImportError:
    from preprocessing import load_feature_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "stroke_risk_prepared.csv"
DEFAULT_CONFIG = Path(__file__).with_name("feature_config.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Prepared CSV path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Feature config JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--validation-size", type=float, default=0.15)
    return parser.parse_args()


def stratified_partitions(
    data: pd.DataFrame,
    target_column: str,
    *,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    random_state: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = 1.0 - train_size - validation_size
    if train_size <= 0 or validation_size <= 0 or test_size <= 0:
        raise ValueError("Train, validation, and test proportions must all be positive.")
    if abs(train_size + validation_size + test_size - 1.0) > 1e-9:
        raise ValueError("Split proportions must sum to 1.")
    if target_column not in data:
        raise ValueError(f"Target column {target_column!r} is missing from the dataset.")
    if data[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")
    if data.index.has_duplicates:
        raise ValueError("Input frame must have unique row indices to verify partition disjointness.")

    train, holdout = train_test_split(
        data,
        train_size=train_size,
        random_state=random_state,
        stratify=data[target_column],
    )
    test_fraction_of_holdout = test_size / (validation_size + test_size)
    validation, final_test = train_test_split(
        holdout,
        test_size=test_fraction_of_holdout,
        random_state=random_state,
        stratify=holdout[target_column],
    )
    train_indices = set(train.index)
    validation_indices = set(validation.index)
    test_indices = set(final_test.index)
    if (
        train_indices & validation_indices
        or train_indices & test_indices
        or validation_indices & test_indices
    ):
        raise RuntimeError("Split partitions overlap.")
    if len(train) + len(validation) + len(final_test) != len(data):
        raise RuntimeError("Split partitions do not cover all input rows.")
    return train.copy(), validation.copy(), final_test.copy()


def _counts_by_class(data: pd.DataFrame, target: str, classes: list[str]) -> dict[str, int]:
    counts = data[target].astype(str).value_counts()
    return {risk_class: int(counts.get(risk_class, 0)) for risk_class in classes}


def create_splits(
    data_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    random_state: int = DEFAULT_SEED,
    train_size: float = 0.70,
    validation_size: float = 0.15,
) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {data_path}")
    config = load_feature_config(config_path)
    data = pd.read_csv(data_path)
    target_column = config["target_column"]
    expected_columns = set(config["features_safe_for_prediction"]) | {target_column}
    if set(data.columns) != expected_columns:
        raise ValueError("Prepared CSV columns do not match the saved approved feature configuration.")
    unexpected_classes = set(data[target_column].astype(str).unique()).difference(
        config["target_classes"]
    )
    if unexpected_classes:
        raise ValueError(f"Unexpected target classes: {sorted(unexpected_classes)}")

    train, validation, final_test = stratified_partitions(
        data,
        target_column,
        train_size=train_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    validation_path = output_dir / "validation.csv"
    test_path = output_dir / "final_test_locked.csv"
    train.reset_index(drop=True).to_csv(train_path, index=False)
    validation.reset_index(drop=True).to_csv(validation_path, index=False)
    final_test.reset_index(drop=True).to_csv(test_path, index=False)

    test_size = 1.0 - train_size - validation_size
    manifest: dict[str, Any] = {
        "random_state": random_state,
        "split_proportions": {
            "train": train_size,
            "validation": validation_size,
            "final_test": test_size,
        },
        "target_column": target_column,
        "target_classes": config["target_classes"],
        "train": {
            "file": train_path.name,
            "rows": len(train),
            "class_counts": _counts_by_class(train, target_column, config["target_classes"]),
        },
        "validation": {
            "file": validation_path.name,
            "rows": len(validation),
            "class_counts": _counts_by_class(validation, target_column, config["target_classes"]),
            "purpose": "model selection and tuning only",
        },
        "final_test": {
            "file": test_path.name,
            "rows": len(final_test),
            "status": "locked until final evaluation",
            "class_counts": "withheld until final evaluation",
        },
        "feature_policy": "Columns match feature_config.json; no identifier or leakage columns.",
        "balancing_policy": "Balancing utilities may read train.csv only.",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Input rows: {len(data):,}")
    for split_name, frame in (("train", train), ("validation", validation)):
        print(f"{split_name.title()} rows: {len(frame):,}; class counts: "
              f"{_counts_by_class(frame, target_column, config['target_classes'])}")
    print(f"Final test rows: {len(final_test):,}; saved locked and not summarized or used for balancing.")
    print(f"Split files and manifest written to: {output_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    create_splits(
        args.data.resolve(),
        args.config.resolve(),
        args.output_dir.resolve(),
        random_state=args.seed,
        train_size=args.train_size,
        validation_size=args.validation_size,
    )


if __name__ == "__main__":
    main()