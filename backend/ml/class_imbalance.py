"""Compare class-weighting and training-only random oversampling; never reads final test data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight

try:
    from .preprocessing import load_feature_config, select_model_features
except ImportError:
    from preprocessing import load_feature_config, select_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_CONFIG = Path(__file__).with_name("feature_config.json")
DEFAULT_IMAGE = PROJECT_ROOT / "images" / "class_imbalance" / "train_distribution_before_after.png"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "class_imbalance.md"
RISK_COLORS = {"Low": "#4CAF7D", "Moderate": "#E5A84B", "High": "#D96B6B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [columns]
    for values in frame.itertuples(index=False, name=None):
        rows.append([f"{value:.4f}" if isinstance(value, float) else str(value) for value in values])

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"

    return "\n".join(
        [format_row(rows[0]), format_row(["---"] * len(columns))]
        + [format_row(row) for row in rows[1:]]
    )


def balanced_class_weights(y_train: pd.Series, classes: list[str]) -> dict[str, float]:
    observed = set(y_train.astype(str).unique())
    missing_classes = set(classes).difference(observed)
    if missing_classes:
        raise ValueError(f"Training data is missing configured classes: {sorted(missing_classes)}")
    weights = compute_class_weight(
        class_weight="balanced",
        classes=pd.Index(classes).to_numpy(),
        y=y_train.astype(str).to_numpy(),
    )
    return {risk_class: float(weight) for risk_class, weight in zip(classes, weights, strict=True)}


def oversample_training_partition(
    training_frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Randomly duplicate minority training rows only; no synthetic feature values are made."""
    target_column = config["target_column"]
    if target_column not in training_frame:
        raise ValueError("Training partition must include the target column.")
    y_train = training_frame[target_column].astype(str)
    X_train = select_model_features(training_frame, config)
    sampler = RandomOverSampler(random_state=random_state)
    balanced_x, balanced_y = sampler.fit_resample(X_train, y_train)
    return balanced_x, pd.Series(balanced_y, name=target_column)


def _plot_distributions(
    train_counts: pd.Series,
    balanced_counts: pd.Series,
    classes: list[str],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, counts, title in zip(
        axes,
        (train_counts, balanced_counts),
        ("Training Before Balancing", "Training After Random Oversampling"),
        strict=True,
    ):
        colors = [RISK_COLORS[risk_class] for risk_class in classes]
        bars = axis.bar(classes, [int(counts.get(risk_class, 0)) for risk_class in classes], color=colors)
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("Stroke_Risk class")
        axis.set_ylabel("Training records")
        axis.grid(axis="y", color="#DCE6E9", linewidth=0.8)
        axis.set_axisbelow(True)
        for bar in bars:
            axis.annotate(
                f"{int(bar.get_height()):,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
    figure.suptitle("Class Counts: Training Data Only", x=0.07, ha="left", weight="bold")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def analyze_imbalance(
    split_dir: Path,
    config_path: Path,
    image_path: Path,
    report_path: Path,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    train_path = split_dir / "train.csv"
    validation_path = split_dir / "validation.csv"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Run backend/ml/split_data.py before class-imbalance analysis.")
    config = load_feature_config(config_path)
    target = config["target_column"]
    classes = config["target_classes"]
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    expected_columns = set(config["features_safe_for_prediction"]) | {target}
    if set(train.columns) != expected_columns or set(validation.columns) != expected_columns:
        raise ValueError("Train/validation columns do not match the approved feature config.")

    train_y = train[target].astype(str)
    validation_y = validation[target].astype(str)
    if set(train_y.unique()) != set(classes) or set(validation_y.unique()) != set(classes):
        raise ValueError("Train and validation sets must contain each configured target class.")
    train_counts = train_y.value_counts().reindex(classes, fill_value=0)
    validation_counts = validation_y.value_counts().reindex(classes, fill_value=0)
    weights = balanced_class_weights(train_y, classes)

    balanced_x, balanced_y = oversample_training_partition(
        train, config, random_state=random_state
    )
    balanced_counts = balanced_y.value_counts().reindex(classes, fill_value=0)
    if len(balanced_x) != len(balanced_y):
        raise RuntimeError("Resampled features and labels have different row counts.")
    _plot_distributions(train_counts, balanced_counts, classes, image_path)

    counts_table = pd.DataFrame(
        {
            "train_before": train_counts,
            "train_after_random_oversampling": balanced_counts,
            "validation_untouched": validation_counts,
            "balanced_class_weight": pd.Series(weights),
        }
    ).rename_axis("Stroke_Risk")
    report = f"""# Multi-Class Imbalance and Split Strategy

## Scope and data boundary

- Source partitions: `data/splits/train.csv` and `data/splits/validation.csv` only.
- `data/splits/final_test_locked.csv` was not opened by this analysis or used for balancing.
- No model was fitted and no final estimator/balancing method was selected.
- Class weighting is computed from training labels only. Random oversampling is applied only to the training partition, for distribution comparison and visualization.

## Training and validation class counts

{_markdown_table(counts_table.reset_index())}

Training and validation class proportions are preserved by stratified splitting. The final test partition remains reserved for one-time final evaluation after model development and validation are complete.

![Training class counts before and after random oversampling](../images/class_imbalance/train_distribution_before_after.png)

The plot compares original training counts against a temporary random-oversampled training sample. Each row in the oversampled sample is an exact duplicate of a training row; no validation or final-test row is included and no synthetic test observations are created. The temporary sample is not saved as a reusable dataset.

## Candidate approaches

### Balanced class weights

Use estimator support such as `class_weight="balanced"` or the computed per-class weights above. This preserves the original training rows, requires no synthetic examples, and is a strong baseline for many linear models and tree classifiers. It changes the loss contribution, not the sample counts. It may be unavailable or behave differently for some estimators, so it should be selected based on validation metrics.

### Random oversampling

`RandomOverSampler` duplicates observed minority-class training rows until class counts match. It supports mixed numeric/categorical records without interpolating invalid fractional category values. It can increase overfitting by repeating examples. The reusable `build_random_oversampling_pipeline()` places preprocessing before the sampler inside an imbalanced-learn pipeline; sampler fitting happens only during `fit` and is skipped during validation/test prediction.

### SMOTE / SMOTENC

Vanilla SMOTE over one-hot encoded categorical inputs can interpolate fractional indicator values that do not represent valid categories. SMOTENC is designed for mixed numeric/categorical data and is a more appropriate synthetic alternative, but it should be evaluated inside training folds with imputation and scaling learned on those folds. No SMOTE/SMOTENC samples are generated in this module.

## Current recommendation

Use balanced class weights as the first comparison baseline because they preserve the observed rows and avoid synthetic samples. Compare it against random oversampling within stratified cross-validation or using the validation set, using per-class recall/F1 and macro-F1. This is a balancing-strategy recommendation only, not a final model selection. Keep the final test locked throughout those comparisons.

## Split protocol

- Stratified 70% training / 15% validation / 15% final test.
- Fixed split seed: `{random_state}`.
- Validation is reserved for tuning and model selection.
- Final test is saved as `final_test_locked.csv`; its class distribution and metrics are intentionally withheld until final evaluation.
- The `split_manifest.json` records the seed, proportions, train/validation counts, and locked test status.

No performance claims or causal conclusions are made.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "training_counts_before": {key: int(value) for key, value in train_counts.items()},
        "training_counts_after_random_oversampling": {
            key: int(value) for key, value in balanced_counts.items()
        },
        "validation_counts_untouched": {
            key: int(value) for key, value in validation_counts.items()
        },
        "balanced_class_weights": weights,
        "resampled_rows_created": len(balanced_y) - len(train_y),
    }
    print("Training class counts before balancing:")
    print(train_counts.to_string())
    print("Training class counts after random oversampling:")
    print(balanced_counts.to_string())
    print("Balanced class weights (training labels only):")
    print(weights)
    print("Validation class counts (no resampling):")
    print(validation_counts.to_string())
    print("Final test partition was not read.")
    print(f"Figure written to: {image_path}")
    print(f"Report written to: {report_path}")
    return summary


def main() -> None:
    args = parse_args()
    analyze_imbalance(
        args.split_dir.resolve(),
        args.config.resolve(),
        args.image.resolve(),
        args.report.resolve(),
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()