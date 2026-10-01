"""Train baseline multi-class models on train.csv and evaluate on validation.csv only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from .preprocessing import build_training_pipeline, load_feature_config, select_model_features
except ImportError:
    from preprocessing import build_training_pipeline, load_feature_config, select_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_CONFIG = Path(__file__).with_name("feature_config.json")
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "images" / "model_evaluation"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
RISK_COLORS = {"Low": "#4CAF7D", "Moderate": "#E5A84B", "High": "#D96B6B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def _model_specs(random_state: int) -> dict[str, tuple[str, Any, bool, str]]:
    """Return model label, estimator, numeric-scaling flag, and imbalance method."""
    return {
        "logistic_regression": (
            "Logistic Regression",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1500,
                random_state=random_state,
                solver="lbfgs",
            ),
            True,
            "class_weight=balanced",
        ),
        "random_forest": (
            "Random Forest",
            RandomForestClassifier(
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_estimators=300,
                n_jobs=-1,
                random_state=random_state,
            ),
            False,
            "class_weight=balanced_subsample",
        ),
        "xgboost": (
            "XGBoost",
            XGBClassifier(
                colsample_bytree=0.85,
                eval_metric="mlogloss",
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=2,
                n_estimators=250,
                n_jobs=-1,
                objective="multi:softprob",
                num_class=3,
                random_state=random_state,
                reg_lambda=1.0,
                subsample=0.85,
                tree_method="hist",
            ),
            False,
            "balanced per-row sample_weight computed from training labels",
        ),
        "linear_svm": (
            "Linear SVM",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=10000,
                random_state=random_state,
            ),
            True,
            "class_weight=balanced",
        ),
    }


def _aligned_continuous_scores(
    pipeline: Any,
    X_validation: pd.DataFrame,
    class_order: list[str],
    *,
    class_labels: list[str] | None = None,
) -> tuple[np.ndarray, str]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        raw_scores = pipeline.predict_proba(X_validation)
        score_source = "predicted probability"
    else:
        raw_scores = pipeline.decision_function(X_validation)
        score_source = "one-vs-rest decision score"
    raw_scores = np.asarray(raw_scores)
    if raw_scores.ndim != 2:
        raise ValueError("Multi-class scoring requires one continuous score column per class.")
    model_classes = class_labels or [str(value) for value in model.classes_]
    missing_classes = set(class_order).difference(model_classes)
    if missing_classes:
        raise ValueError(f"Fitted model is missing classes: {sorted(missing_classes)}")
    class_indices = [model_classes.index(risk_class) for risk_class in class_order]
    return raw_scores[:, class_indices], score_source


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    continuous_scores: np.ndarray,
    class_order: list[str],
) -> dict[str, Any]:
    y_true = y_true.astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    if continuous_scores.shape != (len(y_true), len(class_order)):
        raise ValueError("Continuous score array shape must be (rows, target classes).")
    if set(y_true.unique()) != set(class_order):
        raise ValueError("Validation labels must contain every configured target class.")

    per_class_roc_auc: dict[str, float] = {}
    per_class_average_precision: dict[str, float] = {}
    for index, risk_class in enumerate(class_order):
        binary_target = y_true.eq(risk_class).astype(int)
        class_scores = continuous_scores[:, index]
        per_class_roc_auc[risk_class] = float(roc_auc_score(binary_target, class_scores))
        per_class_average_precision[risk_class] = float(
            average_precision_score(binary_target, class_scores)
        )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_order, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=class_order, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=class_order, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=class_order, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(np.mean(list(per_class_roc_auc.values()))),
        "roc_auc_ovr_per_class": per_class_roc_auc,
        "pr_auc_ovr_macro_average_precision": float(
            np.mean(list(per_class_average_precision.values()))
        ),
        "average_precision_ovr_per_class": per_class_average_precision,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_order).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=class_order,
            target_names=class_order,
            output_dict=True,
            zero_division=0,
        ),
    }


def _save_confusion_matrix(
    matrix: list[list[int]],
    classes: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        np.asarray(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=axis,
    )
    axis.set_title(f"{model_name}: Validation Confusion Matrix", loc="left", weight="bold")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_roc_and_pr_curves(
    y_true: pd.Series,
    scores: np.ndarray,
    classes: list[str],
    model_name: str,
    output_prefix: Path,
) -> None:
    figure_roc, axis_roc = plt.subplots(figsize=(7.5, 6))
    figure_pr, axis_pr = plt.subplots(figsize=(7.5, 6))
    for index, risk_class in enumerate(classes):
        binary_target = y_true.eq(risk_class).astype(int)
        class_scores = scores[:, index]
        fpr, tpr, _ = roc_curve(binary_target, class_scores)
        roc_auc = roc_auc_score(binary_target, class_scores)
        axis_roc.plot(
            fpr,
            tpr,
            color=RISK_COLORS[risk_class],
            label=f"{risk_class} (AUC={roc_auc:.3f})",
            linewidth=2,
        )
        precision, recall, _ = precision_recall_curve(binary_target, class_scores)
        average_precision = average_precision_score(binary_target, class_scores)
        axis_pr.plot(
            recall,
            precision,
            color=RISK_COLORS[risk_class],
            label=f"{risk_class} (AP={average_precision:.3f})",
            linewidth=2,
        )
        axis_pr.axhline(
            binary_target.mean(),
            color=RISK_COLORS[risk_class],
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
        )

    axis_roc.plot([0, 1], [0, 1], color="#748A92", linestyle="--", linewidth=1)
    axis_roc.set(
        title=f"{model_name}: One-vs-Rest ROC (validation)",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis_roc.legend(frameon=False, loc="lower right")
    axis_pr.set(
        title=f"{model_name}: One-vs-Rest Precision-Recall (validation)",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis_pr.legend(frameon=False, loc="upper right")
    figure_roc.tight_layout()
    figure_pr.tight_layout()
    figure_roc.savefig(
        output_prefix.with_name(f"{output_prefix.name}_roc.png"),
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    figure_pr.savefig(
        output_prefix.with_name(f"{output_prefix.name}_precision_recall.png"),
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure_roc)
    plt.close(figure_pr)


def _save_comparison_chart(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    metric_columns = [
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "pr_auc_ovr_macro_average_precision",
    ]
    labels = {
        "macro_precision": "Macro precision",
        "macro_recall": "Macro recall",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "roc_auc_ovr_macro": "Macro OVR ROC-AUC",
        "pr_auc_ovr_macro_average_precision": "Macro OVR PR-AUC (AP)",
    }
    frame = pd.DataFrame(
        {model_key: result["metrics"] for model_key, result in results.items()}
    ).T.loc[:, metric_columns]
    axis = frame.rename(columns=labels).plot(kind="bar", figsize=(13, 6), width=0.82)
    axis.set_title("Baseline Model Comparison on Validation Data", loc="left", weight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_xticklabels([result["display_name"] for result in results.values()], rotation=15, ha="right")
    axis.grid(axis="y", color="#DCE6E9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(axis.figure)


def _rank_tuning_candidates(results: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        results,
        key=lambda model_key: (
            -results[model_key]["metrics"]["macro_f1"],
            -results[model_key]["metrics"]["classification_report"]["Low"]["recall"],
            -results[model_key]["metrics"]["roc_auc_ovr_macro"],
            -results[model_key]["metrics"]["pr_auc_ovr_macro_average_precision"],
        ),
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [columns]
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            [
                f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in values
            ]
        )

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"

    return "\n".join(
        [format_row(rows[0]), format_row(["---"] * len(columns))]
        + [format_row(row) for row in rows[1:]]
    )


def run_baselines(
    split_dir: Path,
    config_path: Path,
    report_dir: Path,
    figure_dir: Path,
    model_dir: Path,
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    train_path = split_dir / "train.csv"
    validation_path = split_dir / "validation.csv"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Run backend/ml/split_data.py before baseline model training.")

    config = load_feature_config(config_path)
    target = config["target_column"]
    classes = [str(value) for value in config["target_classes"]]
    expected_columns = set(config["features_safe_for_prediction"]) | {target}
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    if set(train.columns) != expected_columns or set(validation.columns) != expected_columns:
        raise ValueError("Train/validation columns do not match the approved feature config.")
    if set(train[target].astype(str).unique()) != set(classes):
        raise ValueError("Training partition does not contain all configured classes.")
    if set(validation[target].astype(str).unique()) != set(classes):
        raise ValueError("Validation partition does not contain all configured classes.")

    X_train = select_model_features(train, config)
    y_train = train[target].astype(str)
    X_validation = select_model_features(validation, config)
    y_validation = validation[target].astype(str)
    xgb_sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    results: dict[str, dict[str, Any]] = {}
    classification_reports: dict[str, Any] = {}
    for model_key, (display_name, estimator, scale_numeric, imbalance_strategy) in _model_specs(
        random_state
    ).items():
        print(f"Training {display_name} on {len(X_train):,} training rows...")
        pipeline = build_training_pipeline(
            config,
            estimator,
            scale_numeric=scale_numeric,
        )
        fit_parameters = {}
        y_fit = y_train
        prediction_labels: list[str] | None = None
        if model_key == "xgboost":
            label_encoder = LabelEncoder().fit(y_train)
            y_fit = label_encoder.transform(y_train)
            prediction_labels = [str(label) for label in label_encoder.classes_]
            fit_parameters["model__sample_weight"] = xgb_sample_weights
        pipeline.fit(X_train, y_fit, **fit_parameters)

        predictions = pipeline.predict(X_validation)
        if model_key == "xgboost":
            predictions = label_encoder.inverse_transform(predictions.astype(int))
        continuous_scores, score_source = _aligned_continuous_scores(
            pipeline,
            X_validation,
            classes,
            class_labels=prediction_labels,
        )
        metrics = calculate_metrics(y_validation, predictions, continuous_scores, classes)
        metrics["score_source"] = score_source
        classification_reports[model_key] = metrics["classification_report"]
        results[model_key] = {
            "display_name": display_name,
            "imbalance_strategy": imbalance_strategy,
            "scale_numeric": scale_numeric,
            "metrics": metrics,
            "candidate_model_path": str(model_dir / f"{model_key}_baseline.joblib"),
        }

        _save_confusion_matrix(
            metrics["confusion_matrix"],
            classes,
            display_name,
            figure_dir / f"{model_key}_confusion_matrix.png",
        )
        _save_roc_and_pr_curves(
            y_validation,
            continuous_scores,
            classes,
            display_name,
            figure_dir / model_key,
        )
        joblib.dump(pipeline, model_dir / f"{model_key}_baseline.joblib")
        print(
            f"  macro F1={metrics['macro_f1']:.4f}; "
            f"Low recall={metrics['classification_report']['Low']['recall']:.4f}; "
            f"macro OVR ROC-AUC={metrics['roc_auc_ovr_macro']:.4f}; "
            f"macro OVR PR-AUC={metrics['pr_auc_ovr_macro_average_precision']:.4f}"
        )

    ranked_models = _rank_tuning_candidates(results)
    tuning_candidates = ranked_models[:2]
    comparison_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "pr_auc_ovr_macro_average_precision",
    ]
    comparison = pd.DataFrame(
        {
            model_key: {
                metric: result["metrics"][metric] for metric in comparison_columns
            }
            for model_key, result in results.items()
        }
    ).T
    comparison.insert(
        0,
        "model",
        [results[model_key]["display_name"] for model_key in comparison.index],
    )
    comparison.to_csv(report_dir / "model_comparison.csv", index_label="model_key")
    _save_comparison_chart(results, figure_dir / "model_comparison.png")

    result_document = {
        "evaluation_split": "validation",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "final_test_accessed": False,
        "target_column": target,
        "class_order": classes,
        "random_state": random_state,
        "selection_rule": (
            "Rank by validation macro-F1; tie-break by Low-class recall, then macro one-vs-rest "
            "ROC-AUC and macro one-vs-rest average precision. Advance the top two only to "
            "hyperparameter tuning; this is not final model selection."
        ),
        "tuning_candidates": [
            {
                "model_key": model_key,
                "model": results[model_key]["display_name"],
                "macro_f1": results[model_key]["metrics"]["macro_f1"],
                "low_recall": results[model_key]["metrics"]["classification_report"]["Low"]["recall"],
                "roc_auc_ovr_macro": results[model_key]["metrics"]["roc_auc_ovr_macro"],
                "pr_auc_ovr_macro_average_precision": results[model_key]["metrics"]["pr_auc_ovr_macro_average_precision"],
            }
            for model_key in tuning_candidates
        ],
        "models": results,
    }
    (report_dir / "model_evaluation.json").write_text(
        json.dumps(result_document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "classification_reports.json").write_text(
        json.dumps(classification_reports, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    tuning_details = "\n".join(
        f"{rank}. **{results[model_key]['display_name']}**: macro-F1 "
        f"{results[model_key]['metrics']['macro_f1']:.4f}, Low recall "
        f"{results[model_key]['metrics']['classification_report']['Low']['recall']:.4f}, "
        f"macro OVR ROC-AUC {results[model_key]['metrics']['roc_auc_ovr_macro']:.4f}, "
        f"macro OVR PR-AUC {results[model_key]['metrics']['pr_auc_ovr_macro_average_precision']:.4f}."
        for rank, model_key in enumerate(tuning_candidates, start=1)
    )
    report = f"""# Baseline Multi-Class Model Evaluation

## Evaluation boundary

- Models were fit using only `data/splits/train.csv` and evaluated using only `data/splits/validation.csv`.
- The locked final test partition was not opened, scored, or used for model selection.
- All features were selected from `features_safe_for_prediction` in `backend/ml/feature_config.json`; `Patient_ID` and the three leakage columns are excluded.
- Preprocessing is fit inside each model pipeline on training rows only. No resampling is applied to validation data.
- Results are baseline validation estimates, not medical claims or final test performance.

## Validation comparison

Macro averages weight each class equally, which is important given the class imbalance. Weighted averages reflect class support. ROC-AUC and PR-AUC are macro one-vs-rest; PR-AUC is calculated as per-class average precision and then averaged. SVM curves use its continuous one-vs-rest decision scores, while other models use predicted class probabilities.

{_markdown_table(comparison.reset_index(drop=True))}

## Tuning candidates

The documented ranking rule is validation macro-F1 first, with Low-class recall, macro one-vs-rest ROC-AUC, and macro one-vs-rest average precision as tie-breakers. These candidates should advance to hyperparameter tuning; this does not select a final model.

{tuning_details}

## Model settings

| Model | Imbalance approach | Numeric scaling |
| --- | --- | --- |
{chr(10).join(f"| {result['display_name']} | {result['imbalance_strategy']} | {'yes' if result['scale_numeric'] else 'no'} |" for result in results.values())}

## Artifacts

- Structured metrics: `reports/model_evaluation.json`
- Comparison table: `reports/model_comparison.csv`
- Per-model classification reports: `reports/classification_reports.json`
- Per-model confusion matrices, one-vs-rest ROC and precision-recall curves, and model comparison chart: `images/model_evaluation/`
- Train-fitted candidate pipelines: `models/*_baseline.joblib`

Each model's detailed class-wise precision, recall, F1, support, confusion matrix, ROC-AUC, and average precision is in the JSON results. No hyperparameter search was performed in this module.
"""
    (report_dir / "model_evaluation.md").write_text(report, encoding="utf-8")

    print("\nValidation comparison:")
    print(comparison.set_index("model").round(4).to_string())
    print("\nAdvance to hyperparameter tuning (not final selection):")
    print(tuning_details)
    print(f"Results written to: {report_dir}")
    print(f"Figures written to: {figure_dir}")
    print(f"Train-fitted candidate pipelines written to: {model_dir}")
    print("Final test partition was not accessed.")
    return result_document


def main() -> None:
    args = parse_args()
    run_baselines(
        args.split_dir.resolve(),
        args.config.resolve(),
        args.reports.resolve(),
        args.figures.resolve(),
        args.models.resolve(),
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()