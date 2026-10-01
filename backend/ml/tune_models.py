"""Tune baseline multiclass candidates with CV on train, select on validation, test once."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    from .estimators import BalancedXGBClassifier
    from .preprocessing import build_preprocessor, load_feature_config, select_model_features
    from .train_models import (
        _aligned_continuous_scores,
        _save_confusion_matrix,
        _save_roc_and_pr_curves,
        calculate_metrics,
    )
except ImportError:
    from estimators import BalancedXGBClassifier
    from preprocessing import build_preprocessor, load_feature_config, select_model_features
    from train_models import (
        _aligned_continuous_scores,
        _save_confusion_matrix,
        _save_roc_and_pr_curves,
        calculate_metrics,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_CONFIG = Path(__file__).with_name("feature_config.json")
DEFAULT_BASELINE_RESULTS = PROJECT_ROOT / "reports" / "model_evaluation.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "images" / "model_tuning"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
CLASS_ORDER = ["Low", "Moderate", "High"]
RISK_COLORS = {"Low": "#4CAF7D", "Moderate": "#E5A84B", "High": "#D96B6B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--refresh-validation-comparison-only",
        action="store_true",
        help="Regenerate baseline-vs-tuned validation artifacts from saved JSON without reading any data split.",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Reuse this run's existing CV CSVs and tuned pipelines for completed families.",
    )
    return parser.parse_args()


def _search_specs(
    random_state: int,
) -> dict[str, tuple[str, Pipeline, list[dict[str, list[Any]]], bool]]:
    logistic = Pipeline(
        [
            ("preprocessing", "passthrough"),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("preprocessing", "passthrough"),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    xgboost = Pipeline(
        [
            ("preprocessing", "passthrough"),
            ("model", BalancedXGBClassifier(random_state=random_state)),
        ]
    )
    svm = Pipeline(
        [
            ("preprocessing", "passthrough"),
            (
                "model",
                SVC(class_weight="balanced", decision_function_shape="ovr"),
            ),
        ]
    )
    return {
        "logistic_regression": (
            "Logistic Regression",
            logistic,
            [
                {
                    "model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
                    "model__solver": ["lbfgs", "saga"],
                    "model__class_weight": ["balanced", None],
                }
            ],
            True,
        ),
        "random_forest": (
            "Random Forest",
            forest,
            [
                {
                    "model__n_estimators": [200, 300, 450],
                    "model__max_depth": [None, 14, 20, 28],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", 0.6, 0.9],
                }
            ],
            False,
        ),
        "xgboost": (
            "XGBoost",
            xgboost,
            [
                {
                    "model__n_estimators": [150, 250, 350, 450],
                    "model__max_depth": [3, 4, 5, 7],
                    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.7, 0.85, 1.0],
                }
            ],
            False,
        ),
        "svm": (
            "SVM",
            svm,
            [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
                    "model__gamma": ["scale", "auto", 0.01, 0.1],
                },
            ],
            True,
        ),
    }


def _make_pipeline(
    estimator_pipeline: Pipeline,
    config: dict[str, Any],
    scale_numeric: bool,
) -> Pipeline:
    pipeline = clone(estimator_pipeline)
    pipeline.set_params(
        preprocessing=build_preprocessor(config, scale_numeric=scale_numeric)
    )
    return pipeline


def _tuning_rank(results: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        results,
        key=lambda key: (
            -results[key]["validation_metrics"]["macro_f1"],
            -results[key]["validation_metrics"]["classification_report"]["Low"]["recall"],
            -results[key]["validation_metrics"]["roc_auc_ovr_macro"],
            -results[key]["validation_metrics"]["pr_auc_ovr_macro_average_precision"],
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


def _save_validation_comparison(
    baseline: dict[str, Any],
    tuned: dict[str, dict[str, Any]],
    report_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    rows = []
    metric_keys = [
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
    for key, result in tuned.items():
        baseline_key = "linear_svm" if key == "svm" else key
        baseline_result = baseline["models"].get(baseline_key)
        for stage, metric_values in (
            ("baseline_validation", baseline_result["metrics"] if baseline_result else {}),
            ("tuned_validation", result["validation_metrics"]),
        ):
            rows.append(
                {
                    "model_key": key,
                    "model": result["display_name"],
                    "stage": stage,
                    **{metric: metric_values.get(metric, np.nan) for metric in metric_keys},
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(report_dir / "baseline_vs_tuned_validation.csv", index=False)
    figure, axis = plt.subplots(figsize=(12, 6))
    metric = "macro_f1"
    sns.barplot(
        data=comparison,
        x="model",
        y=metric,
        hue="stage",
        palette={"baseline_validation": "#A9BAC0", "tuned_validation": "#5F9FB5"},
        ax=axis,
    )
    axis.set_title("Baseline vs Tuned Macro-F1 on Validation Data", loc="left", weight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=12)
    axis.legend(title="", frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "baseline_vs_tuned_macro_f1.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return comparison


def refresh_validation_comparison(
    baseline_results_path: Path,
    tuning_results_path: Path,
    report_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    """Regenerate validation-only baseline/tuned comparisons from saved metrics."""
    baseline = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    tuning = json.loads(tuning_results_path.read_text(encoding="utf-8"))
    tuned = tuning["all_tuned_models"]
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return _save_validation_comparison(baseline, tuned, report_dir, figure_dir)


def _best_cv_rows(search: RandomizedSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    return results.sort_values("rank_test_macro_f1").reset_index(drop=True)


def run_tuning(
    split_dir: Path,
    config_path: Path,
    baseline_results_path: Path,
    report_dir: Path,
    figure_dir: Path,
    model_dir: Path,
    *,
    random_state: int = RANDOM_STATE,
    cv_folds: int = 3,
    iterations: int = 8,
    resume_completed: bool = False,
) -> dict[str, Any]:
    train_path = split_dir / "train.csv"
    validation_path = split_dir / "validation.csv"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Run backend/ml/split_data.py before hyperparameter tuning.")
    if not baseline_results_path.is_file():
        raise FileNotFoundError(f"Module 6 baseline results not found: {baseline_results_path}")
    if cv_folds < 2 or iterations < 1:
        raise ValueError("Use at least two CV folds and one randomized-search iteration.")

    config = load_feature_config(config_path)
    target = config["target_column"]
    classes = [str(value) for value in config["target_classes"]]
    expected_columns = set(config["features_safe_for_prediction"]) | {target}
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    if set(train.columns) != expected_columns or set(validation.columns) != expected_columns:
        raise ValueError("Train/validation columns do not match the approved feature configuration.")
    y_train = train[target].astype(str)
    y_validation = validation[target].astype(str)
    if set(y_train.unique()) != set(classes) or set(y_validation.unique()) != set(classes):
        raise ValueError("Train and validation partitions must contain all configured classes.")
    X_train = select_model_features(train, config)
    X_validation = select_model_features(validation, config)
    baseline = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    if baseline.get("final_test_accessed") is not False:
        raise ValueError("Module 6 baseline metadata does not confirm final test isolation.")

    scorer = {
        "macro_f1": "f1_macro",
        "low_recall": make_scorer(
            recall_score,
            labels=["Low"],
            average="macro",
            zero_division=0,
        ),
    }
    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tuned_results: dict[str, dict[str, Any]] = {}

    for model_key, (display_name, estimator_pipeline, parameter_space, scale_numeric) in _search_specs(
        random_state
    ).items():
        print(
            f"Tuning {display_name}: {iterations} randomized candidates × "
            f"{cv_folds}-fold stratified CV",
            flush=True,
        )
        pipeline = _make_pipeline(estimator_pipeline, config, scale_numeric)
        cv_result_path = report_dir / f"cv_results_{model_key}.csv"
        tuned_path = model_dir / f"{model_key}_tuned.joblib"
        resumed = (
            resume_completed
            and model_key != "svm"
            and cv_result_path.is_file()
            and tuned_path.is_file()
        )
        if resumed:
            cv_rows = pd.read_csv(cv_result_path).sort_values("rank_test_macro_f1")
            best_row = cv_rows.iloc[0]
            best_params = ast.literal_eval(best_row["params"])
            best_cv_macro_f1 = float(best_row["mean_test_macro_f1"])
            best_cv_low_recall = float(best_row["mean_test_low_recall"])
            best_pipeline = joblib.load(tuned_path)
            fit_seconds = 0.0
            search_train_rows = len(X_train)
            print(f"  Resumed completed CV results from {cv_result_path.name}", flush=True)
        else:
            search_X = X_train
            search_y = y_train
            search_train_rows = len(X_train)
            if model_key == "svm":
                search_train_rows = min(6000, len(X_train))
                search_X, _, search_y, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=search_train_rows,
                    stratify=y_train,
                    random_state=random_state,
                )
                print(
                    f"  SVM CV uses a stratified subset of {search_train_rows:,} training rows; "
                    "best parameters will be refit on all training rows.",
                    flush=True,
                )
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameter_space,
                n_iter=iterations,
                scoring=scorer,
                refit="macro_f1",
                cv=cross_validation,
                n_jobs=1,
                random_state=random_state,
                return_train_score=False,
                error_score="raise",
                verbose=0,
            )
            started = time.perf_counter()
            search.fit(search_X, search_y)
            fit_seconds = time.perf_counter() - started
            best_params = search.best_params_
            best_cv_macro_f1 = float(search.best_score_)
            best_cv_low_recall = float(
                search.cv_results_["mean_test_low_recall"][search.best_index_]
            )
            cv_rows = _best_cv_rows(search)
            cv_rows.to_csv(cv_result_path, index=False)
            best_pipeline = clone(pipeline).set_params(**best_params)
            refit_started = time.perf_counter()
            best_pipeline.fit(X_train, y_train)
            fit_seconds += time.perf_counter() - refit_started

        validation_predictions = best_pipeline.predict(X_validation)
        validation_scores, score_source = _aligned_continuous_scores(
            best_pipeline,
            X_validation,
            classes,
        )
        validation_metrics = calculate_metrics(
            y_validation,
            validation_predictions,
            validation_scores,
            classes,
        )
        validation_metrics["score_source"] = score_source
        joblib.dump(best_pipeline, tuned_path)
        tuned_results[model_key] = {
            "display_name": display_name,
            "best_params": best_params,
            "best_cv_macro_f1": best_cv_macro_f1,
            "best_cv_low_recall": best_cv_low_recall,
            "search_train_rows": search_train_rows,
            "validation_metrics": validation_metrics,
            "fit_seconds": fit_seconds,
            "best_pipeline_path": str(tuned_path),
            "classification_report": validation_metrics["classification_report"],
            "confusion_matrix": validation_metrics["confusion_matrix"],
        }
        _save_confusion_matrix(
            validation_metrics["confusion_matrix"],
            classes,
            f"Tuned {display_name} (validation)",
            figure_dir / f"{model_key}_tuned_validation_confusion_matrix.png",
        )
        _save_roc_and_pr_curves(
            y_validation,
            validation_scores,
            classes,
            f"Tuned {display_name} (validation)",
            figure_dir / f"{model_key}_tuned_validation",
        )
        print(
            f"  best CV macro-F1={best_cv_macro_f1:.4f}; "
            f"validation macro-F1={validation_metrics['macro_f1']:.4f}; "
            f"validation Low recall="
            f"{validation_metrics['classification_report']['Low']['recall']:.4f}; "
            f"seconds={fit_seconds:.1f}"
        )

    comparison = _save_validation_comparison(
        baseline,
        tuned_results,
        report_dir,
        figure_dir,
    )
    validation_ranking = sorted(
        tuned_results,
        key=lambda key: (
            -tuned_results[key]["validation_metrics"]["macro_f1"],
            -tuned_results[key]["validation_metrics"]["classification_report"]["Low"]["recall"],
            -tuned_results[key]["validation_metrics"]["roc_auc_ovr_macro"],
            -tuned_results[key]["validation_metrics"]["pr_auc_ovr_macro_average_precision"],
        ),
    )
    selected_key = validation_ranking[0]
    selected_result = tuned_results[selected_key]
    selected_tuned_pipeline = joblib.load(selected_result["best_pipeline_path"])

    # Final model is selected using validation only. Refit that configuration on train+validation.
    development_data = pd.concat([train, validation], ignore_index=True)
    X_development = select_model_features(development_data, config)
    y_development = development_data[target].astype(str)
    final_pipeline = clone(selected_tuned_pipeline)
    final_pipeline.fit(X_development, y_development)

    # The locked test is first opened only after CV tuning and validation-based model selection.
    test_path = split_dir / "final_test_locked.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"Locked final-test file not found: {test_path}")
    final_test = pd.read_csv(test_path)
    if set(final_test.columns) != expected_columns:
        raise ValueError("Final test columns do not match the approved feature configuration.")
    y_test = final_test[target].astype(str)
    if set(y_test.unique()) != set(classes):
        raise ValueError("Final test partition must contain all configured classes.")
    X_test = select_model_features(final_test, config)
    test_predictions = final_pipeline.predict(X_test)
    test_scores, test_score_source = _aligned_continuous_scores(
        final_pipeline,
        X_test,
        classes,
    )
    test_metrics = calculate_metrics(y_test, test_predictions, test_scores, classes)
    test_metrics["score_source"] = test_score_source
    final_model_path = model_dir / "final_stroke_risk_pipeline.joblib"
    preprocessor_path = model_dir / "final_preprocessing_pipeline.joblib"
    config_copy_path = model_dir / "final_feature_config.json"
    metadata_path = model_dir / "final_model_metadata.json"
    joblib.dump(final_pipeline, final_model_path)
    joblib.dump(final_pipeline.named_steps["preprocessing"], preprocessor_path)
    shutil.copyfile(config_path, config_copy_path)
    metadata = {
        "model_name": selected_result["display_name"],
        "model_key": selected_key,
        "selection_split": "validation",
        "selection_rule": (
            "Among tuned models, maximize validation macro-F1; break ties using Low-class recall, "
            "macro one-vs-rest ROC-AUC, then macro one-vs-rest average precision. The final test "
            "was not used to choose the model."
        ),
        "target_column": target,
        "class_order": classes,
        "approved_features": config["features_safe_for_prediction"],
        "excluded_leakage_features": config["features_excluded_due_to_leakage"],
        "identifier_columns": config["identifier_columns"],
        "best_params": selected_result["best_params"],
        "best_cv_macro_f1": selected_result["best_cv_macro_f1"],
        "validation_metrics": selected_result["validation_metrics"],
        "final_test_metrics": test_metrics,
        "fit_rows": len(development_data),
        "final_test_rows": len(final_test),
        "final_test_accessed_only_after_selection": True,
        "random_state": random_state,
        "saved_pipeline": str(final_model_path),
        "saved_preprocessor": str(preprocessor_path),
        "saved_feature_config": str(config_copy_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "tuning_results.json").write_text(
        json.dumps(
            {
                "cross_validation": {
                    "splitter": "StratifiedKFold",
                    "n_splits": cv_folds,
                    "shuffle": True,
                    "random_state": random_state,
                    "scoring": ["macro_f1", "Low_class_recall"],
                    "refit_metric": "macro_f1",
                    "final_test_accessed_during_tuning": False,
                },
                "validation_ranking": [
                    {
                        "model_key": key,
                        "model": tuned_results[key]["display_name"],
                        "best_params": tuned_results[key]["best_params"],
                        "best_cv_macro_f1": tuned_results[key]["best_cv_macro_f1"],
                        "search_train_rows": tuned_results[key]["search_train_rows"],
                        "validation_metrics": tuned_results[key]["validation_metrics"],
                    }
                    for key in validation_ranking
                ],
                "selected_model_key": selected_key,
                "selected_model": selected_result["display_name"],
                "final_test_metrics": test_metrics,
                "all_tuned_models": tuned_results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "final_test_classification_report.json").write_text(
        json.dumps(test_metrics["classification_report"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    comparison.to_csv(report_dir / "baseline_vs_tuned_validation.csv", index=False)

    _save_confusion_matrix(
        test_metrics["confusion_matrix"],
        classes,
        f"Final {selected_result['display_name']} (test)",
        figure_dir / "final_test_confusion_matrix.png",
    )
    _save_roc_and_pr_curves(
        y_test,
        test_scores,
        classes,
        f"Final {selected_result['display_name']} (test)",
        figure_dir / "final_test",
    )

    candidate_lines = "\n".join(
        f"{position}. **{tuned_results[key]['display_name']}**: CV macro-F1 "
        f"{tuned_results[key]['best_cv_macro_f1']:.4f}; validation macro-F1 "
        f"{tuned_results[key]['validation_metrics']['macro_f1']:.4f}; validation Low recall "
        f"{tuned_results[key]['validation_metrics']['classification_report']['Low']['recall']:.4f}."
        for position, key in enumerate(validation_ranking, start=1)
    )
    baseline_tuned_rows = comparison[
        comparison["model_key"].eq(selected_key)
    ][["stage", "macro_f1", "macro_recall", "weighted_f1", "roc_auc_ovr_macro"]]
    report = f"""# Hyperparameter Tuning and Final Evaluation

## Data boundaries

- RandomizedSearchCV used only `data/splits/train.csv`, with `StratifiedKFold(n_splits={cv_folds}, shuffle=True, random_state={random_state})`.
- Search scoring used macro-F1 as the refit objective and Low-class recall as a secondary recorded metric.
- Preprocessing stayed inside each CV estimator pipeline; imputation, scaling, and encoding were learned within each training fold.
- `data/splits/validation.csv` was used after tuning to compare tuned models and select the final configuration.
- The locked test file was first opened only after final model selection, and was evaluated once for that selected pipeline. It did not affect tuning or model choice.
- No oversampling or synthetic examples were used in these searches. Class weights are applied by estimator; XGBoost calculates balanced sample weights inside each CV `fit` call.

## Hyperparameter spaces

- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
- XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
- Logistic Regression: `C`, `solver`, and `class_weight`.
- SVM: `C`, `kernel` (linear/RBF), and `gamma` for RBF.

Each family evaluated {iterations} randomized configurations using {cv_folds}-fold stratified CV. The SVM search uses a seeded, stratified subset of up to 6,000 training rows because full-data RBF SVC CV exceeded practical runtime; the best SVM parameters are refit on all training rows before validation. Other searches use all training rows. Full candidate scores and parameters are in `cv_results_*.csv` and `reports/tuning_results.json`.

## Tuned validation ranking

Models are ranked by validation macro-F1, with Low recall, macro OVR ROC-AUC, and macro OVR average precision as tie-breakers. The final model was selected from validation results before opening the test split.

{candidate_lines}

### Baseline vs tuned validation: selected family

{_markdown_table(baseline_tuned_rows)}

## Selected model

Validation selected **{selected_result['display_name']}** with macro-F1 {selected_result['validation_metrics']['macro_f1']:.4f}. Its hyperparameters are saved in metadata. The final estimator pipeline was refit on train+validation ({len(development_data):,} rows), then evaluated once on the untouched test partition ({len(final_test):,} rows).

## Final test metrics

| Metric | Value |
| --- | ---: |
| Accuracy | {test_metrics['accuracy']:.4f} |
| Macro precision | {test_metrics['macro_precision']:.4f} |
| Macro recall | {test_metrics['macro_recall']:.4f} |
| Macro F1 | {test_metrics['macro_f1']:.4f} |
| Weighted precision | {test_metrics['weighted_precision']:.4f} |
| Weighted recall | {test_metrics['weighted_recall']:.4f} |
| Weighted F1 | {test_metrics['weighted_f1']:.4f} |
| Macro OVR ROC-AUC | {test_metrics['roc_auc_ovr_macro']:.4f} |
| Macro OVR PR-AUC (average precision) | {test_metrics['pr_auc_ovr_macro_average_precision']:.4f} |

Class-wise results and confusion matrix are in `reports/final_test_classification_report.json` and `reports/tuning_results.json`. Final test curves and confusion matrix are saved under `images/model_tuning/`.

## Saved artifacts

- Final estimator + preprocessing pipeline: `{final_model_path.name}`
- Standalone fitted preprocessing transformer: `{preprocessor_path.name}`
- Feature policy snapshot: `{config_copy_path.name}`
- Model metadata and test metrics: `{metadata_path.name}`
- Tuned validation pipelines: `models/*_tuned.joblib`

The test metrics are a one-time final estimate for the selected model. Do not retune or change selection based on these test results.
"""
    (report_dir / "hyperparameter_tuning_report.md").write_text(report, encoding="utf-8")

    print("\nTuned validation ranking:")
    for position, key in enumerate(validation_ranking, start=1):
        result = tuned_results[key]
        print(
            f"{position}. {result['display_name']}: CV macro-F1="
            f"{result['best_cv_macro_f1']:.4f}, validation macro-F1="
            f"{result['validation_metrics']['macro_f1']:.4f}, params={result['best_params']}"
        )
    print(f"\nSelected on validation: {selected_result['display_name']}")
    print("Final test metrics (one evaluation):")
    print({key: round(value, 4) for key, value in test_metrics.items() if isinstance(value, float)})
    print(f"Final model: {final_model_path}")
    print(f"Metadata: {metadata_path}")
    return metadata


def main() -> None:
    args = parse_args()
    if args.refresh_validation_comparison_only:
        refresh_validation_comparison(
            args.baseline_results.resolve(),
            (args.reports / "tuning_results.json").resolve(),
            args.reports.resolve(),
            args.figures.resolve(),
        )
        print("Validation comparison refreshed from saved reports; no data partitions were opened.")
        return
    run_tuning(
        args.split_dir.resolve(),
        args.config.resolve(),
        args.baseline_results.resolve(),
        args.reports.resolve(),
        args.figures.resolve(),
        args.models.resolve(),
        random_state=args.seed,
        cv_folds=args.cv_folds,
        iterations=args.iterations,
        resume_completed=args.resume_completed,
    )


if __name__ == "__main__":
    main()