import numpy as np
import pandas as pd

from backend.ml.train_models import _rank_tuning_candidates, calculate_metrics


def test_multiclass_metrics_include_macro_weighted_and_ovr_auc():
    classes = ["High", "Low", "Moderate"]
    y_true = pd.Series(["High", "Low", "Moderate", "High", "Moderate", "Low"])
    y_pred = np.array(["High", "Moderate", "Moderate", "High", "Low", "Low"])
    scores = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
            [0.60, 0.10, 0.30],
            [0.10, 0.30, 0.60],
            [0.20, 0.70, 0.10],
        ]
    )

    metrics = calculate_metrics(y_true, y_pred, scores, classes)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["weighted_f1"] <= 1
    assert 0 <= metrics["roc_auc_ovr_macro"] <= 1
    assert 0 <= metrics["pr_auc_ovr_macro_average_precision"] <= 1
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
    assert "Low" in metrics["classification_report"]


def test_tuning_rank_prioritizes_macro_f1_then_low_recall():
    results = {
        "higher_low_recall": {
            "metrics": {
                "macro_f1": 0.80,
                "classification_report": {"Low": {"recall": 0.75}},
                "roc_auc_ovr_macro": 0.82,
                "pr_auc_ovr_macro_average_precision": 0.70,
            }
        },
        "higher_macro_f1": {
            "metrics": {
                "macro_f1": 0.81,
                "classification_report": {"Low": {"recall": 0.60}},
                "roc_auc_ovr_macro": 0.80,
                "pr_auc_ovr_macro_average_precision": 0.68,
            }
        },
    }
    assert _rank_tuning_candidates(results) == ["higher_macro_f1", "higher_low_recall"]