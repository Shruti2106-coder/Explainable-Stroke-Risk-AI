"""Global and per-patient SHAP explanations for the finalized risk pipeline."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    from backend.ml.preprocessing import get_transformed_feature_metadata, select_model_features
except ImportError:
    from ml.preprocessing import get_transformed_feature_metadata, select_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MODEL = PROJECT_ROOT / "models" / "final_stroke_risk_pipeline.joblib"
DEFAULT_CONFIG = PROJECT_ROOT / "models" / "final_feature_config.json"
DEFAULT_METADATA = PROJECT_ROOT / "models" / "final_model_metadata.json"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "shap"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "images" / "shap"
PROBABILITY_ORDER = ("Low", "Moderate", "High")
EXPLANATION_DISCLAIMER = (
    "SHAP values describe how this model's output changes with feature inputs. "
    "They are not medical causation, diagnosis, or clinical advice."
)


def _dense_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values)


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _display_transformed_name(
    transformed_name: str,
    source_feature: str,
    categorical_features: set[str],
) -> str:
    name = transformed_name.split("__", maxsplit=1)[-1]
    missing_prefix = f"missingindicator_{source_feature}"
    if name.startswith(missing_prefix):
        return f"{source_feature} (missingness indicator)"
    category_prefix = f"{source_feature}_"
    if source_feature in categorical_features and name.startswith(category_prefix):
        category = name[len(category_prefix) :]
        return f"{source_feature} = {category}"
    return source_feature


class ShapExplanationService:
    """Load the saved pipeline once and serve global/local explanations."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        config_path: Path = DEFAULT_CONFIG,
        metadata_path: Path = DEFAULT_METADATA,
        split_dir: Path = DEFAULT_SPLIT_DIR,
        report_dir: Path = DEFAULT_REPORT_DIR,
        image_dir: Path = DEFAULT_IMAGE_DIR,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Final model pipeline not found: {model_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Final feature configuration not found: {config_path}")
        self.pipeline = joblib.load(model_path)
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model = self.pipeline.named_steps["model"]
        self.preprocessor = self.pipeline.named_steps["preprocessing"]
        self.estimator = getattr(self.model, "estimator_", self.model)
        self.feature_names = self.preprocessor.get_feature_names_out().tolist()
        self.feature_metadata = get_transformed_feature_metadata(self.preprocessor, self.config)
        self.classes = [str(value) for value in self.model.classes_]
        self.safe_features = self.config["features_safe_for_prediction"]
        self.categorical_features = set(self.config["categorical_features"])
        self.split_dir = split_dir
        self.report_dir = report_dir
        self.image_dir = image_dir
        self.explainer, self.explanation_output = self._build_explainer()

    def _build_explainer(self) -> tuple[Any, str]:
        if callable(getattr(self.estimator, "get_booster", None)):
            return shap.TreeExplainer(self.estimator), "raw_class_margin"

        train_path = self.split_dir / "train.csv"
        if not train_path.is_file():
            raise FileNotFoundError(f"Training split not found for SHAP background: {train_path}")
        background_frame = pd.read_csv(train_path, nrows=100)
        background_features = select_model_features(background_frame, self.config)
        background = _dense_array(self.preprocessor.transform(background_features))
        masker = shap.maskers.Independent(background, max_samples=min(100, len(background)))
        return (
            shap.Explainer(
                self.model.predict_proba,
                masker=masker,
                feature_names=self.feature_names,
            ),
            "class_probability",
        )

    def _transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        features = select_model_features(frame, self.config)
        transformed = _dense_array(self.preprocessor.transform(features))
        return features, transformed

    def _explain(self, transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        explanation = self.explainer(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 2:
            values = values[:, :, np.newaxis]
        if values.ndim != 3:
            raise RuntimeError(f"Expected 3D multiclass SHAP values, received {values.shape}.")
        if values.shape[1] != len(self.feature_names):
            raise RuntimeError(
                f"SHAP returned {values.shape[1]} features; expected {len(self.feature_names)}."
            )
        if values.shape[2] != len(self.classes):
            raise RuntimeError(
                f"SHAP returned {values.shape[2]} classes; expected {len(self.classes)}."
            )
        base_values = np.asarray(explanation.base_values)
        if base_values.ndim == 1:
            base_values = np.tile(base_values, (values.shape[0], 1))
        if base_values.shape != (values.shape[0], len(self.classes)):
            raise RuntimeError(f"Unexpected SHAP base-value shape: {base_values.shape}.")
        return values, base_values

    def _group_by_source(self, values: np.ndarray) -> np.ndarray:
        grouped = np.zeros((values.shape[0], len(self.safe_features), values.shape[2]))
        feature_indices = {feature: index for index, feature in enumerate(self.safe_features)}
        for transformed_index, item in enumerate(self.feature_metadata):
            original_index = feature_indices[item["source_feature"]]
            grouped[:, original_index, :] += values[:, transformed_index, :]
        return grouped

    def get_global_explanation(self, max_rows: int = 900) -> dict[str, Any]:
        if max_rows < len(self.classes):
            raise ValueError("Global explanation sample must allow at least one row per class.")
        validation_path = self.split_dir / "validation.csv"
        if not validation_path.is_file():
            raise FileNotFoundError(f"Validation split not found: {validation_path}")
        data = pd.read_csv(validation_path)
        target = self.config["target_column"]
        per_class_limit = max(1, max_rows // len(self.classes))
        samples = []
        for risk_class in self.classes:
            class_rows = data.loc[data[target].astype(str).eq(risk_class)]
            samples.append(
                class_rows.sample(
                    n=min(per_class_limit, len(class_rows)),
                    random_state=42,
                )
            )
        sample = pd.concat(samples, ignore_index=True)
        features, transformed = self._transform(sample)
        shap_values, _ = self._explain(transformed)
        grouped_values = self._group_by_source(shap_values)

        feature_importance = []
        for feature_index, feature_name in enumerate(self.safe_features):
            row: dict[str, Any] = {
                "feature": feature_name,
                "mean_abs_shap": float(np.mean(np.abs(grouped_values[:, feature_index, :]))),
            }
            for class_index, risk_class in enumerate(self.classes):
                class_values = grouped_values[:, feature_index, class_index]
                row[f"mean_abs_shap_{risk_class}"] = float(np.mean(np.abs(class_values)))
                row[f"mean_shap_{risk_class}"] = float(np.mean(class_values))
            feature_importance.append(row)
        feature_importance.sort(key=lambda row: row["mean_abs_shap"], reverse=True)

        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        importance_frame = pd.DataFrame(feature_importance)
        csv_path = self.report_dir / "global_feature_importance.csv"
        json_path = self.report_dir / "global_explanation.json"
        bar_path = self.image_dir / "global_feature_importance.png"
        summary_path = self.image_dir / "global_shap_summary.png"
        importance_frame.to_csv(csv_path, index=False)
        self._plot_global_importance(importance_frame, bar_path)
        self._plot_global_summary(shap_values, transformed, summary_path)

        result = {
            "explainer": type(self.explainer).__name__,
            "model": self.metadata["model_name"],
            "explained_split": "validation",
            "rows_explained": len(sample),
            "class_order": self.classes,
            "importance_aggregation": (
                "Mean absolute SHAP after summing one-hot/missingness dimensions back to "
                "their original approved feature, averaged across the three class outputs."
            ),
            "feature_importance": feature_importance,
            "class_specific_summary_plots": self._save_class_summaries(
                shap_values, transformed
            ),
            "artifacts": {
                "feature_importance_csv": str(csv_path),
                "feature_importance_plot": str(bar_path),
                "summary_plot": str(summary_path),
            },
            "disclaimer": EXPLANATION_DISCLAIMER,
        }
        json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        result["artifacts"]["global_explanation_json"] = str(json_path)
        return result

    def _plot_global_importance(self, importance: pd.DataFrame, output_path: Path) -> None:
        top = importance.head(20).sort_values("mean_abs_shap", ascending=True)
        figure, axis = plt.subplots(figsize=(9, max(5, len(top) * 0.3)))
        axis.barh(top["feature"], top["mean_abs_shap"], color="#5F9FB5")
        axis.set_title("Global SHAP Importance by Approved Feature", loc="left", weight="bold")
        axis.set_xlabel("Mean absolute SHAP value (aggregated across classes)")
        axis.set_ylabel("")
        axis.grid(axis="x", color="#DCE6E9", linewidth=0.8)
        axis.set_axisbelow(True)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    def _plot_global_summary(
        self,
        shap_values: np.ndarray,
        transformed: np.ndarray,
        output_path: Path,
    ) -> None:
        pooled_values = shap_values.transpose(2, 0, 1).reshape(
            -1, shap_values.shape[1]
        )
        pooled_features = np.tile(transformed, (len(self.classes), 1))
        display_names = [
            _display_transformed_name(
                item["transformed_name"],
                item["source_feature"],
                self.categorical_features,
            )
            for item in self.feature_metadata
        ]
        shap.summary_plot(
            pooled_values,
            features=pooled_features,
            feature_names=display_names,
            max_display=25,
            show=False,
            plot_size=(11, 8),
        )
        figure = plt.gcf()
        figure.suptitle(
            "Pooled Multiclass SHAP Summary (class-margin outputs)",
            x=0.02,
            ha="left",
            weight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    def _save_class_summaries(
        self,
        shap_values: np.ndarray,
        transformed: np.ndarray,
    ) -> dict[str, str]:
        output_paths: dict[str, str] = {}
        display_names = [
            _display_transformed_name(
                item["transformed_name"],
                item["source_feature"],
                self.categorical_features,
            )
            for item in self.feature_metadata
        ]
        for class_index, risk_class in enumerate(self.classes):
            path = self.image_dir / f"shap_summary_{risk_class.lower()}.png"
            shap.summary_plot(
                shap_values[:, :, class_index],
                features=transformed,
                feature_names=display_names,
                max_display=20,
                show=False,
                plot_size=(10, 7),
            )
            figure = plt.gcf()
            figure.suptitle(
                f"SHAP Summary: {risk_class} Class Output",
                x=0.02,
                ha="left",
                weight="bold",
            )
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
            plt.close(figure)
            output_paths[risk_class] = str(path)
        return output_paths

    def get_prediction_explanation(self, patient_data: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        if isinstance(patient_data, pd.DataFrame):
            if len(patient_data) != 1:
                raise ValueError("Local explanation expects exactly one patient row.")
            raw_frame = patient_data.copy()
        elif isinstance(patient_data, dict):
            raw_frame = pd.DataFrame([patient_data])
        else:
            raise TypeError("patient_data must be a dictionary or a one-row DataFrame.")

        features, transformed = self._transform(raw_frame)
        prediction = str(self.model.predict(transformed)[0])
        probabilities_array = np.asarray(self.model.predict_proba(transformed))[0]
        class_indices = {risk_class: index for index, risk_class in enumerate(self.classes)}
        probabilities = {
            risk_class: float(probabilities_array[class_indices[risk_class]])
            for risk_class in PROBABILITY_ORDER
        }
        shap_values, base_values = self._explain(transformed)
        class_index = class_indices[prediction]
        source_contributions = {feature: 0.0 for feature in self.safe_features}
        transformed_contributions = []
        for transformed_index, item in enumerate(self.feature_metadata):
            contribution = float(shap_values[0, transformed_index, class_index])
            source_feature = item["source_feature"]
            transformed_name = item["transformed_name"]
            display_name = _display_transformed_name(
                transformed_name,
                source_feature,
                self.categorical_features,
            )
            source_contributions[source_feature] += contribution
            transformed_contributions.append(
                {
                    "feature": display_name,
                    "source_feature": source_feature,
                    "transformed_feature": transformed_name,
                    "value": _json_value(transformed[0, transformed_index]),
                    "shap_value": contribution,
                }
            )

        local_contributions = [
            {
                "feature": feature,
                "value": _json_value(features.iloc[0][feature]),
                "shap_value": float(value),
                "direction": "toward_prediction" if value > 0 else "away_from_prediction",
            }
            for feature, value in source_contributions.items()
            if value != 0
        ]
        toward = sorted(
            [item for item in local_contributions if item["shap_value"] > 0],
            key=lambda item: item["shap_value"],
            reverse=True,
        )
        away = sorted(
            [item for item in local_contributions if item["shap_value"] < 0],
            key=lambda item: item["shap_value"],
        )
        expanded_toward = sorted(
            [item for item in transformed_contributions if item["shap_value"] > 0],
            key=lambda item: item["shap_value"],
            reverse=True,
        )[:15]
        expanded_away = sorted(
            [item for item in transformed_contributions if item["shap_value"] < 0],
            key=lambda item: item["shap_value"],
        )[:15]
        result = {
            "predicted_risk_class": prediction,
            "probabilities": probabilities,
            "explained_class": prediction,
            "explanation_output": self.explanation_output,
            "base_value": float(base_values[0, class_index]),
            "features_contributing_toward_prediction": toward[:15],
            "features_contributing_away_from_prediction": away[:15],
            "encoded_feature_details_toward": expanded_toward,
            "encoded_feature_details_away": expanded_away,
            "contribution_note": (
                "Positive/negative values are contributions to the selected class output "
                f"({self.explanation_output}), relative to its SHAP baseline; they are not "
                "probability changes or causal effects."
            ),
            "disclaimer": EXPLANATION_DISCLAIMER,
        }
        return result

@lru_cache(maxsize=1)
def _default_service() -> ShapExplanationService:
    return ShapExplanationService()


def get_global_explanation(max_rows: int = 900) -> dict[str, Any]:
    """Return global feature importance and save frontend-ready SHAP artifacts."""
    return _default_service().get_global_explanation(max_rows=max_rows)


def get_prediction_explanation(patient_data: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """Return one patient prediction and its local class-output SHAP contributions."""
    return _default_service().get_prediction_explanation(patient_data)