"""Generate descriptive EDA figures and a report from the configured prepared data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from prepare_data import load_feature_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "stroke_risk_prepared.csv"
DEFAULT_CONFIG = Path(__file__).with_name("feature_config.json")
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "images" / "eda"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "exploratory_data_analysis.md"
CLASS_ORDER = ["Low", "Moderate", "High"]
RISK_PALETTE = {"Low": "#4CAF7D", "Moderate": "#E5A84B", "High": "#D96B6B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Prepared CSV path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Feature config JSON")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR, help="Figure output directory")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="EDA report path")
    return parser.parse_args()


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _grid_shape(item_count: int, columns: int = 4) -> tuple[int, int]:
    return (item_count + columns - 1) // columns, columns


def _plot_target_distribution(data: pd.DataFrame, target: str, output_dir: Path) -> None:
    counts = data[target].value_counts().reindex(CLASS_ORDER, fill_value=0)
    percentages = counts.div(counts.sum()).mul(100)
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        counts.index,
        counts.values,
        color=[RISK_PALETTE[risk_class] for risk_class in counts.index],
        width=0.62,
    )
    axis.set_title("Stroke Risk Class Distribution", loc="left", weight="bold")
    axis.set_ylabel("Records")
    axis.set_xlabel("Stroke_Risk class")
    axis.yaxis.grid(True, color="#D9E3E7", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, risk_class in zip(bars, counts.index, strict=True):
        axis.annotate(
            f"{counts[risk_class]:,}\n{percentages[risk_class]:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    save_figure(figure, output_dir / "target_distribution.png")


def _plot_numeric_distributions(
    data: pd.DataFrame, numeric_features: list[str], output_dir: Path
) -> None:
    rows, columns = _grid_shape(len(numeric_features))
    figure, axes = plt.subplots(rows, columns, figsize=(17, rows * 3.2), squeeze=False)
    for axis, feature in zip(axes.flat, numeric_features, strict=False):
        sns.histplot(data=data, x=feature, bins="auto", kde=True, ax=axis, color="#5F9FB5")
        axis.set_title(feature, loc="left", fontsize=10, weight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Records")
    for axis in axes.flat[len(numeric_features) :]:
        axis.set_visible(False)
    figure.suptitle("Safe Numeric Feature Distributions", x=0.04, ha="left", weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(figure, output_dir / "numeric_distributions.png")


def _plot_numeric_boxplots(
    data: pd.DataFrame, target: str, numeric_features: list[str], output_dir: Path
) -> None:
    rows, columns = _grid_shape(len(numeric_features))
    figure, axes = plt.subplots(rows, columns, figsize=(17, rows * 3.4), squeeze=False)
    for axis, feature in zip(axes.flat, numeric_features, strict=False):
        sns.boxplot(
            data=data,
            x=target,
            y=feature,
            order=CLASS_ORDER,
            hue=target,
            hue_order=CLASS_ORDER,
            palette=RISK_PALETTE,
            dodge=False,
            showfliers=False,
            legend=False,
            ax=axis,
        )
        axis.set_title(feature, loc="left", fontsize=10, weight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
    for axis in axes.flat[len(numeric_features) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Safe Numeric Features by Stroke_Risk (boxes show median and IQR)",
        x=0.04,
        ha="left",
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(figure, output_dir / "numeric_boxplots_by_risk.png")


def _plot_categorical_counts(
    data: pd.DataFrame, target: str, categorical_features: list[str], output_dir: Path
) -> None:
    rows, columns = _grid_shape(len(categorical_features), columns=4)
    figure, axes = plt.subplots(rows, columns, figsize=(20, rows * 3.6), squeeze=False)
    for axis, feature in zip(axes.flat, categorical_features, strict=False):
        sns.countplot(
            data=data,
            x=feature,
            hue=target,
            hue_order=CLASS_ORDER,
            palette=RISK_PALETTE,
            ax=axis,
        )
        axis.set_title(feature, loc="left", fontsize=10, weight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Records")
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        if axis.legend_ is not None:
            axis.legend_.remove()
    for axis in axes.flat[len(categorical_features) :]:
        axis.set_visible(False)
    handles = [Patch(facecolor=RISK_PALETTE[risk_class], label=risk_class) for risk_class in CLASS_ORDER]
    figure.legend(handles=handles, title=target, loc="upper right", ncol=3, frameon=False)
    figure.suptitle("Categorical Counts by Stroke_Risk", x=0.04, ha="left", weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(figure, output_dir / "categorical_counts_by_risk.png")


def _plot_categorical_composition(
    data: pd.DataFrame, target: str, categorical_features: list[str], output_dir: Path
) -> None:
    rows, columns = _grid_shape(len(categorical_features), columns=4)
    figure, axes = plt.subplots(rows, columns, figsize=(18, rows * 3.4), squeeze=False)
    for axis, feature in zip(axes.flat, categorical_features, strict=False):
        composition = pd.crosstab(data[feature], data[target], normalize="index").reindex(
            columns=CLASS_ORDER, fill_value=0
        )
        composition.plot(
            kind="barh",
            stacked=True,
            color=[RISK_PALETTE[risk_class] for risk_class in CLASS_ORDER],
            width=0.78,
            ax=axis,
            legend=False,
        )
        axis.set_title(feature, loc="left", fontsize=10, weight="bold")
        axis.set_xlabel("Share within category")
        axis.set_ylabel("")
        axis.set_xlim(0, 1)
        axis.tick_params(axis="y", labelsize=7)
    for axis in axes.flat[len(categorical_features) :]:
        axis.set_visible(False)
    handles = [Patch(facecolor=RISK_PALETTE[risk_class], label=risk_class) for risk_class in CLASS_ORDER]
    figure.legend(handles=handles, title=target, loc="upper right", ncol=3, frameon=False)
    figure.suptitle(
        "Stroke_Risk Composition Within Each Category (descriptive only)",
        x=0.04,
        ha="left",
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(figure, output_dir / "categorical_risk_composition.png")


def _plot_correlation(data: pd.DataFrame, numeric_features: list[str], output_dir: Path) -> None:
    correlation = data[numeric_features].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.35,
        cbar_kws={"label": "Spearman correlation"},
        ax=axis,
    )
    axis.set_title("Spearman Correlation: Safe Numeric Features", loc="left", weight="bold")
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    save_figure(figure, output_dir / "numeric_correlation_heatmap.png")
    correlation.to_csv(output_dir / "numeric_spearman_correlation.csv")


def _plot_age_groups(data: pd.DataFrame, target: str, output_dir: Path) -> pd.DataFrame:
    age_group_column = "Age_Group"
    age_groups = pd.cut(
        data["Age"],
        bins=[-float("inf"), 30, 45, 60, 75, float("inf")],
        labels=["<=30", "31-45", "46-60", "61-75", ">75"],
    )
    composition = pd.crosstab(age_groups, data[target], normalize="index").reindex(
        columns=CLASS_ORDER, fill_value=0
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    composition.plot(
        kind="bar",
        stacked=True,
        color=[RISK_PALETTE[risk_class] for risk_class in CLASS_ORDER],
        ax=axis,
    )
    axis.set_title("Stroke_Risk Composition by Age Group", loc="left", weight="bold")
    axis.set_xlabel("Age group (descriptive bins)")
    axis.set_ylabel("Share within age group")
    axis.set_ylim(0, 1)
    axis.legend(title=target, frameon=False, ncol=3)
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    save_figure(figure, output_dir / "risk_composition_by_age_group.png")
    composition.index.name = age_group_column
    return composition


def _numeric_summaries(
    data: pd.DataFrame, numeric_features: list[str], target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = data[numeric_features].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.rename(
        columns={"25%": "q1", "50%": "median", "75%": "q3", "std": "std_dev"}
    )
    summary["missing"] = data[numeric_features].isna().sum()
    grouped_rows = []
    for target_class, group in data.groupby(target, observed=False):
        for feature in numeric_features:
            values = group[feature]
            grouped_rows.append(
                {
                    target: target_class,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                }
            )
    by_target = pd.DataFrame(grouped_rows).set_index([target, "feature"])
    return summary, by_target


def _categorical_summaries(
    data: pd.DataFrame, categorical_features: list[str], target: str
) -> pd.DataFrame:
    parts = []
    for feature in categorical_features:
        counts = pd.crosstab(data[feature], data[target], dropna=False).reindex(
            columns=CLASS_ORDER, fill_value=0
        )
        percentages = pd.crosstab(
            data[feature], data[target], normalize="index", dropna=False
        ).reindex(columns=CLASS_ORDER, fill_value=0).mul(100)
        for category in counts.index:
            row = {"feature": feature, "category": category, "count": int(counts.loc[category].sum())}
            for risk_class in CLASS_ORDER:
                row[f"{risk_class}_count"] = int(counts.loc[category, risk_class])
                row[f"{risk_class}_percent"] = float(percentages.loc[category, risk_class])
            parts.append(row)
    return pd.DataFrame(parts)


def _write_report(
    data: pd.DataFrame,
    config: dict,
    numeric_summary: pd.DataFrame,
    numeric_by_target: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    age_composition: pd.DataFrame,
    report_path: Path,
) -> None:
    target = config["target_column"]
    counts = data[target].value_counts().reindex(CLASS_ORDER, fill_value=0)
    percentages = counts.div(counts.sum()).mul(100)
    target_table = pd.DataFrame({"records": counts, "percent": percentages.round(2)})
    age_median = float(data["Age"].median())
    age_q1, age_q3 = data["Age"].quantile([0.25, 0.75]).tolist()
    bmi_medians = data.groupby(target, observed=False)["BMI"].median().reindex(CLASS_ORDER)
    age_medians = data.groupby(target, observed=False)["Age"].median().reindex(CLASS_ORDER)
    age_non_missing = int(data["Age"].notna().sum())
    bp_medians = data[["Blood_Pressure_Systolic", "Blood_Pressure_Diastolic"]].median()
    glucose_median = float(data["Blood_Glucose"].median())
    hba1c_median = float(data["HbA1c"].median())
    cholesterol_medians = data[
        ["Total_Cholesterol", "HDL", "LDL", "Triglycerides"]
    ].median()
    missing_cells = int(data[config["features_safe_for_prediction"]].isna().sum().sum())
    missing_features = int(data[config["features_safe_for_prediction"]].isna().any().sum())
    gender_table = categorical_summary.loc[categorical_summary["feature"].eq("Gender")]
    smoking_table = categorical_summary.loc[categorical_summary["feature"].eq("Smoking_Status")]
    diabetes_table = categorical_summary.loc[categorical_summary["feature"].eq("Diabetes")]
    conditions = categorical_summary.loc[
        categorical_summary["feature"].isin(
            ["Diabetes", "Hypertension", "Heart_Disease", "Previous_TIA", "Atrial_Fibrillation"]
        )
    ]
    family_history = categorical_summary.loc[
        categorical_summary["feature"].isin(["Family_History_Stroke", "Family_History_Heart_Disease"])
    ]

    def table_for(frame: pd.DataFrame, columns: list[str]) -> str:
        rows = [columns]
        for values in frame.loc[:, columns].itertuples(index=False, name=None):
            formatted = []
            for value in values:
                if pd.isna(value):
                    formatted.append("")
                elif isinstance(value, (float,)): 
                    formatted.append(f"{value:.2f}")
                else:
                    formatted.append(str(value))
            rows.append(formatted)

        def format_row(values: list[str]) -> str:
            escaped = [value.replace("|", "\\|") for value in values]
            return "| " + " | ".join(escaped) + " |"

        return "\n".join(
            [format_row(rows[0]), format_row(["---"] * len(columns))]
            + [format_row(row) for row in rows[1:]]
        )

    observations = [
        f"- The prepared target contains {len(data):,} records: "
        + ", ".join(f"{risk_class} {counts[risk_class]:,} ({percentages[risk_class]:.2f}%)" for risk_class in CLASS_ORDER)
        + ". The class counts are visibly uneven.",
        f"- Age has {age_non_missing:,} observed values, median {age_median:.1f} years, and interquartile range {age_q1:.1f}-{age_q3:.1f}. Class-specific age medians are "
        + ", ".join(f"{risk_class} {age_medians[risk_class]:.1f}" for risk_class in CLASS_ORDER)
        + ". These are descriptive differences, not causal evidence.",
        "- BMI class medians are "
        + ", ".join(f"{risk_class} {bmi_medians[risk_class]:.1f}" for risk_class in CLASS_ORDER)
        + "; the boxplots show the spread and overlap around those medians.",
        f"- Median systolic/diastolic readings are {bp_medians['Blood_Pressure_Systolic']:.1f}/{bp_medians['Blood_Pressure_Diastolic']:.1f}. The blood-pressure panels show each distribution by target class without implying an effect.",
        f"- Median blood glucose is {glucose_median:.1f} and median HbA1c is {hba1c_median:.1f}; median total cholesterol, HDL, LDL, and triglycerides are "
        + ", ".join(f"{feature} {cholesterol_medians[feature]:.1f}" for feature in cholesterol_medians.index)
        + ". The comparison figure reports group spread as well as medians.",
        f"- {missing_cells:,} missing values remain in {missing_features} safe features in the prepared data. EDA excludes missing entries only from the individual statistic or plot that needs them.",
        "- Count plots show the number of records in each observed categorical level, split by Stroke_Risk. The composition plot normalizes within each category so class proportions can be compared; neither figure supports causal interpretation.",
    ]
    gender_note = "Gender target composition: " + "; ".join(
        f"{row['category']}: Low {row['Low_percent']:.1f}%, Moderate {row['Moderate_percent']:.1f}%, High {row['High_percent']:.1f}%"
        for _, row in gender_table.iterrows()
    )
    smoking_note = "Smoking-status target composition: " + "; ".join(
        f"{row['category']}: Low {row['Low_percent']:.1f}%, Moderate {row['Moderate_percent']:.1f}%, High {row['High_percent']:.1f}%"
        for _, row in smoking_table.iterrows()
    )
    diabetes_note = "Diabetes target composition: " + "; ".join(
        f"{row['category']}: Low {row['Low_percent']:.1f}%, Moderate {row['Moderate_percent']:.1f}%, High {row['High_percent']:.1f}%"
        for _, row in diabetes_table.iterrows()
    )
    figure_descriptions = [
        "`images/eda/target_distribution.png`: class counts and percentages in the prepared dataset.",
        "`images/eda/numeric_distributions.png`: histograms and KDE overlays for all configured safe numeric features; empty/missing entries are omitted per panel.",
        "`images/eda/numeric_boxplots_by_risk.png`: median, interquartile range, and observed spread of every safe numeric feature split by target class; outlier markers are suppressed for readability, not removed from data.",
        "`images/eda/categorical_counts_by_risk.png`: counts for every configured categorical feature, split by target class.",
        "`images/eda/categorical_risk_composition.png`: within-category proportions of Low, Moderate, and High target records for every categorical feature.",
        "`images/eda/numeric_correlation_heatmap.png`: pairwise Spearman correlations among configured safe numeric features; the categorical target is not ordinal-encoded.",
        "`images/eda/risk_composition_by_age_group.png`: target-class proportions for descriptive age bins (<=30, 31-45, 46-60, 61-75, >75).",
    ]
    report = f"""# Exploratory Data Analysis

## Scope

EDA uses the prepared dataset from Module 2 and the persisted feature policy in `backend/ml/feature_config.json`. It includes only configured safe features plus `{target}`; `Patient_ID` and the three leakage candidates remain excluded. The source CSV is not modified. No model is trained.

## Dataset and target

- Prepared rows: **{len(data):,}**
- Safe features: **{len(config['features_safe_for_prediction'])}** ({len(config['numeric_features'])} numeric, {len(config['categorical_features'])} categorical)
- Target: `{target}`
- Missing safe-feature values retained: **{missing_cells:,}** across **{missing_features}** features

{table_for(target_table.rename_axis(target).reset_index(), [target, "records", "percent"])}

## Main observations

{chr(10).join(observations)}

{gender_note}

{smoking_note}

{diabetes_note}

### Selected condition and family-history group composition

The following table gives the percentage distribution of Stroke_Risk within each category. It is descriptive and does not establish causation.

{table_for(conditions, ["feature", "category", "count", "Low_percent", "Moderate_percent", "High_percent"])}

Family-history category composition:

{table_for(family_history, ["feature", "category", "count", "Low_percent", "Moderate_percent", "High_percent"])}

### Age-group composition

{table_for(age_composition.reset_index(), ["Age_Group", "Low", "Moderate", "High"])}

## Numerical summary

Quartiles and central tendency for each configured safe numeric feature:

{table_for(numeric_summary.reset_index(names="feature"), ["feature", "count", "mean", "std_dev", "min", "q1", "median", "q3", "max", "missing"])}

Numeric feature summaries grouped by Stroke_Risk are also exported to `reports/eda_numeric_by_risk.csv`.

## Figures

{chr(10).join(f'- {description}' for description in figure_descriptions)}

The Spearman matrix is also saved as `images/eda/numeric_spearman_correlation.csv`.

## Method and limitations

All findings describe this prepared dataset only. Group differences and correlations are associations, not causal medical claims. Numerical plots use available values for each feature; this means per-feature sample sizes can differ because some fields are missing. No significance testing, predictive validation, or medical interpretation was performed.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def run_eda(data_path: Path, config_path: Path, image_dir: Path, report_path: Path) -> None:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset not found: {data_path}. Run backend/ml/prepare_data.py first."
        )
    config = load_feature_config(config_path)
    data = pd.read_csv(data_path)
    target = config["target_column"]
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    required = set(config["features_safe_for_prediction"]) | {target}
    if set(data.columns) != required:
        missing = required.difference(data.columns)
        unexpected = set(data.columns).difference(required)
        raise ValueError(f"Prepared CSV/config mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    absent_classes = set(CLASS_ORDER).difference(data[target].dropna().astype(str).unique())
    if absent_classes:
        raise ValueError(f"Prepared data is missing target classes: {sorted(absent_classes)}")

    image_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.82)
    _plot_target_distribution(data, target, image_dir)
    _plot_numeric_distributions(data, numeric_features, image_dir)
    _plot_numeric_boxplots(data, target, numeric_features, image_dir)
    _plot_categorical_counts(data, target, categorical_features, image_dir)
    _plot_categorical_composition(data, target, categorical_features, image_dir)
    _plot_correlation(data, numeric_features, image_dir)
    age_composition = _plot_age_groups(data, target, image_dir)

    numeric_summary, numeric_by_target = _numeric_summaries(data, numeric_features, target)
    categorical_summary = _categorical_summaries(data, categorical_features, target)
    numeric_summary.to_csv(report_path.parent / "eda_numeric_summary.csv")
    numeric_by_target.to_csv(report_path.parent / "eda_numeric_by_risk.csv")
    categorical_summary.to_csv(report_path.parent / "eda_categorical_by_risk.csv", index=False)
    age_composition.to_csv(report_path.parent / "eda_age_group_by_risk.csv")
    _write_report(
        data,
        config,
        numeric_summary,
        numeric_by_target,
        categorical_summary,
        age_composition,
        report_path,
    )

    print(f"Prepared rows analyzed: {len(data):,}")
    print(f"Safe numeric features: {len(numeric_features)}")
    print(f"Safe categorical features: {len(categorical_features)}")
    print("Target distribution:")
    print(data[target].value_counts().reindex(CLASS_ORDER).to_string())
    print(f"Figures written to: {image_dir}")
    print(f"EDA report written to: {report_path}")


def main() -> None:
    args = parse_args()
    run_eda(args.data.resolve(), args.config.resolve(), args.images.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()