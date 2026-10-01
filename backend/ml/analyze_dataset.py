"""Inspect the provided stroke-risk CSV and write a data-quality/leakage report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "stroke_risk_prediction_dataset.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "dataset_analysis.md"
TARGET = "Stroke_Risk"
LEAKAGE_CANDIDATES = (
    "Stroke_Risk_Score",
    "AI_Health_Recommendation",
    "Doctor_Consultation_Needed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help="Markdown report output path"
    )
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame) -> str:
    headers = ["", *(str(column) for column in frame.columns)]
    rows = [headers]
    for row_number, index in enumerate(frame.index):
        values = [frame.iloc[row_number, column_number] for column_number in range(frame.shape[1])]
        rows.append([str(index), *(str(value) for value in values)])

    def format_row(values: list[str]) -> str:
        escaped = [value.replace("|", "\\|") for value in values]
        return "| " + " | ".join(escaped) + " |"

    separator = ["---"] * len(headers)
    return "\n".join(
        [format_row(rows[0]), format_row(separator)]
        + [format_row(row) for row in rows[1:]]
    )


def analyze(csv_path: Path, report_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV was not found: {csv_path}")

    data = pd.read_csv(csv_path)
    required_columns = {TARGET, *LEAKAGE_CANDIDATES}
    missing_required = required_columns.difference(data.columns)
    if missing_required:
        raise ValueError(
            "Dataset is missing columns required for this analysis: "
            + ", ".join(sorted(missing_required))
        )

    categorical_columns = [
        column
        for column in data.columns
        if pd.api.types.is_string_dtype(data[column].dtype)
        or pd.api.types.is_object_dtype(data[column].dtype)
        or pd.api.types.is_bool_dtype(data[column].dtype)
        or isinstance(data[column].dtype, pd.CategoricalDtype)
    ]
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    missing_counts = data.isna().sum()
    target_counts = data[TARGET].value_counts(dropna=False)
    target_percent = data[TARGET].value_counts(normalize=True, dropna=False).mul(100)
    target_distribution = pd.DataFrame(
        {"count": target_counts, "percent": target_percent.round(2)}
    )
    score_by_target = data.groupby(TARGET, dropna=False)["Stroke_Risk_Score"].agg(
        ["count", "nunique", "min", "max", "mean"]
    )
    recommendation_by_target = pd.crosstab(
        data[TARGET], data["AI_Health_Recommendation"], dropna=False
    )
    consultation_by_target = pd.crosstab(
        data[TARGET], data["Doctor_Consultation_Needed"], dropna=False
    )
    consultation_rate = pd.crosstab(
        data[TARGET],
        data["Doctor_Consultation_Needed"],
        normalize="index",
        dropna=False,
    ).mul(100).round(2)
    recommendation_class_count = (
        data.groupby("AI_Health_Recommendation", dropna=False)[TARGET]
        .nunique(dropna=False)
    )

    print(f"Dataset: {csv_path}")
    print(f"Shape: {data.shape}")
    print("\nColumn names:")
    print(data.columns.tolist())
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nFirst 5 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(data.head().to_string(index=False))
    print("\nMissing values by column (non-zero only):")
    print(missing_counts[missing_counts.gt(0)].to_string() or "None")
    print(f"\nDuplicate rows: {int(data.duplicated().sum())}")
    print("\nUnique values for categorical columns:")
    for column in categorical_columns:
        values = data[column].value_counts(dropna=False).to_dict()
        print(f"{column}: {values}")
    print(f"\n{TARGET} distribution:")
    print(target_distribution.to_string())
    print("\nStroke_Risk_Score by Stroke_Risk:")
    print(score_by_target.to_string())
    print("\nAI_Health_Recommendation by Stroke_Risk:")
    print(recommendation_by_target.to_string())
    print("\nDoctor_Consultation_Needed by Stroke_Risk:")
    print(consultation_by_target.to_string())
    print("\nConsultation percentage within each Stroke_Risk class:")
    print(consultation_rate.to_string())

    schema = pd.DataFrame(
        {
            "data_type": data.dtypes.astype(str),
            "missing_values": missing_counts,
            "unique_non_null": data.nunique(dropna=True),
        }
    )
    score_ranges_overlap = False
    score_ranges = score_by_target[["min", "max"]].dropna()
    for class_index, first_class in enumerate(score_ranges.index):
        for second_class in score_ranges.index[class_index + 1 :]:
            first_range = score_ranges.loc[first_class]
            second_range = score_ranges.loc[second_class]
            if max(first_range["min"], second_range["min"]) <= min(
                first_range["max"], second_range["max"]
            ):
                score_ranges_overlap = True

    if not score_ranges_overlap:
        score_finding = (
            "The observed score ranges do not overlap across target classes. "
            "This is strong evidence that the score encodes the class or was "
            "generated using the target; treating it as a predictor risks direct leakage."
        )
    else:
        score_finding = (
            "The observed score ranges overlap, but the score is still a strong "
            "leakage candidate because it may be used to derive the target. "
            "Its generating rule is not documented in this dataset."
        )

    recommendations_are_class_exclusive = bool(recommendation_class_count.le(1).all())
    if recommendations_are_class_exclusive:
        recommendation_finding = (
            "Each observed recommendation maps to only one target class in this CSV. "
            "It is strongly target-tied and should be excluded from predictor inputs "
            "unless its provenance is independently established."
        )
    else:
        recommendation_finding = (
            "Recommendations occur across multiple target classes, but their "
            "distribution is target-dependent and their provenance is unknown; "
            "they remain a leakage candidate."
        )

    high_class = "High"
    high_consultation_yes = None
    if high_class in consultation_by_target.index and "Yes" in consultation_by_target:
        high_row = consultation_by_target.loc[high_class]
        high_consultation_yes = int(high_row["Yes"])
    if high_consultation_yes == int(target_counts.get(high_class, -1)):
        consultation_finding = (
            "Every High-risk record is marked Yes for consultation, while the other "
            "classes have mixed values. This is a strong downstream proxy for the "
            "target and may leak target-related decision policy."
        )
    else:
        consultation_finding = (
            "Consultation status is associated with target class in the observed "
            "cross-tabulation. As a likely downstream decision, it remains a "
            "leakage candidate; its provenance should be checked before modeling."
        )

    type_summary = (
        f"{len(numeric_columns)} numeric columns and "
        f"{len(categorical_columns)} categorical/text/boolean columns"
    )
    target_classes = target_distribution.index.astype(str).tolist()
    total_missing = int(missing_counts.sum())
    columns_with_missing = int(missing_counts.gt(0).sum())
    duplicate_rows = int(data.duplicated().sum())
    source_display = (
        csv_path.relative_to(Path.cwd()).as_posix()
        if csv_path.is_relative_to(Path.cwd())
        else str(csv_path)
    )
    report = f"""# Dataset Analysis Report

## Dataset overview

- Source: `{source_display}`
- Shape: **{data.shape[0]:,} rows × {data.shape[1]} columns**
- Target: `{TARGET}`
- Feature type inventory (including the target and identifier): {type_summary}
- Duplicate rows: **{duplicate_rows:,}**
- Missing cells: **{total_missing:,}** across **{columns_with_missing}** columns

## Column inventory

{markdown_table(schema)}

## Target classes and distribution

Observed classes: {', '.join(f'`{value}`' for value in target_classes)}.

{markdown_table(target_distribution)}

The target is imbalanced: the most frequent class is `{target_distribution['count'].idxmax()}`. Any future evaluation should use stratified splitting and report per-class metrics; accuracy alone would hide minority-class behavior.

## Potential target leakage

These columns are preserved in the dataset. No columns were removed or model features selected during this analysis.

### `Stroke_Risk_Score`

{score_finding}

{markdown_table(score_by_target)}

### `AI_Health_Recommendation`

{recommendation_finding}

Recommendations by target class:

{markdown_table(recommendation_by_target)}

### `Doctor_Consultation_Needed`

{consultation_finding}

Counts by target class:

{markdown_table(consultation_by_target)}

Row percentages by target class:

{markdown_table(consultation_rate)}

## Initial observations

- The dataset has {data.shape[0]:,} records and {data.shape[1]} columns; `Patient_ID` is an identifier and must not be used as a model feature.
- {columns_with_missing} columns contain missing values ({total_missing:,} cells total); missing-data handling must be learned/applied using training data only.
- There are {duplicate_rows:,} exact duplicate rows.
- The target class distribution is uneven, with `{target_distribution['count'].idxmax()}` as the majority class and `{target_distribution['count'].idxmin()}` as the minority class.
- All three reviewed columns are plausible leakage sources. In a future modeling module, exclude them from predictors by default until their generation timing and business meaning are established. Keep the original dataset intact.
- No model was trained, no resampling or preprocessing was applied, and no performance claims are made in Module 1.

## Method and limitations

This report is computed directly from the supplied CSV using pandas. Associations and non-overlapping value ranges are evidence of target dependence, not proof of the dataset's generation process. The CSV does not document how the score, recommendation, or consultation flag was produced.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {report_path}")


def main() -> None:
    args = parse_args()
    analyze(args.csv.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()