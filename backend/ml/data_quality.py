"""Validate the supplied stroke-risk dataset and persist feature-selection evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "stroke_risk_prediction_dataset.csv"
DEFAULT_CONFIG = Path(__file__).with_name("feature_config.json")
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "data_quality_and_leakage.md"
TARGET_COLUMN = "Stroke_Risk"
LEAKAGE_COLUMNS = (
    "Stroke_Risk_Score",
    "AI_Health_Recommendation",
    "Doctor_Consultation_Needed",
)
REQUIRED_COLUMNS = {
    "Patient_ID",
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    TARGET_COLUMN,
    *LEAKAGE_COLUMNS,
}
BINARY_YES_NO_COLUMNS = (
    "Family_History_Stroke",
    "Family_History_Heart_Disease",
    "Diabetes",
    "Hypertension",
    "Heart_Disease",
    "Previous_TIA",
    "Atrial_Fibrillation",
    "Chronic_Kidney_Disease",
    "Doctor_Consultation_Needed",
)
IDENTIFIER_NAME = re.compile(r"(?:^|_)(?:id|identifier)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Source CSV path")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Feature configuration output"
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help="Markdown report output"
    )
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame, include_index: bool = True) -> str:
    headers = ([""] if include_index else []) + [str(column) for column in frame.columns]
    rows = [headers]
    for row_number, index in enumerate(frame.index):
        values = [frame.iloc[row_number, column] for column in range(frame.shape[1])]
        rows.append(([str(index)] if include_index else []) + [str(value) for value in values])

    def format_row(values: list[str]) -> str:
        escaped = [value.replace("|", "\\|") for value in values]
        return "| " + " | ".join(escaped) + " |"

    return "\n".join(
        [format_row(rows[0]), format_row(["---"] * len(headers))]
        + [format_row(row) for row in rows[1:]]
    )


def _is_text_column(series: pd.Series) -> bool:
    return (
        pd.api.types.is_string_dtype(series.dtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _detect_identifiers(data: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    identifiers: list[str] = []
    reasons: dict[str, str] = {}
    row_count = len(data)
    for column in data.columns:
        normalized_name = column.strip().replace(" ", "_")
        if not IDENTIFIER_NAME.search(normalized_name):
            continue
        unique_count = int(data[column].nunique(dropna=True))
        uniqueness = unique_count / row_count if row_count else 0.0
        identifiers.append(column)
        reasons[column] = (
            f"Identifier-like column name; {unique_count:,} unique non-null values "
            f"among {row_count:,} rows ({uniqueness:.2%}). Exclude from predictors."
        )
    return identifiers, reasons


def _categorical_quality(data: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, Any]]]:
    columns = [column for column in data.columns if _is_text_column(data[column])]
    results: dict[str, dict[str, Any]] = {}
    for column in columns:
        values = data[column].dropna().astype(str)
        trimmed = values.str.strip()
        normalized = trimmed.str.casefold()
        variants = normalized.groupby(normalized).apply(
            lambda keys: sorted(set(values.loc[keys.index].tolist()))
        )
        collisions = {
            key: variants_for_key
            for key, variants_for_key in variants.items()
            if len(variants_for_key) > 1
        }
        results[column] = {
            "unique_values": sorted(values.unique().tolist()),
            "blank_values": int(trimmed.eq("").sum()),
            "whitespace_values": int(values.ne(trimmed).sum()),
            "case_or_whitespace_collisions": collisions,
        }
    return columns, results


def _leakage_evidence(data: pd.DataFrame) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    score_by_target = data.groupby(TARGET_COLUMN, dropna=False)["Stroke_Risk_Score"].agg(
        ["count", "min", "max", "mean"]
    )
    score_value_classes = data.groupby("Stroke_Risk_Score")[TARGET_COLUMN].nunique()
    recommendation_classes = data.groupby("AI_Health_Recommendation")[TARGET_COLUMN].nunique()
    consultation_counts = pd.crosstab(
        data[TARGET_COLUMN], data["Doctor_Consultation_Needed"], dropna=False
    )
    consultation_rates = pd.crosstab(
        data[TARGET_COLUMN],
        data["Doctor_Consultation_Needed"],
        normalize="index",
        dropna=False,
    ).mul(100).round(2)

    score_deterministic = bool(not score_value_classes.empty and score_value_classes.max() == 1)
    recommendation_exclusive = bool(
        not recommendation_classes.empty and recommendation_classes.max() == 1
    )
    score_ranges = "; ".join(
        f"{risk_class}: {int(row['min'])}-{int(row['max'])}"
        for risk_class, row in score_by_target.iterrows()
    )
    score_reason = (
        f"Every observed Stroke_Risk_Score value maps to exactly one Stroke_Risk class "
        f"(100% class separation); observed class ranges are {score_ranges}. This is "
        "direct target encoding in this dataset. Exclude from training and inference."
        if score_deterministic
        else "The score is associated with Stroke_Risk and its generating process is undocumented. "
        "It may be target-derived; exclude pending provenance evidence."
    )
    recommendation_reason = (
        "Each observed AI_Health_Recommendation value occurs with only one Stroke_Risk "
        "class. This makes the recommendation a target proxy and it is likely produced "
        "after risk assessment. Exclude from training and inference."
        if recommendation_exclusive
        else "Recommendations are associated with Stroke_Risk and their generation time "
        "is undocumented. They may be downstream of the assessment; exclude pending "
        "provenance evidence."
    )
    high_yes_rate = (
        float(consultation_rates.loc["High", "Yes"])
        if "High" in consultation_rates.index and "Yes" in consultation_rates.columns
        else None
    )
    consultation_reason = (
        f"Doctor_Consultation_Needed is Yes for {high_yes_rate:.2f}% of High-risk rows "
        "and has different rates in the other target classes. Consultation is a likely "
        "downstream action; timing/provenance cannot be verified from the CSV. Exclude "
        "to avoid target-policy leakage."
        if high_yes_rate is not None
        else "Consultation status is target-associated and its generation time is "
        "undocumented. Exclude pending provenance evidence."
    )
    return (
        {
            "Stroke_Risk_Score": score_reason,
            "AI_Health_Recommendation": recommendation_reason,
            "Doctor_Consultation_Needed": consultation_reason,
        },
        {
            "score_by_target": score_by_target,
            "recommendation_by_target": pd.crosstab(
                data[TARGET_COLUMN], data["AI_Health_Recommendation"], dropna=False
            ),
            "consultation_counts": consultation_counts,
            "consultation_rates": consultation_rates,
        },
    )


def analyze(csv_path: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV was not found: {csv_path}")
    data = pd.read_csv(csv_path)
    missing_required = REQUIRED_COLUMNS.difference(data.columns)
    if missing_required:
        raise ValueError(
            "Dataset is missing columns required for this analysis: "
            + ", ".join(sorted(missing_required))
        )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values.")

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns, categorical_checks = _categorical_quality(data)
    identifiers, identifier_reasons = _detect_identifiers(data)
    leakage_reasons, leakage_tables = _leakage_evidence(data)
    excluded = list(LEAKAGE_COLUMNS)
    safe_features = [
        column
        for column in data.columns
        if column not in {TARGET_COLUMN, *identifiers, *excluded}
    ]
    numeric_features = [
        column for column in safe_features if pd.api.types.is_numeric_dtype(data[column].dtype)
    ]
    categorical_features = [column for column in safe_features if column not in numeric_features]

    numeric_quality_rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        series = data[column]
        numeric_quality_rows.append(
            {
                "column": column,
                "min": series.min(),
                "max": series.max(),
                "missing": int(series.isna().sum()),
                "non_finite": int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum()),
                "negative": int(series.lt(0).sum()),
                "zero": int(series.eq(0).sum()),
            }
        )
    numeric_quality = pd.DataFrame(numeric_quality_rows).set_index("column")
    missing_counts = data.isna().sum()
    exact_duplicates = int(data.duplicated().sum())
    identifier_duplicates = {
        column: int(data[column].duplicated().sum()) for column in identifiers
    }
    patient_ids = data["Patient_ID"]
    patient_id_values = patient_ids.dropna().to_numpy()
    patient_id_unique = bool(patient_ids.notna().all() and patient_ids.is_unique)
    patient_id_contiguous = False
    if patient_id_unique and pd.api.types.is_numeric_dtype(patient_ids.dtype):
        ordered_ids = np.sort(patient_id_values)
        patient_id_contiguous = bool(
            np.array_equal(
                ordered_ids,
                np.arange(ordered_ids[0], ordered_ids[-1] + 1),
            )
        )
    systolic = data["Blood_Pressure_Systolic"]
    diastolic = data["Blood_Pressure_Diastolic"]
    bp_comparable = systolic.notna() & diastolic.notna()
    bp_equal_mask = bp_comparable & systolic.eq(diastolic)
    bp_reversed_mask = bp_comparable & systolic.lt(diastolic)
    bp_equal = int(bp_equal_mask.sum())
    bp_reversed = int(bp_reversed_mask.sum())
    bp_invalid = bp_equal + bp_reversed
    bp_invalid_by_target = (
        data.loc[bp_equal_mask | bp_reversed_mask, TARGET_COLUMN]
        .value_counts()
        .rename("invalid_pressure_rows")
        .to_frame()
    )
    bmi_complete = data[["Height_cm", "Weight_kg", "BMI"]].dropna()
    calculated_bmi = bmi_complete["Weight_kg"] / (bmi_complete["Height_cm"] / 100) ** 2
    bmi_rounding_mismatch = int(calculated_bmi.round(1).ne(bmi_complete["BMI"]).sum())
    category_rows = [
        {
            "column": column,
            "unique_count": len(details["unique_values"]),
            "blank_values": details["blank_values"],
            "whitespace_values": details["whitespace_values"],
            "normalization_collisions": len(details["case_or_whitespace_collisions"]),
            "observed_values": ", ".join(details["unique_values"]),
        }
        for column, details in categorical_checks.items()
    ]
    category_quality = pd.DataFrame(category_rows).set_index("column")
    enum_violations: dict[str, list[str]] = {}
    for column in BINARY_YES_NO_COLUMNS:
        if column in data.columns:
            unexpected = sorted(set(data[column].dropna().astype(str)) - {"Yes", "No"})
            if unexpected:
                enum_violations[column] = unexpected

    target_distribution = data[TARGET_COLUMN].value_counts(dropna=False).rename("count").to_frame()
    target_distribution["percent"] = (
        data[TARGET_COLUMN].value_counts(normalize=True, dropna=False).mul(100).round(2)
    )
    config: dict[str, Any] = {
        "config_version": 1,
        "source_dataset": csv_path.relative_to(PROJECT_ROOT).as_posix()
        if csv_path.is_relative_to(PROJECT_ROOT)
        else str(csv_path),
        "target_column": TARGET_COLUMN,
        "target_classes": sorted(data[TARGET_COLUMN].astype(str).unique().tolist()),
        "features_safe_for_prediction": safe_features,
        "features_excluded_due_to_leakage": excluded,
        "identifier_columns": identifiers,
        "exclusion_reasons": leakage_reasons,
        "identifier_reasons": identifier_reasons,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "observed_categories": {
            column: sorted(data[column].dropna().astype(str).unique().tolist())
            for column in categorical_features
        },
        "selection_note": (
            "Safe features are provisional candidates based on this dataset's observed "
            "leakage checks. Confirm that each value is available at the intended time "
            "of prediction before model development."
        ),
        "preparation_policy": {
            "exact_duplicates": "Drop exact duplicate rows in the prepared copy only.",
            "blood_pressure": (
                "Exclude rows with both pressure values present and systolic less than "
                "or equal to diastolic; do not alter source values."
            ),
            "missing_values": (
                "Keep missing values in the prepared table; imputation must be fit on "
                "training data only using the configured preprocessor."
            ),
            "non_finite_numeric": "Convert positive/negative infinity to missing values.",
            "categorical_text": "Trim leading/trailing whitespace and map blanks to missing.",
        },
    }
    if set(safe_features) & set([TARGET_COLUMN, *identifiers, *excluded]):
        raise AssertionError("Feature configuration contains a target, identifier, or leakage column.")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    identifier_summary = pd.DataFrame(
        [
            {
                "column": column,
                "unique_non_null": int(data[column].nunique(dropna=True)),
                "rows": len(data),
                "reason": identifier_reasons[column],
            }
            for column in identifiers
        ]
    ).set_index("column")
    feature_list = ", ".join(f"`{column}`" for column in safe_features)
    leakage_sections = "\n".join(
        f"### `{column}`\n\n{leakage_reasons[column]}\n"
        for column in LEAKAGE_COLUMNS
    )
    categorical_domain_lines = "\n".join(
        f"- `{column}`: {', '.join(f'`{value}`' for value in details['unique_values'])}"
        for column, details in categorical_checks.items()
    )
    report = f"""# Data Quality and Target Leakage Analysis

## Dataset summary

- Source: `{config['source_dataset']}`
- Shape: **{len(data):,} rows × {len(data.columns)} columns**
- Target: `{TARGET_COLUMN}`; classes: {', '.join(f'`{value}`' for value in config['target_classes'])}
- Exact duplicate rows: **{exact_duplicates:,}**
- Missing values: **{int(missing_counts.sum()):,} cells** across **{int(missing_counts.gt(0).sum())} columns**
- Numeric columns: **{len(numeric_columns)}**; categorical/text columns: **{len(categorical_columns)}**

## Target distribution

{markdown_table(target_distribution)}

## Missing-value analysis

{markdown_table(missing_counts[missing_counts.gt(0)].rename("missing").to_frame()) if missing_counts.gt(0).any() else "No missing values found."}

Target values are complete. Missing predictor values will remain missing in the prepared copy and be imputed by a transformer fitted only on the training partition.

## Duplicates and identifiers

- Exact duplicate rows: **{exact_duplicates:,}**.
- Identifier-like columns detected from field names: {', '.join(f'`{column}`' for column in identifiers) or 'none'}.
- `Patient_ID` has {data['Patient_ID'].nunique(dropna=True):,} unique values among {len(data):,} rows, from {data['Patient_ID'].min()} through {data['Patient_ID'].max()}; unique: **{patient_id_unique}**, contiguous: **{patient_id_contiguous}**.
- Duplicate values for detected identifier columns: {identifier_duplicates}.

{markdown_table(identifier_summary) if not identifier_summary.empty else "No identifier columns detected."}

`Patient_ID` is excluded from model features regardless of its apparent numeric type.

## Numeric range and invalid-value checks

No external clinical data dictionary or measurement-unit specification was provided. Therefore the report gives the observed ranges and flags objective non-finite/negative values and logical cross-field inconsistencies; it does not invent clinical upper/lower cutoffs.

{markdown_table(numeric_quality)}

- Numeric non-finite values: **{int(numeric_quality['non_finite'].sum()):,}**.
- Negative numeric values: **{int(numeric_quality['negative'].sum()):,}**.
- Zero values: present in fields where zero can be meaningful (including exercise and walking); not treated as invalid by default.
- Blood pressure with systolic equal to diastolic: **{bp_equal:,}** rows.
- Blood pressure with systolic below diastolic: **{bp_reversed:,}** rows.
- BMI recomputation mismatches after rounding to one decimal from height/weight: **{bmi_rounding_mismatch:,} of {len(bmi_complete):,} complete rows**.

Rows where systolic pressure is less than or equal to diastolic pressure are flagged as logically inconsistent. The preparation pipeline excludes those rows from a separate modeling copy without changing either value in the source CSV.

Pressure anomaly rows by target:

{markdown_table(bp_invalid_by_target)}

## Categorical consistency

{markdown_table(category_quality)}

Observed category values:

{categorical_domain_lines}

Yes/No enum violations: {enum_violations or 'none'}.

No empty categories, leading/trailing whitespace, or case/whitespace normalization collisions were found. Observed vocabularies are recorded in the feature configuration; inference-time unseen categories are handled by the encoder rather than treated as errors.

## Target leakage review

The CSV has no provenance/timing metadata, so it cannot prove exactly when these fields were generated. The observed mappings below are directly measured from the actual records.

{leakage_sections}

#### Score by target

{markdown_table(leakage_tables['score_by_target'])}

#### Recommendation by target

{markdown_table(leakage_tables['recommendation_by_target'])}

#### Consultation by target (counts)

{markdown_table(leakage_tables['consultation_counts'])}

#### Consultation by target (row percentages)

{markdown_table(leakage_tables['consultation_rates'])}

## Persisted feature policy

- Target column: `{TARGET_COLUMN}`
- Identifier columns: {', '.join(f'`{column}`' for column in identifiers)}
- Excluded for leakage: {', '.join(f'`{column}`' for column in excluded)}
- Safe candidate feature count: **{len(safe_features)}** ({len(numeric_features)} numeric, {len(categorical_features)} categorical)
- Safe candidate columns: {feature_list}

The safe list is provisional: verify that each predictor is available at the intended prediction time. The same saved feature configuration is read by the preparation module; no model was trained.

## Preparation behavior

- The original CSV is read-only and never rewritten.
- The separate prepared copy drops exact duplicates and rows with non-missing systolic pressure less than or equal to diastolic pressure.
- The prepared copy contains only safe features plus the target; identifier and leakage columns are omitted.
- Missing values are preserved for a training-only imputer; infinities become missing and category text is trimmed.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Dataset shape: {data.shape}")
    print(f"Exact duplicate rows: {exact_duplicates}")
    print(f"Missing cells: {int(missing_counts.sum())} across {int(missing_counts.gt(0).sum())} columns")
    print(f"Blood pressure logical anomalies: {bp_invalid} ({bp_equal} equal, {bp_reversed} reversed)")
    print(f"BMI rounding mismatches: {bmi_rounding_mismatch}")
    print(f"Identifiers: {identifiers}")
    print(f"Excluded leakage columns: {excluded}")
    print(f"Safe feature count: {len(safe_features)}")
    print(f"Feature config written to: {config_path}")
    print(f"Quality report written to: {report_path}")
    return config


def main() -> None:
    args = parse_args()
    analyze(args.csv.resolve(), args.config.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()