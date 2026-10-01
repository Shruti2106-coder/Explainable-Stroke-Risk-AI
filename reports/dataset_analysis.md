# Dataset Analysis Report

## Dataset overview

- Source: `data/stroke_risk_prediction_dataset.csv`
- Shape: **50,000 rows × 40 columns**
- Target: `Stroke_Risk`
- Feature type inventory (including the target and identifier): 18 numeric columns and 22 categorical/text/boolean columns
- Duplicate rows: **0**
- Missing cells: **13,522** across **13** columns

## Column inventory

|  | data_type | missing_values | unique_non_null |
| --- | --- | --- | --- |
| Patient_ID | int64 | 0 | 50000 |
| Age | float64 | 219 | 73 |
| Gender | str | 0 | 3 |
| Country | str | 0 | 25 |
| Height_cm | float64 | 1525 | 501 |
| Weight_kg | float64 | 1523 | 851 |
| BMI | float64 | 0 | 494 |
| Blood_Pressure_Systolic | int64 | 0 | 71 |
| Blood_Pressure_Diastolic | int64 | 0 | 51 |
| Heart_Rate | int64 | 0 | 56 |
| Blood_Glucose | float64 | 960 | 1501 |
| HbA1c | float64 | 1006 | 61 |
| Total_Cholesterol | float64 | 1004 | 1901 |
| HDL | float64 | 1021 | 651 |
| LDL | float64 | 1050 | 1601 |
| Triglycerides | float64 | 1032 | 2901 |
| Smoking_Status | str | 0 | 3 |
| Alcohol_Consumption | str | 0 | 3 |
| Physical_Activity_Level | str | 0 | 3 |
| Sleep_Hours | float64 | 1026 | 61 |
| Stress_Level | str | 0 | 3 |
| Diet_Quality | str | 0 | 3 |
| Family_History_Stroke | str | 0 | 2 |
| Family_History_Heart_Disease | str | 0 | 2 |
| Diabetes | str | 0 | 2 |
| Hypertension | str | 0 | 2 |
| Heart_Disease | str | 0 | 2 |
| Previous_TIA | str | 0 | 2 |
| Atrial_Fibrillation | str | 0 | 2 |
| Chronic_Kidney_Disease | str | 0 | 2 |
| Medication_Adherence | str | 1051 | 3 |
| Exercise_Hours_Per_Week | float64 | 1054 | 101 |
| Daily_Walking_Minutes | float64 | 1051 | 121 |
| Work_Type | str | 0 | 5 |
| Residence_Type | str | 0 | 2 |
| Air_Pollution_Exposure | str | 0 | 3 |
| Stroke_Risk_Score | int64 | 0 | 96 |
| AI_Health_Recommendation | str | 0 | 13 |
| Doctor_Consultation_Needed | str | 0 | 2 |
| Stroke_Risk | str | 0 | 3 |

## Target classes and distribution

Observed classes: `Moderate`, `High`, `Low`.

|  | count | percent |
| --- | --- | --- |
| Moderate | 34375 | 68.75 |
| High | 12112 | 24.22 |
| Low | 3513 | 7.03 |

The target is imbalanced: the most frequent class is `Moderate`. Any future evaluation should use stratified splitting and report per-class metrics; accuracy alone would hide minority-class behavior.

## Potential target leakage

These columns are preserved in the dataset. No columns were removed or model features selected during this analysis.

### `Stroke_Risk_Score`

The observed score ranges do not overlap across target classes. This is strong evidence that the score encodes the class or was generated using the target; treating it as a predictor risks direct leakage.

|  | count | nunique | min | max | mean |
| --- | --- | --- | --- | --- | --- |
| High | 12112 | 36 | 65 | 100 | 73.13408190224571 |
| Low | 3513 | 30 | 2 | 34 | 28.836891545687447 |
| Moderate | 34375 | 30 | 35 | 64 | 50.96762181818182 |

### `AI_Health_Recommendation`

Each observed recommendation maps to only one target class in this CSV. It is strongly target-tied and should be excluded from predictor inputs unless its provenance is independently established.

Recommendations by target class:

|  | Annual Health Checkup | Balanced Diet | Consult Neurologist Immediately | Continue Regular Exercise | Emergency Cardiovascular Assessment | Exercise Regularly | Healthy Lifestyle | Improve Diet Quality | Maintain Healthy Weight | Monitor Blood Glucose Daily | Monitor Blood Pressure | Reduce Sodium Intake | Strict Blood Pressure Control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High | 0 | 0 | 2978 | 0 | 3096 | 0 | 0 | 0 | 0 | 3007 | 0 | 0 | 3031 |
| Low | 690 | 716 | 0 | 709 | 0 | 0 | 713 | 0 | 685 | 0 | 0 | 0 | 0 |
| Moderate | 0 | 0 | 0 | 0 | 0 | 8585 | 0 | 8534 | 0 | 0 | 8650 | 8606 | 0 |

### `Doctor_Consultation_Needed`

Every High-risk record is marked Yes for consultation, while the other classes have mixed values. This is a strong downstream proxy for the target and may leak target-related decision policy.

Counts by target class:

|  | No | Yes |
| --- | --- | --- |
| High | 0 | 12112 |
| Low | 3164 | 349 |
| Moderate | 10257 | 24118 |

Row percentages by target class:

|  | No | Yes |
| --- | --- | --- |
| High | 0.0 | 100.0 |
| Low | 90.07 | 9.93 |
| Moderate | 29.84 | 70.16 |

## Initial observations

- The dataset has 50,000 records and 40 columns; `Patient_ID` is an identifier and must not be used as a model feature.
- 13 columns contain missing values (13,522 cells total); missing-data handling must be learned/applied using training data only.
- There are 0 exact duplicate rows.
- The target class distribution is uneven, with `Moderate` as the majority class and `Low` as the minority class.
- All three reviewed columns are plausible leakage sources. In a future modeling module, exclude them from predictors by default until their generation timing and business meaning are established. Keep the original dataset intact.
- No model was trained, no resampling or preprocessing was applied, and no performance claims are made in Module 1.

## Method and limitations

This report is computed directly from the supplied CSV using pandas. Associations and non-overlapping value ranges are evidence of target dependence, not proof of the dataset's generation process. The CSV does not document how the score, recommendation, or consultation flag was produced.
