export type RiskClass = 'Low' | 'Moderate' | 'High'
export type NumericRange = { min: number; max: number }

export interface ModelMetrics {
  accuracy: number
  macro_precision: number
  macro_recall: number
  macro_f1: number
  weighted_precision: number
  weighted_recall: number
  weighted_f1: number
  roc_auc_ovr_macro: number
  pr_auc_ovr_macro_average_precision: number
  confusion_matrix: number[][]
  classification_report: Record<string, { precision?: number; recall?: number; 'f1-score'?: number; support?: number }>
}

export interface ModelComparisonRow {
  model_key: string
  model: string
  stage: 'baseline_validation' | 'tuned_validation'
  accuracy: number
  macro_precision: number
  macro_recall: number
  macro_f1: number
  weighted_precision: number
  weighted_recall: number
  weighted_f1: number
  roc_auc_ovr_macro: number
  pr_auc_ovr_macro_average_precision: number
}

export interface ModelInfo {
  model_name: string
  model_type: string
  target_column: string
  target_classes: RiskClass[]
  approved_features: string[]
  numeric_features: string[]
  categorical_features: string[]
  categorical_options: Record<string, string[]>
  numeric_ranges: Record<string, NumericRange>
  optional_features: string[]
  excluded_leakage_features: string[]
  identifier_columns: string[]
  fit_rows: number
  validation_macro_f1: number
  final_test_metrics: ModelMetrics
  final_test_class_order: RiskClass[]
  model_comparison_validation: ModelComparisonRow[]
  disclaimer: string
}

export type PatientPayload = Record<string, string | number | null>

export interface PredictionResult {
  predicted_risk: RiskClass
  probability_low: number
  probability_moderate: number
  probability_high: number
  probabilities: { low: number; moderate: number; high: number }
  disclaimer: string
}

export interface LocalContribution {
  feature: string
  value: string | number | null
  shap_value: number
  direction: 'toward_prediction' | 'away_from_prediction'
}

export interface LocalExplanation {
  predicted_risk_class: RiskClass
  probabilities: Record<RiskClass, number>
  explained_class: RiskClass
  explanation_output: string
  base_value: number
  features_contributing_toward_prediction: LocalContribution[]
  features_contributing_away_from_prediction: LocalContribution[]
  encoded_feature_details_toward: Array<LocalContribution & { source_feature: string; transformed_feature: string }>
  encoded_feature_details_away: Array<LocalContribution & { source_feature: string; transformed_feature: string }>
  contribution_note: string
  disclaimer: string
}

export interface GlobalFeatureImportance {
  feature: string
  mean_abs_shap: number
  [key: string]: string | number
}

export interface GlobalExplanation {
  explainer: string
  model: string
  explained_split: string
  rows_explained: number
  class_order: RiskClass[]
  feature_importance: GlobalFeatureImportance[]
  artifacts: Record<string, string>
  disclaimer: string
}