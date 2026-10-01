# SHAP Explanations

The service in `shap_explanation.py` loads the finalized estimator pipeline, feature-config snapshot, and model metadata. It inspects the fitted model: the current pipeline wraps `BalancedXGBClassifier`, whose fitted inner estimator is `XGBClassifier`, so SHAP `TreeExplainer` is used for its multiclass raw class margins. Non-XGBoost estimators use a generic probability-based `shap.Explainer` with a training-only background sample.

## Global

`get_global_explanation()` explains a deterministic stratified sample from `data/splits/validation.csv`; it never reads the locked test set. Encoded category and missingness-indicator attributions are summed back to their original approved source feature before global importance is ranked. The pooled SHAP beeswarm and class-specific summary plots retain readable transformed category labels.

Artifacts are saved under `reports/shap/` and `images/shap/`: `global_explanation.json`, `global_feature_importance.csv`, the global mean-absolute-SHAP bar chart, one pooled summary plot, and one class-specific summary per output class.

## Local

`get_prediction_explanation(patient_data)` accepts exactly one dictionary or one-row DataFrame. It returns the predicted class, Low/Moderate/High probabilities, original-feature contributions toward/away from the predicted class, and optional encoded-feature details. One-hot columns are named as `source feature = category` and their values are also grouped back into the source feature.

For this XGBoost pipeline, local SHAP contributions are in raw class-margin units, not probability changes. A positive contribution moves that class output upward relative to the SHAP baseline; a negative contribution moves it downward. SHAP describes model behavior and does not establish medical causation or diagnosis. Local explanations are returned to the caller but are not persisted, avoiding unnecessary storage of patient health inputs.

## API

- `GET /explanations/global`
- `POST /explanations/prediction` with the patient feature dictionary as the JSON body.

The selected feature config drops IDs and known leakage fields before prediction/explanation.