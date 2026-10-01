# Preprocessing Pipeline

`preprocessing.py` is the shared preprocessing entry point for training, validation, and API inference. It loads and validates `feature_config.json` and selects exactly `features_safe_for_prediction`; extra columns such as `Patient_ID`, `Stroke_Risk_Score`, `AI_Health_Recommendation`, `Doctor_Consultation_Needed`, and the target are ignored as inputs.

## Transformations

- Numeric: median imputation, missingness indicators, and `StandardScaler` by default. Use `scale_numeric=False` for estimators that do not need scaled numeric values.
- Categorical: most-frequent imputation, missingness indicators, and one-hot encoding with `handle_unknown="ignore"` for unseen API categories.
- Numeric infinities are treated as missing. Empty categorical strings are normalized to missing.
- `get_transformed_feature_metadata()` returns each output feature name and the original input column it came from, including encoded categories and imputation indicators. This mapping can be joined to later SHAP output.
- `build_training_pipeline(config, estimator)` returns one unfitted sklearn `Pipeline` containing the preprocessor and supplied estimator. This is the recommended interface for fitting and cross-validation because each fit/fold learns imputation, scaling, and encoding from only its own training rows.
- `tune_models.py` places this preprocessor inside `RandomizedSearchCV` so every stratified fold fits imputation/scaling/encoding on its fold-training rows only.

## Training and inference

Split the rows first. Prefer `build_training_pipeline(config, estimator)`, then fit the returned pipeline with the training features and labels only; use that same fitted pipeline to score validation/test rows and later API requests. The standalone `fit_preprocessor(training_frame, config)` helper is available when a fitted transformer is needed separately; it must receive only the training partition. For direct inference transforms, call `transform_features(fitted_preprocessor, frame, config)`; extra identifier/leakage columns are dropped by feature selection and unknown categorical levels are ignored by the encoder.

Do not fit this transformer on the complete dataset. Module 4 creates and tests the code but does not fit or save a fitted transformer. In a later training module, fit the complete estimator pipeline using training data and persist the fitted pipeline artifact only after training is complete.

Run the focused tests from the workspace root:

```powershell
python -m pytest tests/test_preprocessing.py
```