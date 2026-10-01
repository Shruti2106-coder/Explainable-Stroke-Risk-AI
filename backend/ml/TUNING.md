# Hyperparameter Tuning and Final Evaluation

Run after Module 6 baseline results exist:

```powershell
python -m backend.ml.tune_models
```

The default uses 3-fold shuffled stratified CV and 8 randomized parameter configurations per estimator, with macro-F1 as the refit score and Low-class recall also recorded. Each model family covers the requested parameters. Adjust `--cv-folds`, `--iterations`, and `--seed` explicitly when running a different search.

## Leakage controls

- The search reads `train.csv` and `validation.csv` only. The preprocessing `ColumnTransformer` sits inside the CV pipeline, so imputers, missingness indicators, scaling, and category encoding are learned separately within each training fold.
- The final test file is read only after all CV searches finish, tuned pipelines are scored on validation, and one winner is selected by validation macro-F1 (ties: Low recall, macro OVR ROC-AUC, macro OVR average precision).
- The selected tuned configuration is refit on train + validation before its single final-test evaluation. Test metrics are never used to rank, tune, or choose models.
- No oversampling is used by the tuner. Logistic Regression, Random Forest, and SVM use their estimator class-weight support; XGBoost computes balanced per-row sample weights in each estimator fit, therefore separately for each CV fold.
- The original feature configuration remains the source of truth; no identifiers or leakage columns enter model inputs.

## Outputs

- `reports/tuning_results.json`: CV results summary, validation ranking, parameters, and final test metrics.
- `reports/cv_results_<model>.csv`: all randomized-search candidate/fold results.
- `reports/baseline_vs_tuned_validation.csv`: baseline and tuned validation comparison.
- `reports/final_test_classification_report.json`: final class-wise test metrics.
- `reports/hyperparameter_tuning_report.md`: methodology and actual results.
- `images/model_tuning/`: tuned validation confusion matrices, ROC/PR curves, validation comparison, and final-test plots.
- `models/final_stroke_risk_pipeline.joblib`: selected model and fitted preprocessing pipeline.
- `models/final_preprocessing_pipeline.joblib`, `models/final_feature_config.json`, and `models/final_model_metadata.json`.

The final test metrics are a one-time final estimate. Do not use them to change hyperparameters or choose a different model.