# Stratified Splitting and Class Imbalance

Run from the workspace root after Module 2 preparation:

```powershell
python backend/ml/split_data.py
python backend/ml/class_imbalance.py
```

`split_data.py` creates a fixed-seed stratified 70% training, 15% validation, and 15% final-test split under `data/splits/`. The test file is named `final_test_locked.csv`; the manifest does not reveal its class counts. `class_imbalance.py` intentionally reads only `train.csv` and `validation.csv`, never the locked test file.

The analysis calculates balanced class weights from training labels and compares the original training counts with a temporary `RandomOverSampler` result. The temporary resampled rows are not saved. Random oversampling duplicates observed rows, so mixed numeric/categorical records stay valid. The validation set remains unmodified.

The shared `build_random_oversampling_pipeline(config, estimator)` places Module 4 preprocessing before a sampler inside an `imblearn.pipeline.Pipeline`. Use this pipeline for training/cross-validation: preprocessing and sampling are fitted per training fold; `predict` does not resample validation or test rows. For class-weight approaches, compute weights from each training fold only and pass them to an estimator that supports them.

Vanilla SMOTE on one-hot features can create fractional category indicators. SMOTENC is the more appropriate synthetic option for mixed data, but it has not been run here; evaluate it inside training folds only and compare validation per-class recall/F1 and macro-F1. This module does not train models or select a final estimator or balancing method. The current class-weight recommendation is a starting baseline, not a final selection.

The final test partition must remain unopened for class rebalancing, feature selection, hyperparameter tuning, or performance inspection until the single final evaluation stage.

Focused tests:

```powershell
python -m pytest tests/test_data_splitting.py tests/test_class_imbalance.py
```