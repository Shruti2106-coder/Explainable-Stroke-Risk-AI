"""Sklearn-compatible estimators requiring fold-local target handling."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Encode string labels and calculate balanced sample weights inside each fit call."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> BalancedXGBClassifier:
        labels = np.asarray(y).astype(str)
        self.label_encoder_ = LabelEncoder().fit(labels)
        encoded_labels = self.label_encoder_.transform(labels)
        sample_weights = compute_sample_weight(class_weight="balanced", y=encoded_labels)
        self.estimator_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=len(self.label_encoder_.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(X, encoded_labels, sample_weight=sample_weights)
        self.classes_ = self.label_encoder_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded_predictions = self.estimator_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded_predictions)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)