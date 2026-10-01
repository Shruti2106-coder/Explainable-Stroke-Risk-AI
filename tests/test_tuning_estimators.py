import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from backend.ml.estimators import BalancedXGBClassifier


def test_balanced_xgboost_wrapper_encodes_and_decodes_multiclass_labels():
    features = np.arange(60, dtype=float).reshape(20, 3)
    labels = np.array(["Low"] * 4 + ["Moderate"] * 10 + ["High"] * 6)
    estimator = BalancedXGBClassifier(
        n_estimators=3,
        max_depth=2,
        learning_rate=0.1,
        random_state=11,
        n_jobs=1,
    )

    cloned = clone(estimator)
    cloned.fit(features, labels)

    predictions = cloned.predict(features)
    probabilities = cloned.predict_proba(features)
    assert set(cloned.classes_) == {"Low", "Moderate", "High"}
    assert set(predictions).issubset(set(labels))
    assert probabilities.shape == (20, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    check_is_fitted(cloned, "estimator_")