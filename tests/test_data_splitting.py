import pandas as pd

from backend.ml.split_data import stratified_partitions


def test_stratified_partitions_are_disjoint_and_keep_class_proportions():
    data = pd.DataFrame(
        {
            "feature": range(600),
            "Stroke_Risk": ["Low"] * 60 + ["Moderate"] * 360 + ["High"] * 180,
        }
    )
    train, validation, final_test = stratified_partitions(
        data,
        "Stroke_Risk",
        train_size=0.7,
        validation_size=0.15,
        random_state=19,
    )

    assert len(train) + len(validation) + len(final_test) == len(data)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(final_test.index)
    assert set(validation.index).isdisjoint(final_test.index)
    for partition in (train, validation, final_test):
        proportions = partition["Stroke_Risk"].value_counts(normalize=True)
        assert abs(proportions["Low"] - 0.1) < 0.02
        assert abs(proportions["Moderate"] - 0.6) < 0.02
        assert abs(proportions["High"] - 0.3) < 0.02
