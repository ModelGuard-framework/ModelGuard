import pandas as pd
import numpy as np


DATASET_PATH = "dataset/test_dataset.csv"


def test_no_missing_values():
    df = pd.read_csv(DATASET_PATH)

    assert not df.isnull().any().any()


def test_no_duplicate_records():
    df = pd.read_csv(DATASET_PATH)

    assert df.duplicated().sum() == 0


def test_numeric_features():
    df = pd.read_csv(DATASET_PATH)

    features = [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4"
    ]

    for feature in features:
        assert pd.api.types.is_numeric_dtype(df[feature])


def test_no_infinite_values():
    df = pd.read_csv(DATASET_PATH)

    numeric_data = df.select_dtypes(include=np.number)

    assert not np.isinf(numeric_data.to_numpy()).any()