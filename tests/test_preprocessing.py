import pandas as pd
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "dataset/test_dataset.csv"


def test_preprocessing_preserves_rows():
    df = pd.read_csv(DATASET_PATH)

    X = df.drop("label", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape[0] == X.shape[0]


def test_preprocessing_preserves_features():
    df = pd.read_csv(DATASET_PATH)

    X = df.drop("label", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape[1] == X.shape[1]


def test_preprocessing_produces_numeric_values():
    df = pd.read_csv(DATASET_PATH)

    X = df.drop("label", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.dtype.kind in "fc"


def test_preprocessing_has_no_missing_values():
    df = pd.read_csv(DATASET_PATH)

    X = df.drop("label", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert not pd.isnull(X_scaled).any()