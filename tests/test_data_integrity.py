import pandas as pd


DATASET_PATH = "dataset/test_dataset.csv"

FEATURES = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4"
]

VALID_LABELS = {0, 1}


def test_no_duplicate_records():
    df = pd.read_csv(DATASET_PATH)

    assert df.duplicated().sum() == 0


def test_labels_are_valid():
    df = pd.read_csv(DATASET_PATH)

    assert set(df["label"].unique()).issubset(VALID_LABELS)


def test_no_extreme_outliers():
    df = pd.read_csv(DATASET_PATH)

    for feature in FEATURES:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        outliers = df[
            (df[feature] < lower_bound)
            | (df[feature] > upper_bound)
        ]

        assert len(outliers) == 0


def test_required_features_exist():
    df = pd.read_csv(DATASET_PATH)

    for feature in FEATURES:
        assert feature in df.columns