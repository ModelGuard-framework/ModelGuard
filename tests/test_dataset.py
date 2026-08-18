import os
import pandas as pd


DATASET_PATH = "dataset/test_dataset.csv"

EXPECTED_COLUMNS = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "label"
]


def test_dataset_file_exists():
    assert os.path.exists(DATASET_PATH)


def test_dataset_can_be_loaded():
    df = pd.read_csv(DATASET_PATH)
    assert df is not None


def test_dataset_is_not_empty():
    df = pd.read_csv(DATASET_PATH)
    assert not df.empty


def test_required_columns_exist():
    df = pd.read_csv(DATASET_PATH)

    for column in EXPECTED_COLUMNS:
        assert column in df.columns