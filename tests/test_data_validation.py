
from core.config import dataset_path, label_column, model_path
from core.validator import (
    check_dataset_validation,
    find_label_column,
    load_dataset,
    load_model,
)


def test_dataset_validation():
    model = load_model(model_path())
    df = load_dataset(dataset_path())
    label = find_label_column(df, label_column())

    result = check_dataset_validation(df, model, label)
    assert result["status"] == "PASS", result["details"]


def test_no_missing_values():
    df = load_dataset(dataset_path())
    assert int(df.isna().sum().sum()) == 0


def test_no_duplicate_records():
    df = load_dataset(dataset_path())
    assert int(df.duplicated().sum()) == 0


def test_no_infinite_numeric_values():
    df = load_dataset(dataset_path())
    numeric = df.select_dtypes(include="number")
    assert numeric.shape[1] > 0
    assert numeric.notna().all().all()
