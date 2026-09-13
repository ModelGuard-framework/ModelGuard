
from core.config import dataset_path, label_column, model_path
from core.validator import (
    check_ml_security,
    find_label_column,
    load_dataset,
    load_model,
)


def test_ml_security_indicators():
    model = load_model(model_path())
    df = load_dataset(dataset_path())
    label = find_label_column(df, label_column())

    result = check_ml_security(df, model, label)

    # WARNING is allowed because these are indicators for review, not proof of
    # an attack. A structural failure is not allowed.
    assert result["status"] in {"PASS", "WARNING"}, result["details"]


def test_no_duplicate_samples():
    df = load_dataset(dataset_path())
    assert int(df.duplicated().sum()) == 0
