
from core.config import dataset_path, label_column, model_path
from core.validator import (
    check_results,
    find_label_column,
    get_feature_columns,
    load_dataset,
    load_model,
)


def get_inputs():
    model = load_model(model_path())
    df = load_dataset(dataset_path())
    label = find_label_column(df, label_column())
    features = get_feature_columns(model, df, label)
    X = df[features]
    y = df[label] if label else None
    return model, X, y


def test_result_validation():
    model, X, y = get_inputs()
    result = check_results(model, X, y)
    assert result["status"] == "PASS", result["details"]


def test_metrics_are_valid_when_labels_exist():
    model, X, y = get_inputs()
    if y is None:
        return

    result = check_results(model, X, y)
    for value in result["metrics"].values():
        assert 0.0 <= value <= 1.0
