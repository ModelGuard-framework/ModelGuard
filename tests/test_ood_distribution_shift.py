
from core.config import dataset_path, label_column, model_path
from core.validator import (
    check_distribution_shift,
    check_ood,
    find_label_column,
    get_feature_columns,
    load_dataset,
    load_model,
)


def get_model_input():
    model = load_model(model_path())
    df = load_dataset(dataset_path())
    label = find_label_column(df, label_column())
    features = get_feature_columns(model, df, label)
    return model, df[features], features


def test_ood_without_reference_uses_self_consistency():
    _, X, features = get_model_input()
    result = check_ood(X, reference_df=None, features=features)
    assert result["status"] in {"PASS", "WARNING"}, result["details"]
    assert result["used_reference"] is False


def test_ood_with_reference_dataset():
    _, X, features = get_model_input()
    # Use the dataset itself as its own reference: a dataset should not be
    # OOD relative to itself.
    result = check_ood(X, reference_df=X, features=features)
    assert result["status"] == "PASS", result["details"]
    assert result["used_reference"] is True
    assert result["ood_count"] == 0


def test_distribution_shift_skipped_without_reference():
    _, X, features = get_model_input()
    result = check_distribution_shift(X, reference_df=None, features=features)
    assert result["status"] == "SKIPPED", result["details"]


def test_distribution_shift_passes_against_itself():
    _, X, features = get_model_input()
    result = check_distribution_shift(X, reference_df=X, features=features)
    assert result["status"] == "PASS", result["details"]
    assert result["shifted_features"] == []
