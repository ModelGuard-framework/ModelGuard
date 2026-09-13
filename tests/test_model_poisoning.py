
from core.config import dataset_path, label_column, model_path
from core.validator import (
    check_model_poisoning,
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
    return model, df[features]


def test_model_poisoning_indicators():
    model, X = get_model_input()
    result = check_model_poisoning(model, X)

    # WARNING is allowed because these are indicators for review, not proof
    # of a compromised model. A structural failure (e.g. model can't predict
    # at all) is not allowed.
    assert result["status"] in {"PASS", "WARNING"}, result["details"]


def test_model_poisoning_reports_baseline_share():
    model, X = get_model_input()
    result = check_model_poisoning(model, X)
    assert 0.0 <= result["baseline_dominant_share"] <= 1.0
