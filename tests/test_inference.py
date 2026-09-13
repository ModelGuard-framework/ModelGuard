
from core.config import dataset_path, label_column, model_path
from core.validator import (
    check_inference,
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


def test_model_generates_predictions():
    model, X = get_model_input()
    result = check_inference(model, X)
    assert result["status"] == "PASS", result["details"]


def test_prediction_count_matches_input():
    model, X = get_model_input()
    predictions = model.predict(X)
    assert len(predictions) == len(X)


def test_predictions_are_not_empty():
    model, X = get_model_input()
    predictions = model.predict(X)
    assert len(predictions) > 0
