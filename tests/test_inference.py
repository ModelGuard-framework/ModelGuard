import pandas as pd
import joblib


MODEL_PATH = "model/sample_model.joblib"
DATASET_PATH = "dataset/test_dataset.csv"


def test_model_generates_predictions():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATASET_PATH)
    X = df.drop("label", axis=1)

    predictions = model.predict(X)

    assert predictions is not None


def test_prediction_count_matches_input():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATASET_PATH)
    X = df.drop("label", axis=1)

    predictions = model.predict(X)

    assert len(predictions) == len(X)


def test_predictions_are_not_empty():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATASET_PATH)
    X = df.drop("label", axis=1)

    predictions = model.predict(X)

    assert len(predictions) > 0


def test_predictions_are_valid():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATASET_PATH)
    X = df.drop("label", axis=1)

    predictions = model.predict(X)

    assert set(predictions).issubset({0, 1})