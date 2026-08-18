import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


MODEL_PATH = "model/sample_model.joblib"
DATASET_PATH = "dataset/test_dataset.csv"


def get_results():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATASET_PATH)

    X = df.drop("label", axis=1)
    y_true = df["label"]

    y_pred = model.predict(X)

    return y_true, y_pred


def test_accuracy():
    y_true, y_pred = get_results()

    accuracy = accuracy_score(y_true, y_pred)

    assert 0 <= accuracy <= 1


def test_precision():
    y_true, y_pred = get_results()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    assert 0 <= precision <= 1


def test_recall():
    y_true, y_pred = get_results()

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    assert 0 <= recall <= 1


def test_f1_score():
    y_true, y_pred = get_results()

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    assert 0 <= f1 <= 1


def test_confusion_matrix():
    y_true, y_pred = get_results()

    matrix = confusion_matrix(y_true, y_pred)

    assert matrix.shape == (2, 2)