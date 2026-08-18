import pandas as pd
import numpy as np
import joblib


MODEL_PATH = "model/sample_model.joblib"
DATASET_PATH = "dataset/test_dataset.csv"


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)

    X = df.drop("label", axis=1)

    return model, X


def test_small_perturbation():
    model, X = load_model_and_data()

    X_perturbed = X.copy()
    X_perturbed["feature_1"] += 0.1

    predictions = model.predict(X_perturbed)

    assert len(predictions) == len(X)


def test_noise_input():
    model, X = load_model_and_data()

    X_noisy = X.copy()

    noise = np.random.normal(
        0,
        0.1,
        X_noisy.shape
    )

    X_noisy = X_noisy + noise

    predictions = model.predict(X_noisy)

    assert len(predictions) == len(X)


def test_missing_feature():
    model, X = load_model_and_data()

    X_missing = X.copy()

    X_missing["feature_1"] = np.nan

    try:
        predictions = model.predict(X_missing)

        assert len(predictions) == len(X)

    except ValueError:
        # Model correctly rejects invalid input
        assert True


def test_out_of_distribution_input():
    model, X = load_model_and_data()

    X_ood = X.copy()

    # Create values far outside the normal range
    X_ood["feature_1"] = 1000
    X_ood["feature_2"] = 1000
    X_ood["feature_3"] = 1000
    X_ood["feature_4"] = 1000

    predictions = model.predict(X_ood)

    assert len(predictions) == len(X)