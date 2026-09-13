
import numpy as np
import pandas as pd

from core.imputation import impute_dataframe
from core.validator import run_data_imputation


def make_missing_df():
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "feature_2": [10.0, np.nan, 30.0, 40.0, 50.0],
            "label": [0, 1, 0, 1, 0],
        }
    )


def test_mean_imputation_fills_all_missing_values():
    df = make_missing_df()
    imputed, report = impute_dataframe(df, ["feature_1", "feature_2"], strategy="mean")
    assert imputed[["feature_1", "feature_2"]].isna().sum().sum() == 0
    assert report["total_imputed"] == 2
    assert report["missing_after"] == 0


def test_knn_imputation_fills_all_missing_values():
    df = make_missing_df()
    imputed, report = impute_dataframe(df, ["feature_1", "feature_2"], strategy="knn")
    assert imputed[["feature_1", "feature_2"]].isna().sum().sum() == 0
    assert report["strategy"] == "knn"


def test_unknown_strategy_raises():
    df = make_missing_df()
    try:
        impute_dataframe(df, ["feature_1"], strategy="bogus")
        assert False, "expected ValueError for unknown strategy"
    except ValueError:
        pass


def test_run_data_imputation_reports_pass_status():

    df = make_missing_df()

    result, imputed_df = run_data_imputation(
        df,
        ["feature_1", "feature_2"],
        strategy="mean"
    )

    assert result["status"] == "PASS", result["details"]

    assert imputed_df[["feature_1", "feature_2"]].isna().sum().sum() == 0