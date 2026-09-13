
"""
Data imputation utilities for ModelGuard.

Handles missing values in a dataset instead of the framework treating them
as an automatic hard failure. Imputation is always reported as a
data-quality intervention (never silently applied) so a reviewer can see
exactly what was changed before trusting downstream validation results.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

SUPPORTED_STRATEGIES = ("mean", "median", "most_frequent", "knn")


def impute_dataframe(
    df: pd.DataFrame,
    features: list[str],
    strategy: str = "mean",
    knn_neighbors: int = 5,
    reference_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Impute missing values in `features` columns of `df`.

    If `reference_df` (e.g. the original training dataset) is supplied,
    imputation statistics (mean/median/most-frequent value, or KNN
    neighbors) are learned from `reference_df` and applied to `df`, for any
    feature that exists in both. This avoids deriving fill values from the
    very dataset being validated, which may be small, already skewed, or
    the same data being screened for anomalies/poisoning. Any feature not
    present in `reference_df` (or the case where no `reference_df` is
    supplied at all) falls back to fitting on `df` itself, as before.

    Returns (imputed_dataframe, report). The report records the strategy
    used, how many values were missing per column, which columns were
    imputed using the reference dataset's statistics, and how many (if
    any) values remain unimputed so callers can decide whether to proceed.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown imputation strategy '{strategy}'. "
            f"Supported strategies: {', '.join(SUPPORTED_STRATEGIES)}."
        )

    working = df.copy()

    missing_before = {
        col: int(working[col].isna().sum())
        for col in features
        if int(working[col].isna().sum()) > 0
    }

    numeric_features = [
        c for c in features if pd.api.types.is_numeric_dtype(working[c])
    ]
    non_numeric_features = [c for c in features if c not in numeric_features]

    ref_available = set(reference_df.columns) if reference_df is not None else set()
    reference_columns_used: list[str] = []

    def _make_imputer(base_strategy: str, fit_source_len: int):
        if base_strategy == "knn":
            neighbors = min(knn_neighbors, max(fit_source_len - 1, 1))
            return KNNImputer(n_neighbors=neighbors)
        return SimpleImputer(strategy=base_strategy)

    def _apply(cols: list[str], base_strategy: str):
        if not cols:
            return
        ref_cols = [c for c in cols if c in ref_available]
        self_cols = [c for c in cols if c not in ref_available]

        if ref_cols:
            imputer = _make_imputer(base_strategy, len(reference_df))
            imputer.fit(reference_df[ref_cols])
            working[ref_cols] = imputer.transform(working[ref_cols])
            reference_columns_used.extend(ref_cols)

        if self_cols:
            imputer = _make_imputer(base_strategy, len(working))
            working[self_cols] = imputer.fit_transform(working[self_cols])

    if strategy == "knn":
        _apply(numeric_features, "knn")
    elif strategy in ("mean", "median"):
        _apply(numeric_features, strategy)
    elif strategy == "most_frequent":
        _apply(features, "most_frequent")

    # mean/median/knn only touch numeric columns; fall back to most-frequent
    # for any non-numeric columns that still contain missing values.
    if strategy in ("mean", "median", "knn") and non_numeric_features:
        remaining = [c for c in non_numeric_features if working[c].isna().any()]
        if remaining:
            _apply(remaining, "most_frequent")

    missing_after = int(working[features].isna().sum().sum()) if features else 0

    report = {
        "strategy": strategy,
        "missing_before": missing_before,
        "total_imputed": int(sum(missing_before.values())),
        "missing_after": missing_after,
        "used_reference_dataset": reference_df is not None,
        "reference_columns_used": reference_columns_used,
    }
    return working, report
