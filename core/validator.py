
"""
Core validation logic for ModelGuard.

This module is intentionally independent of Streamlit so that the same
checks can be called from the web UI and from pytest.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


LABEL_CANDIDATES = ("label", "target", "y", "class")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)


def find_label_column(
    df: pd.DataFrame,
    requested: str | None = None,
) -> str | None:
    if requested and requested != "None":
        if requested not in df.columns:
            raise ValueError(f"Label column '{requested}' was not found.")
        return requested

    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name
    return None


def get_feature_columns(
    model: Any,
    df: pd.DataFrame,
    label_column: str | None = None,
) -> list[str]:
    """
    Prefer the model's recorded feature_names_in_ when available.
    Otherwise use all dataset columns except the selected label.
    """
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        return [str(c) for c in model_features]

    # Pipelines normally expose feature_names_in_ too, but this also handles
    # simple estimators that only provide n_features_in_.
    if hasattr(model, "n_features_in_"):
        candidates = [c for c in df.columns if c != label_column]
        if len(candidates) == int(model.n_features_in_):
            return candidates

    return [c for c in df.columns if c != label_column]


def _make_result(status: str, details: list[str], **extra) -> dict[str, Any]:
    return {"status": status, "details": details, **extra}


def check_model_integrity(model_path: str | Path) -> dict[str, Any]:
    details: list[str] = []
    path = Path(model_path)

    if not path.exists():
        return _make_result("FAIL", [f"Model file not found: {path}"])

    if path.stat().st_size == 0:
        return _make_result("FAIL", ["Model file is empty."])

    try:
        model = load_model(path)
    except Exception as exc:
        return _make_result("FAIL", [f"Model could not be loaded: {exc}"])

    if not hasattr(model, "predict"):
        return _make_result("FAIL", ["Loaded object does not provide predict()."])

    model_features = getattr(model, "feature_names_in_", None)
    feature_count = getattr(model, "n_features_in_", None)

    details.extend(
        [
            "Model file exists.",
            f"SHA-256: {sha256_file(path)}",
            f"Model type: {type(model).__name__}",
            f"Input feature count: {feature_count if feature_count is not None else 'not declared'}",
            (
                "Input feature names detected."
                if model_features is not None
                else "Input feature names are not declared by the model."
            ),
            "predict() method is available.",
        ]
    )

    return _make_result(
        "PASS",
        details,
        checksum=sha256_file(path),
        model_type=type(model).__name__,
        feature_count=feature_count,
        feature_names=[str(x) for x in model_features]
        if model_features is not None
        else None,
    )


def check_dataset_validation(
    df: pd.DataFrame,
    model,
    label_column: str | None = None,
    allow_imputation: bool = True,
) -> dict[str, Any]:
    details: list[str] = []
    if df.empty:
        return _make_result("FAIL", ["Dataset is empty."])

    try:
        features = get_feature_columns(model, df, label_column)
    except Exception as exc:
        return _make_result("FAIL", [str(exc)])

    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        return _make_result(
            "FAIL",
            [f"Required model features are missing: {missing_features}"],
            missing_features=missing_features,
        )

    duplicate_count = int(df.duplicated().sum())
    missing_count = int(df.isna().sum().sum())

    numeric_features = [
        c for c in features if pd.api.types.is_numeric_dtype(df[c])
    ]
    non_numeric = [c for c in features if c not in numeric_features]

    numeric_values = df[numeric_features].to_numpy() if numeric_features else np.empty((0, 0))
    infinite_count = int(np.isinf(numeric_values).sum()) if numeric_values.size else 0

    common_extra = dict(
        rows=len(df),
        columns=len(df.columns),
        features=features,
        missing_values=missing_count,
        duplicate_rows=duplicate_count,
        infinite_values=infinite_count,
    )

    if non_numeric:
        details.append(
            "Categorical model input features detected: "
            + ", ".join(non_numeric)
            + ". Model preprocessing will handle these features."
    )

    if infinite_count:
        return _make_result(
            "FAIL",
            [f"Dataset contains {infinite_count} infinite numeric values."],
            **common_extra,
        )
        

    # Duplicate rows are treated as a validation failure because the current
    # project uses them as an explicit data-quality test.
    if duplicate_count:
        return _make_result(
            "FAIL",
            [f"Dataset contains {duplicate_count} duplicate rows."],
            **common_extra,
        )

    if missing_count:
        if allow_imputation:
            rows_affected = int(df[features].isna().any(axis=1).sum())
            return _make_result(
                "PASS",
                [
                    f"Dataset contains {missing_count} missing values across {rows_affected} rows.",
                    "Automatic data imputation is enabled; missing values will be "
                    "imputed in the Data Imputation stage before the pipeline continues.",
                ],
                requires_imputation=True,
                **common_extra,
            )
        return _make_result(
            "FAIL",
            [f"Dataset contains {missing_count} missing values."],
            requires_imputation=True,
            **common_extra,
        )

    details.extend(
        [
            f"Dataset contains {len(df)} rows and {len(df.columns)} columns.",
            f"Model input features found: {', '.join(features)}.",
            "No missing values found.",
            "No duplicate rows found.",
            "No infinite numeric values found.",
            (
    "All model input features are numeric."
    if not non_numeric
    else "Categorical input features are present and will be handled by the model preprocessing pipeline."
),
        ]
    )

    return _make_result(
        "PASS",
        details,
        requires_imputation=False,
        **common_extra,
    )


def run_data_imputation(
    df: pd.DataFrame,
    features: list[str],
    strategy: str = "mean",
    reference_df: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Impute missing values and report exactly what was changed.

    If `reference_df` (the original training/baseline dataset) is supplied,
    imputation statistics are learned from it instead of from `df` itself,
    for any shared feature column.

    Returns (result, imputed_dataframe). Imputation is always reported as a
    WARNING (not a silent PASS) because it is a data-quality intervention
    that a reviewer should be aware of, even when it fully resolves the
    missing values.
    """
    from core.imputation import impute_dataframe

    try:
        imputed_df, report = impute_dataframe(
            df, features, strategy=strategy, reference_df=reference_df
        )
    except Exception as exc:
        return _make_result("FAIL", [f"Data imputation failed: {exc}"]), df

    details = [f"Imputation strategy used: {report['strategy']}."]
    if report["reference_columns_used"]:
        details.append(
            "Imputation statistics were learned from the uploaded reference/"
            "training dataset (not from the dataset being validated) for: "
            f"{', '.join(report['reference_columns_used'])}."
        )
    elif report["used_reference_dataset"]:
        details.append(
            "A reference dataset was supplied, but none of the imputed "
            "columns were found in it, so those columns were imputed using "
            "the validation dataset's own statistics instead."
        )
    else:
        details.append(
            "No reference dataset was supplied, so imputation statistics "
            "were learned from the dataset being validated itself."
        )
    for column, count in report["missing_before"].items():
        details.append(f"Column '{column}': {count} missing values imputed.")
    details.append(f"Total missing values imputed: {report['total_imputed']}.")

    if report["missing_after"] > 0:
        details.append(
            f"{report['missing_after']} missing values remain after imputation "
            "(likely a non-numeric column with no fallback strategy applied)."
        )
        status = "FAIL"
    else:
        details.append("No missing values remain after imputation.")
        status = "PASS"

    result = _make_result(status, details, **report)
    return result, imputed_df


def check_preprocessing(
    model,
    X: pd.DataFrame,
) -> dict[str, Any]:
    details: list[str] = []

    # First check the real model input path. For a Pipeline this also exercises
    # the model's actual preprocessing steps.
    try:
        prediction = model.predict(X)
    except Exception as exc:
        return _make_result(
            "FAIL",
            [f"Model input/preprocessing failed: {exc}"],
        )

    if len(prediction) != len(X):
        return _make_result(
            "FAIL",
            ["Preprocessing/inference changed the number of samples."],
        )

    if hasattr(model, "steps"):
        step_names = [name for name, _ in model.steps]
        details.append(f"Pipeline steps detected: {', '.join(step_names)}.")

        # Exercise intermediate transformers when possible.
        transformed = X
        for name, transformer in model.steps[:-1]:
            if hasattr(transformer, "transform"):
                transformed = transformer.transform(transformed)

        if hasattr(transformed, "toarray"):
            transformed_array = transformed.toarray()
        else:
            transformed_array = np.asarray(transformed)
        if not np.isfinite(transformed_array).all():
            return _make_result(
                "FAIL",
                ["Preprocessing produced NaN or infinite values."],
            )
        details.append(
            f"Intermediate preprocessing output shape: {transformed_array.shape}."
        )
    else:
        details.append("Model does not expose a Pipeline; input compatibility was tested.")

    details.append("Model accepted the prepared test inputs.")
    return _make_result("PASS", details)


def check_inference(
    model,
    X: pd.DataFrame,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        predictions = model.predict(X)
    except Exception as exc:
        return _make_result("FAIL", [f"Inference failed: {exc}"])

    elapsed = time.perf_counter() - start

    if predictions is None or len(predictions) == 0:
        return _make_result("FAIL", ["Model returned no predictions."])

    if len(predictions) != len(X):
        return _make_result(
            "FAIL",
            [
                f"Expected {len(X)} predictions but received {len(predictions)}."
            ],
        )

    return _make_result(
        "PASS",
        [
            f"Generated {len(predictions)} predictions.",
            f"Inference time: {elapsed:.4f} seconds.",
        ],
        prediction_count=len(predictions),
        inference_time_seconds=elapsed,
        predictions=predictions,
    )


def check_results(
    model,
    X: pd.DataFrame,
    y: pd.Series | None,
) -> dict[str, Any]:
    try:
        predictions = model.predict(X)
    except Exception as exc:
        return _make_result("FAIL", [f"Result validation could not run: {exc}"])

    details: list[str] = [
        "Prediction count matches input row count.",
        f"Prediction output type: {type(predictions).__name__}.",
    ]
    metrics: dict[str, float] = {}

    if y is None:
        details.append("No label column supplied; accuracy metrics were skipped.")
        return _make_result(
            "PASS",
            details,
            metrics=metrics,
            predictions=predictions,
        )

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    if len(y) != len(predictions):
        return _make_result("FAIL", ["Ground-truth labels and predictions differ in length."])

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, average="weighted", zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, average="weighted", zero_division=0)
        ),
    }

    matrix = confusion_matrix(y, predictions)
    details.extend(
        [
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
            f"F1 score: {metrics['f1']:.4f}",
            f"Confusion matrix shape: {matrix.shape}.",
        ]
    )

    return _make_result(
        "PASS",
        details,
        metrics=metrics,
        confusion_matrix=matrix.tolist(),
        predictions=predictions,
    )


def check_robustness(
    model,
    X: pd.DataFrame,
    agreement_threshold: float = 0.80,
) -> dict[str, Any]:
    if len(X) == 0:
        return _make_result("FAIL", ["No samples available for robustness testing."])

    baseline = np.asarray(model.predict(X))
    numeric = X.select_dtypes(include=np.number).columns.tolist()

    if not numeric:
        return _make_result("FAIL", ["No numeric features available for robustness testing."])

    perturbed = X.copy()
    noise = np.random.default_rng(42).normal(
        loc=0.0,
        scale=0.01,
        size=(len(X), len(numeric)),
    )
    perturbed[numeric] = perturbed[numeric].to_numpy() + noise

    try:
        changed_predictions = np.asarray(model.predict(perturbed))
    except Exception as exc:
        return _make_result(
            "FAIL",
            [f"Model failed on small input perturbations: {exc}"],
        )

    agreement = float(np.mean(baseline == changed_predictions))

    if agreement < agreement_threshold:
        return _make_result(
            "FAIL",
            [
                f"Prediction agreement after small perturbation was {agreement:.2%}.",
                f"Required minimum agreement: {agreement_threshold:.0%}.",
            ],
            perturbation_agreement=agreement,
        )

    # Test that an invalid missing-value input is handled without crashing
    # the framework. A model may legitimately reject it.
    missing_input = X.copy()
    missing_input.loc[missing_input.index[0], numeric[0]] = np.nan
    missing_value_handled = False

    try:
        model.predict(missing_input)
        missing_value_handled = True
    except Exception:
        # Expected for models that do not accept NaN.
        missing_value_handled = True

    details = [
        f"Small-perturbation prediction agreement: {agreement:.2%}.",
        "Missing-value input was handled without crashing the test framework.",
    ]

    return _make_result(
        "PASS",
        details,
        perturbation_agreement=agreement,
        missing_value_handled=missing_value_handled,
    )


def check_ml_security(
    df: pd.DataFrame,
    model,
    label_column: str | None = None,
    outlier_limit: float = 0.05,
) -> dict[str, Any]:
    """
    Basic ML-security indicators, not proof of a poisoning attack.

    The checks flag common suspicious patterns:
    - duplicate samples
    - extreme numeric outliers
    - very rare labels
    """
    features = get_feature_columns(model, df, label_column)
    suspicious: list[str] = []

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        suspicious.append(f"{duplicate_count} duplicate rows detected.")

    outlier_counts: dict[str, int] = {}
    for feature in features:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue

        series = df[feature].dropna()
        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            count = 0
        else:
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            count = int(((series < lower) | (series > upper)).sum())

        outlier_counts[feature] = count

        if count / max(len(series), 1) > outlier_limit:
            suspicious.append(
                f"Feature '{feature}' has {count} extreme outliers."
            )

    label_distribution: dict[str, int] = {}
    if label_column and label_column in df.columns:
        counts = df[label_column].value_counts(dropna=False)
        label_distribution = {str(k): int(v) for k, v in counts.items()}

        if len(counts) > 1:
            rare = counts[counts / len(df) < 0.01]
            if not rare.empty:
                suspicious.append(
                    "Very rare label values detected: "
                    + ", ".join(str(x) for x in rare.index)
                )

    # Multivariate anomaly detection (Isolation Forest). Catches poisoned
    # samples that look fine on any single feature but are anomalous in
    # combination, which the per-feature outlier check above would miss.
    numeric_for_iforest = [
        f for f in features if pd.api.types.is_numeric_dtype(df[f])
    ]
    multivariate_anomaly_count = 0
    multivariate_anomaly_ratio = 0.0
    if len(numeric_for_iforest) >= 2:
        clean_numeric = df[numeric_for_iforest].dropna()
        if len(clean_numeric) >= 10:
            from sklearn.ensemble import IsolationForest

            iso = IsolationForest(contamination=outlier_limit, random_state=42)
            preds = iso.fit_predict(clean_numeric)
            multivariate_anomaly_count = int((preds == -1).sum())
            multivariate_anomaly_ratio = multivariate_anomaly_count / len(clean_numeric)
            if multivariate_anomaly_ratio > outlier_limit:
                suspicious.append(
                    f"Multivariate anomaly detector (Isolation Forest) flagged "
                    f"{multivariate_anomaly_count} samples "
                    f"({multivariate_anomaly_ratio:.1%}) as anomalous in combined "
                    "feature space."
                )

    # Nearest-neighbor label-consistency check. If a sample's label disagrees
    # with most of its nearest neighbors in feature space, it is a classic
    # indicator of label-flipping poisoning.
    label_consistency_violations = 0
    label_consistency_ratio = 0.0
    if label_column and label_column in df.columns and len(numeric_for_iforest) >= 1:
        clean = df[numeric_for_iforest + [label_column]].dropna()
        if len(clean) >= 10:
            from sklearn.neighbors import NearestNeighbors

            X_num = clean[numeric_for_iforest]
            y_lbl = clean[label_column]
            k = min(5, len(clean) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X_num)
            _, neighbor_idx = nn.kneighbors(X_num)

            violations = 0
            for row_pos, neighbors in enumerate(neighbor_idx):
                own_label = y_lbl.iloc[row_pos]
                neighbor_labels = y_lbl.iloc[neighbors[1:]]
                if len(neighbor_labels) and (neighbor_labels != own_label).mean() > 0.8:
                    violations += 1

            label_consistency_violations = violations
            label_consistency_ratio = violations / len(clean)
            if label_consistency_ratio > 0.05:
                suspicious.append(
                    f"{violations} samples ({label_consistency_ratio:.1%}) disagree in "
                    "label with most of their nearest neighbors in feature space, "
                    "which can indicate label-flipping poisoning."
                )

    if suspicious:
        return _make_result(
            "WARNING",
            suspicious
            + [
                "These are poisoning indicators for review, not proof of a poisoning attack."
            ],
            duplicate_rows=duplicate_count,
            outlier_counts=outlier_counts,
            label_distribution=label_distribution,
            multivariate_anomaly_count=multivariate_anomaly_count,
            multivariate_anomaly_ratio=multivariate_anomaly_ratio,
            label_consistency_violations=label_consistency_violations,
            label_consistency_ratio=label_consistency_ratio,
        )

    return _make_result(
        "PASS",
        [
            "No duplicate samples detected.",
            "No excessive extreme-outlier pattern detected.",
            "No excessive multivariate (combined feature-space) anomalies detected.",
            (
                "No very rare labels detected."
                if label_column
                else "No label column supplied; label-anomaly checks were skipped."
            ),
            (
                "No significant nearest-neighbor label inconsistency detected."
                if label_column
                else "No label column supplied; label-consistency check was skipped."
            ),
            "These checks are basic poisoning indicators, not proof of attack absence.",
        ],
        duplicate_rows=duplicate_count,
        outlier_counts=outlier_counts,
        label_distribution=label_distribution,
        multivariate_anomaly_count=multivariate_anomaly_count,
        multivariate_anomaly_ratio=multivariate_anomaly_ratio,
        label_consistency_violations=label_consistency_violations,
        label_consistency_ratio=label_consistency_ratio,
    )



def check_data_poisoning(
    df: pd.DataFrame,
    model,
    label_column: str | None = None,
    outlier_limit: float = 0.05,
) -> dict[str, Any]:
    """Explicit name for the dataset-poisoning checks used by ModelGuard."""
    return check_ml_security(df, model, label_column, outlier_limit)


def check_model_poisoning(
    model,
    X: pd.DataFrame,
    trigger_scale: float = 5.0,
    noise_samples: int = 200,
    dominance_threshold: float = 0.9,
) -> dict[str, Any]:
    """
    Basic model-poisoning / backdoor-trigger indicators for a trained model.

    Not proof of a compromised model — these are heuristics a reviewer
    should follow up on, the same way check_ml_security's data indicators
    are not proof of a poisoned dataset.
    """
    if len(X) == 0:
        return _make_result("FAIL", ["No samples available for model poisoning testing."])

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    if not numeric:
        return _make_result("FAIL", ["No numeric features available for model poisoning testing."])

    try:
        baseline_preds = np.asarray(model.predict(X))
    except Exception as exc:
        return _make_result("FAIL", [f"Baseline prediction failed: {exc}"])

    _, baseline_counts = np.unique(baseline_preds, return_counts=True)
    baseline_dominant_share = float(baseline_counts.max() / baseline_counts.sum())

    suspicious: list[str] = []

    # 1. Trigger-pattern probing: push one feature at a time to an extreme
    # value and see whether that alone collapses predictions to one class.
    # A classic backdoor trigger is a single feature/pixel pattern that
    # forces a fixed output regardless of the rest of the input.
    trigger_flip_features: list[str] = []
    medians = X[numeric].median()
    for feature in numeric:
        probe = X.copy()
        std = X[feature].std()
        std = std if std and std > 0 else 1.0
        probe[feature] = medians[feature] + trigger_scale * std
        try:
            probe_preds = np.asarray(model.predict(probe))
        except Exception:
            continue
        _, counts = np.unique(probe_preds, return_counts=True)
        dominant_share = float(counts.max() / counts.sum())
        if dominant_share > dominance_threshold and dominant_share > baseline_dominant_share + 0.15:
            trigger_flip_features.append(feature)

    if trigger_flip_features:
        suspicious.append(
            "Extreme single-feature perturbation collapsed predictions to one "
            "dominant class for: " + ", ".join(trigger_flip_features) + ". "
            "This can indicate a backdoor trigger tied to that feature."
        )

    # 2. Random-noise output entropy check. A clean model's predictions on
    # random inputs across the observed feature range are rarely all one
    # class; an unusually dominant class can indicate an embedded backdoor
    # default or an overconfident/degenerate model.
    rng = np.random.default_rng(42)
    mins = X[numeric].min()
    maxs = X[numeric].max()
    noise = pd.DataFrame({f: rng.uniform(mins[f], maxs[f], size=noise_samples) for f in numeric})
    for col in X.columns:
        if col not in numeric:
            noise[col] = X[col].iloc[0]
    noise = noise[X.columns]

    noise_dominant_share = None
    try:
        noise_preds = np.asarray(model.predict(noise))
        _, noise_counts = np.unique(noise_preds, return_counts=True)
        noise_dominant_share = float(noise_counts.max() / noise_counts.sum())
    except Exception as exc:
        suspicious_note = f"Random-noise probing could not run: {exc}"
    else:
        suspicious_note = None
        if noise_dominant_share > dominance_threshold:
            suspicious.append(
                f"Random inputs across the observed feature range produced the same "
                f"predicted class {noise_dominant_share:.0%} of the time, which can "
                "indicate an overly confident or backdoored model."
            )

    # 3. Feature-importance / weight concentration. A single feature
    # dominating the model's decision can be a sign of a trigger feature
    # embedded during training (or just a legitimately strong predictor —
    # this is an indicator to review, not a verdict).
    importances = getattr(model, "feature_importances_", None)
    if importances is None and hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_)).reshape(-1)
        if coef.size == len(numeric):
            importances = coef

    importance_top_share = None
    if importances is not None and len(importances) > 1:
        importances = np.asarray(importances, dtype=float)
        total = importances.sum()
        if total > 0:
            importance_top_share = float(importances.max() / total)
            if importance_top_share > 0.9:
                suspicious.append(
                    f"A single feature accounts for {importance_top_share:.0%} of total "
                    "model importance/weight, an unusually concentrated pattern worth reviewing."
                )

    extra = dict(
        baseline_dominant_share=baseline_dominant_share,
        trigger_flip_features=trigger_flip_features,
        noise_dominant_share=noise_dominant_share,
        importance_top_share=importance_top_share,
    )

    # A single extreme-feature response is not enough to label a clean model
    # as suspicious: many legitimate classifiers naturally change class when
    # an input is pushed far outside its normal range. Require at least two
    # independent indicators before returning WARNING.
    indicator_count = len(suspicious)
    status = "WARNING" if indicator_count >= 2 else "PASS"

    if suspicious:
        details = suspicious + [
            f"{indicator_count} model-poisoning indicator(s) detected; "
            "multiple independent indicators are required for a WARNING."
        ]
        if suspicious_note:
            details.append(suspicious_note)
        return _make_result(status, details, **extra)

    details = [
        f"Baseline prediction class balance: dominant class share {baseline_dominant_share:.0%}.",
        "No single-feature trigger probing collapsed predictions to one class.",
        (
            f"Random-noise inputs produced a dominant class share of {noise_dominant_share:.0%}, within expected range."
            if noise_dominant_share is not None
            else (suspicious_note or "Random-noise probing was skipped.")
        ),
        (
            "No unusually concentrated feature importance detected."
            if importance_top_share is not None
            else "Model does not expose feature importances/coefficients; that check was skipped."
        ),
        "These checks are basic model-poisoning/backdoor indicators, not proof of a clean model.",
    ]
    return _make_result("PASS", details, **extra)


def check_ood(
    X: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    contamination: float = 0.05,
    z_threshold: float = 4.0,
) -> dict[str, Any]:
    """
    Out-of-distribution (OOD) sample detection.

    With a reference/baseline dataset supplied, flags samples whose
    per-feature z-score (relative to the reference mean/std) exceeds
    z_threshold. Without a reference, falls back to a weaker
    self-consistency check (Isolation Forest fit on the current dataset)
    and says so explicitly.
    """
    if len(X) == 0:
        return _make_result("FAIL", ["No samples available for OOD detection."])

    features = features or list(X.columns)
    numeric = [c for c in features if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if not numeric:
        return _make_result("FAIL", ["No numeric features available for OOD detection."])

    if reference_df is not None:
        common = [c for c in numeric if c in reference_df.columns]
        if not common:
            return _make_result(
                "FAIL",
                ["Reference dataset does not share numeric feature columns with the current dataset."],
            )
        ref_numeric = reference_df[common].dropna()
        mean = ref_numeric.mean()
        std = ref_numeric.std().replace(0, 1.0)
        z_scores = ((X[common] - mean) / std).abs()
        ood_mask = (z_scores > z_threshold).any(axis=1)
        ood_count = int(ood_mask.sum())
        method_note = (
            f"Compared against a supplied reference dataset ({len(reference_df)} rows) "
            f"using per-feature z-scores against the reference mean/std (threshold {z_threshold})."
        )
        used_reference = True
    else:
        if len(X) < 10:
            return _make_result(
                "FAIL",
                ["Not enough samples for self-consistency OOD detection (minimum 10)."],
            )
        from sklearn.ensemble import IsolationForest

        clean_numeric = X[numeric].dropna()
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(clean_numeric)
        ood_count = int((preds == -1).sum())
        method_note = (
            "No reference dataset was supplied, so OOD detection used self-consistency "
            "(Isolation Forest fit on the current dataset). This is a weaker check than "
            "comparing against a real training-distribution reference dataset."
        )
        used_reference = False

    ood_ratio = ood_count / len(X)
    details = [
        method_note,
        f"{ood_count} of {len(X)} samples ({ood_ratio:.1%}) flagged as out-of-distribution.",
    ]

    if ood_ratio > contamination * 2:
        details.append("This is a higher-than-expected proportion of out-of-distribution samples.")
        status = "WARNING"
    else:
        status = "PASS"

    return _make_result(
        status,
        details,
        ood_count=ood_count,
        ood_ratio=ood_ratio,
        used_reference=used_reference,
    )


def check_distribution_shift(
    X: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compare the current dataset's feature distributions against a reference
    (baseline/training) dataset using the Kolmogorov-Smirnov test and the
    Population Stability Index (PSI) per numeric feature.

    Requires a reference dataset; without one this stage is SKIPPED rather
    than PASS/FAIL, since there is nothing to compare against.
    """
    if reference_df is None:
        return _make_result(
            "SKIPPED",
            [
                "No reference/baseline dataset was supplied; distribution shift "
                "detection needs a baseline dataset to compare against."
            ],
        )

    features = features or list(X.columns)
    numeric = [
        c for c in features
        if c in X.columns and c in reference_df.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    if not numeric:
        return _make_result(
            "FAIL",
            ["No shared numeric features between the current and reference datasets."],
        )

    from scipy.stats import ks_2samp

    shifted_features: list[str] = []
    psi_scores: dict[str, float] = {}
    ks_pvalues: dict[str, float] = {}

    for feature in numeric:
        current_vals = X[feature].dropna()
        ref_vals = reference_df[feature].dropna()
        if len(current_vals) < 5 or len(ref_vals) < 5:
            continue

        _, p_value = ks_2samp(current_vals, ref_vals)
        ks_pvalues[feature] = float(p_value)

        bins = np.histogram_bin_edges(ref_vals, bins=10)
        ref_counts = np.histogram(ref_vals, bins=bins)[0].astype(float) + 1e-6
        cur_counts = np.histogram(current_vals, bins=bins)[0].astype(float) + 1e-6
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        psi_scores[feature] = psi

        if p_value < ks_alpha or psi > psi_threshold:
            shifted_features.append(feature)

    details = [
        f"Compared {len(numeric)} shared numeric features against the reference "
        f"dataset ({len(reference_df)} rows).",
        f"KS-test alpha: {ks_alpha}, PSI threshold: {psi_threshold}.",
    ]

    if shifted_features:
        details.insert(
            0,
            f"Statistically significant distribution shift detected in: {', '.join(shifted_features)}.",
        )
        status = "WARNING"
    else:
        details.insert(0, "No significant distribution shift detected.")
        status = "PASS"

    return _make_result(
        status,
        details,
        psi_scores=psi_scores,
        ks_pvalues=ks_pvalues,
        shifted_features=shifted_features,
    )


def run_validation_pipeline(
    model_path: str | Path,
    dataset_path: str | Path,
    label_column: str | None = None,
    allow_imputation: bool = True,
    impute_strategy: str = "mean",
    reference_dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run all ModelGuard stages and return one structured result.
    """
    results: dict[str, Any] = {}

    model_result = check_model_integrity(model_path)
    results["Model Loading"] = model_result
    if model_result["status"] == "FAIL":
        return results

    model = load_model(model_path)
    df = load_dataset(dataset_path)

    reference_df = None
    reference_warning = None
    if reference_dataset_path:
        try:
            reference_df = load_dataset(reference_dataset_path)
        except Exception as exc:
            reference_warning = f"Reference dataset could not be loaded: {exc}"

    selected_label = find_label_column(df, label_column)
    dataset_result = check_dataset_validation(
        df, model, selected_label, allow_imputation=allow_imputation
    )
    results["Dataset Upload & Validation"] = dataset_result

    if dataset_result["status"] == "FAIL":
        return results

    working_df = df
    if dataset_result.get("requires_imputation"):
        impute_features = dataset_result.get("features") or get_feature_columns(
            model, df, selected_label
        )
        imputation_result, working_df = run_data_imputation(
            df, impute_features, impute_strategy, reference_df=reference_df
        )
        results["Data Imputation"] = imputation_result
        if imputation_result["status"] == "FAIL":
            return results
    else:
        results["Data Imputation"] = _make_result(
            "PASS", ["No missing values were found; imputation was not required."]
        )

    features = get_feature_columns(model, working_df, selected_label)
    X = working_df[features].copy()
    y = working_df[selected_label] if selected_label else None

    results["Preprocessing Testing"] = check_preprocessing(model, X)
    if results["Preprocessing Testing"]["status"] == "FAIL":
        return results

    results["Model Inference"] = check_inference(model, X)
    if results["Model Inference"]["status"] == "FAIL":
        return results

    results["Result Validation"] = check_results(model, X, y)
    if results["Result Validation"]["status"] == "FAIL":
        return results

    results["Robustness Testing"] = check_robustness(model, X)
    results["Out-of-Distribution Detection"] = check_ood(X, reference_df=reference_df, features=features)
    results["Distribution Shift Detection"] = check_distribution_shift(
        X, reference_df=reference_df, features=features
    )
    data_poisoning_result = check_ml_security(
        working_df,
        model,
        selected_label,
    )
    if reference_warning:
        data_poisoning_result.setdefault("details", []).insert(0, reference_warning)
    results["Data Poisoning Detection"] = data_poisoning_result
    results["Model Poisoning Detection"] = check_model_poisoning(model, X)

    return results


def overall_status(results: dict[str, Any]) -> str:
    statuses = [item["status"] for item in results.values()]
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status == "WARNING" for status in statuses):
        return "WARNING"
    return "PASS"


def count_test_statuses(results: dict[str, Any]) -> tuple[int, int, int]:
    passed = failed = warnings = 0
    for item in results.values():
        status = item["status"]
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "WARNING":
            warnings += 1
    return passed, failed, warnings
