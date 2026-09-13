
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "model" / "sample_model.joblib"
DEFAULT_DATASET = ROOT / "dataset" / "test_dataset.csv"
DEFAULT_REPORT = ROOT / "reports" / "test_report.docx"

def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL))

def dataset_path() -> Path:
    return Path(os.environ.get("DATASET_PATH", DEFAULT_DATASET))

def label_column() -> str | None:
    value = os.environ.get("LABEL_COLUMN", "")
    return value or None

def report_path() -> Path:
    return Path(os.environ.get("REPORT_PATH", DEFAULT_REPORT))

def reference_dataset_path() -> Path | None:
    value = os.environ.get("REFERENCE_DATASET_PATH", "")
    return Path(value) if value else None

def impute_strategy() -> str:
    return os.environ.get("IMPUTE_STRATEGY", "mean")
