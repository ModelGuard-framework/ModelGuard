
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "dataset" / "test_dataset.csv"

n_samples = 500

data = pd.DataFrame({
    "feature_1": np.random.normal(50, 10, n_samples),
    "feature_2": np.random.normal(30, 5, n_samples),
    "feature_3": np.random.normal(70, 15, n_samples),
    "feature_4": np.random.normal(20, 4, n_samples),
})

data["label"] = (
    (data["feature_1"] + data["feature_2"] > 80)
    .astype(int)
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_PATH, index=False)

print("Test dataset created successfully!")
print(f"Number of records: {len(data)}")
print(f"Columns: {list(data.columns)}")
print(f"Saved to: {OUTPUT_PATH}")
