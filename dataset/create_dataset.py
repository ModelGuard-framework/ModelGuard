import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 500

data = pd.DataFrame({
    "feature_1": np.random.normal(50, 10, n_samples),
    "feature_2": np.random.normal(30, 5, n_samples),
    "feature_3": np.random.normal(70, 15, n_samples),
    "feature_4": np.random.normal(20, 4, n_samples)
})

# Create binary target
data["label"] = (
    (data["feature_1"] + data["feature_2"] > 80)
    .astype(int)
)

data.to_csv("test_dataset.csv", index=False)

print("Test dataset created successfully!")
print(f"Number of records: {len(data)}")
print(f"Columns: {list(data.columns)}")