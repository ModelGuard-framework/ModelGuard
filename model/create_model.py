
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "test_dataset.csv"
MODEL_PATH = ROOT / "model" / "sample_model.joblib"

data = pd.read_csv(DATASET_PATH)

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("neural_network", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=500,
        random_state=42,
    )),
])

model.fit(X_train, y_train)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Sample neural-network model trained successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model saved as:", MODEL_PATH)
