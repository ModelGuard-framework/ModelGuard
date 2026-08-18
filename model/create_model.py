import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load dataset
data = pd.read_csv("dataset/test_dataset.csv")

# Separate features and ground-truth label
X = data.drop("label", axis=1)
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a neural-network model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("neural_network", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=500,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model/sample_model.joblib")

print("Sample neural-network model trained successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model saved as: model/sample_model.joblib")