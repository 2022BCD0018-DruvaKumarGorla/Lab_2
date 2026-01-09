import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = "dataset/winequality-red.csv"

df = pd.read_csv(DATA_PATH, sep=';')

# -----------------------------
# 2. Preprocessing & Feature Selection
# -----------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. Train the Model
# -----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# 4. Evaluate the Model
# -----------------------------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -----------------------------
# 5. Save Outputs
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# Save trained model (model + scaler)
joblib.dump(
    {"model": model, "scaler": scaler},
    "outputs/model.pkl"
)

# Save metrics
results = {
    "MSE": mse,
    "R2": r2
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)

# -----------------------------
# 6. Print Metrics
# -----------------------------
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
