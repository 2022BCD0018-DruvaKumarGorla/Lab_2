import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# 1. Load Dataset
# -----------------------------
DATASET_PATH = "dataset/winequality-red.csv"
df = pd.read_csv(DATASET_PATH, sep=';')


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# 4. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -----------------------------
# 5. Save Outputs
# -----------------------------
os.makedirs("outputs", exist_ok=True)

joblib.dump(
    {"model": model, "scaler": scaler},
    "outputs/model.pkl"
)

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
