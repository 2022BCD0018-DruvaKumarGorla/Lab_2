import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1. Load Dataset
df = pd.read_csv("dataset/winequality-red.csv", sep=';')

# 2. Feature Selection
X = df.drop("quality", axis=1)
y = df["quality"]

# 70/30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 5. Save Outputs
os.makedirs("outputs", exist_ok=True)
joblib.dump({"model": model}, "outputs/model.pkl")

with open("outputs/results.json", "w") as f:
    json.dump({"MSE": mse, "R2": r2}, f, indent=4)

# 6. Print Metrics
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
