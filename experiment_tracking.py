import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import csv
import os
from datetime import datetime

# Sample dataset
data = {
    "area": [500, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000],
    "price": [
        2500000, 4000000, 5000000,
        6000000, 7500000, 9000000,
        10000000, 12500000, 15000000
    ]
}

df = pd.DataFrame(data)

X = df[["area"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

results_file = "experiment_results.csv"

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": name,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "mae": round(mae, 2)
    }

    file_exists = os.path.exists(results_file)

    with open(results_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    print(f"Model: {name}")
    print(f"MAE: ₹{mae:,.2f}")
    print("-" * 30)

print(f"Experiment results saved to {results_file}")
