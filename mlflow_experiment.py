import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Sample house-price dataset
data = {
    "area": [500, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000],
    "price": [
        2500000, 4000000, 5000000,
        6000000, 7500000, 9000000,
        10000000, 12500000, 15000000
    ]
}

df = pd.DataFrame(data)

# 2. Separate input and output
X = df[["area"]]
y = df["price"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# 4. Create or select the MLflow experiment
mlflow.set_experiment("House Price Prediction")

# 5. Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

# 6. Train, evaluate, and save each model
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train the model
        model.fit(X_train, y_train)

        # Generate predictions
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Record metrics in MLflow
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        # Record parameters
        mlflow.log_param("model_type", name)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("testing_samples", len(X_test))

        # Save the trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="pickle"
        )

        # Display results
        print(f"\nModel: {name}")
        print(f"MAE: ₹{mae:,.2f}")
        print(f"MSE: {mse:,.2f}")
        print(f"R2 Score: {r2:.4f}")
        print("Model saved successfully in MLflow.")
        print("-" * 40)

print("\nAll experiments completed successfully!")
print("Metrics and models have been recorded in MLflow.")
