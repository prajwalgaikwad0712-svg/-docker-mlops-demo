import mlflow
import mlflow.sklearn
import pandas as pd

# Connect to the local MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Find the House Price Prediction experiment
experiment = mlflow.get_experiment_by_name(
    "House Price Prediction"
)

if experiment is None:
    raise RuntimeError(
        "Experiment not found. Run mlflow_experiment.py first."
    )

# Find the most recent successful Linear Regression run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED' "
                  "and tags.mlflow.runName = 'Linear Regression'",
    order_by=["attributes.start_time DESC"],
    max_results=1
)

if runs.empty:
    raise RuntimeError(
        "No completed Linear Regression run found."
    )

# Get the run ID
run_id = runs.iloc[0]["run_id"]

# Load the saved model from MLflow
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

print("=== House Price Prediction ===")
print("Model loaded successfully from MLflow.")

try:
    # Ask the user for the house area
    area = float(
        input("Enter house area in square feet: ")
    )

    if area <= 0:
        raise ValueError("Area must be greater than zero.")

    # Prepare the input
    input_data = pd.DataFrame({"area": [area]})

    # Predict the house price
    predicted_price = model.predict(input_data)[0]

    # Display the result
    print(f"\nHouse area: {area:,.0f} square feet")
    print(f"Predicted price: ₹{predicted_price:,.2f}")

except ValueError as error:
    print(f"Invalid input: {error}")
