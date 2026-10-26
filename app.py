import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="",
    layout="centered"
)

# Connect to the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

st.title("House Price Prediction")
st.write(
    "Enter the area of a house to estimate its price "
    "using a trained machine-learning model."
)

@st.cache_resource
def load_model():
    # Find the experiment
    experiment = mlflow.get_experiment_by_name(
        "House Price Prediction"
    )

    if experiment is None:
        raise RuntimeError(
            "MLflow experiment not found."
        )

    # Find the latest completed Linear Regression run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "and tags.mlflow.runName = 'Linear Regression'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise RuntimeError(
            "No saved Linear Regression model found."
        )

    run_id = runs.iloc[0]["run_id"]

    # Load the saved model
    model_uri = f"runs:/{run_id}/model"

    return mlflow.sklearn.load_model(model_uri)


# Load the model
try:
    model = load_model()
    st.success("Machine-learning model loaded successfully.")

    # Input area
    area = st.number_input(
        "House area (square feet)",
        min_value=1,
        max_value=100000,
        value=3000,
        step=100
    )

    # Predict button
    if st.button("Predict Price"):
        input_data = pd.DataFrame({
            "area": [float(area)]
        })

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.metric(
            label="Estimated House Price",
            value=f"₹{prediction:,.2f}"
        )

        st.caption(
            "This is a learning demonstration using a small "
            "synthetic dataset, not a reliable real-estate valuation."
        )

except Exception as error:
    st.error(f"Could not load the model: {error}")
