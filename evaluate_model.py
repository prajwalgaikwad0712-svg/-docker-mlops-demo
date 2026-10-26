import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample house dataset
data = {
    "area": [500, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000],
    "price": [
        2500000,
        4000000,
        5000000,
        6000000,
        7500000,
        9000000,
        10000000,
        12500000,
        15000000
    ]
}

df = pd.DataFrame(data)

# Features and target
X = df[["area"]]
y = df["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train the model on training data only
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices for unseen test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Model Evaluation Report ===")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
