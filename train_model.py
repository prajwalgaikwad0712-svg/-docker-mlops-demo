import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample house data
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

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Input feature and target
X = df[["area"]]
y = df["price"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print("Trained model saved as house_price_model.pkl")
