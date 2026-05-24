import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create dataset
data = {
    "Horsepower": [100, 120, 150, 180, 200],
    "Mileage": [20, 18, 15, 12, 10],
    "Age": [5, 4, 3, 2, 1],
    "Price": [500000, 600000, 800000, 1000000, 1200000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input features
X = df[["Horsepower", "Mileage", "Age"]]

# Output
y = df["Price"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price
prediction = model.predict([[170, 14, 2]])

print("Predicted Car Price:", prediction[0])

# Graph
plt.plot(df["Horsepower"], df["Price"], marker='o')

plt.title("Car Price Prediction")

plt.xlabel("Horsepower")

plt.ylabel("Price")

plt.grid(True)

plt.show()