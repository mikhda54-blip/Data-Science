import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sales dataset
data = {
    "Advertising Spend": [10, 20, 30, 40, 50],
    "Sales": [100, 200, 300, 400, 500]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input feature
X = df[["Advertising Spend"]]

# Output
y = df["Sales"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict future sales
prediction = model.predict([[60]])

print("Predicted Sales:", prediction[0])

# Plot graph
plt.plot(
    df["Advertising Spend"],
    df["Sales"],
    marker='o'
)

# Graph title
plt.title("Sales Prediction")

# Labels
plt.xlabel("Advertising Spend")

plt.ylabel("Sales")

# Grid
plt.grid(True)

# Show graph
plt.show()