import pandas as pd
import matplotlib.pyplot as plt

# Create unemployment data
data = {
    "Year": [2018, 2019, 2020, 2021, 2022, 2023],
    "Unemployment Rate": [5.3, 5.1, 9.5, 8.2, 6.1, 5.4]
}

# Create DataFrame
df = pd.DataFrame(data)

# Print data
print(df)

# Plot graph
plt.plot(df["Year"], df["Unemployment Rate"], marker='o')

# Graph title
plt.title("Unemployment Rate Analysis")

# Labels
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")

# Grid
plt.grid(True)

# Show graph
plt.show()