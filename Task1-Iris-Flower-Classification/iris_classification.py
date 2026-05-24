from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

# Input data
X = iris.data

# Output labels
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = KNeighborsClassifier()

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Example prediction
sample = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(sample)

print("Predicted Flower Class:", iris.target_names[result][0])