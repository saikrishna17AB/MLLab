import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

class_names = iris.target_names

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert class labels into one-hot encoded vectors
# Example: class 0 becomes [1, 0, 0]
num_classes = len(class_names)

Y_train = np.eye(num_classes)[y_train]

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, Y_train)

# Predict continuous scores for the test samples
y_scores = model.predict(X_test)

# Select the class having the highest score
y_pred = np.argmax(y_scores, axis=1)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
