from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Create the KNN model
model = KNeighborsClassifier(
    n_neighbors=5,  # Consider 5 nearest neighbors
    metric="minkowski",
    p=2,  # p=2 means Euclidean distance
)

# Train the model
# KNN stores the training samples
model.fit(X_train, y_train)

# Predict class labels
y_pred = model.predict(X_test)

# Get probabilities for each class
y_prob = model.predict_proba(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display probabilities of first five samples
print("\nPredicted probabilities:")
print(y_prob[:5])
