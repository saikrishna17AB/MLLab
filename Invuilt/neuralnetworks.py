from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

# Create the Neural Network model
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),  # Two hidden layers
    activation="relu",  # ReLU activation
    solver="adam",  # Adam optimizer
    max_iter=2000,
    random_state=42,
)

# Train the neural network
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

# Display number of input, hidden and output neurons
print("\nNumber of neurons in each layer:")
print(model.n_layers_)
print(model.hidden_layer_sizes)
print(model.n_outputs_)
