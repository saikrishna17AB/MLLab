from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create the Logistic Regression model
# lbfgs supports multinomial classification
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

# Train the model
model.fit(X_train, y_train)

# Predict class labels
y_pred = model.predict(X_test)

# Get probability of each class
y_prob = model.predict_proba(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display probabilities of the first five samples
print("\nPredicted probabilities:")
print(y_prob[:5])

# Display the learned coefficients
print("\nWeights:")
print(model.coef_)

print("\nBias:")
print(model.intercept_)from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create the Logistic Regression model
# lbfgs supports multinomial classification
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

# Train the model
model.fit(X_train, y_train)

# Predict class labels
y_pred = model.predict(X_test)

# Get probability of each class
y_prob = model.predict_proba(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display probabilities of the first five samples
print("\nPredicted probabilities:")
print(y_prob[:5])

# Display the learned coefficients
print("\nWeights:")
print(model.coef_)

print("\nBias:")
print(model.intercept_)