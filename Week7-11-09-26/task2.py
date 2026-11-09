import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

    
# Read dataset
df = pd.read_csv("heart-disease.csv")


# Separate input features and target
X = df.drop("target", axis=1).values
y = df["target"]


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create scaler
scaler = StandardScaler()


# Scale training data
Xtrain = scaler.fit_transform(X_train)


# Scale testing data using the same scaler
Xtest = scaler.transform(X_test)


# Convert y from Pandas Series to NumPy arrays
ytrain = y_train.values
ytest = y_test.values






# Create a column of ones for training data
ones_train = np.ones((Xtrain.shape[0], 1))

# Add column of ones to training data
Xtrain = np.hstack((ones_train, Xtrain))


# Create a column of ones for testing data
ones_test = np.ones((Xtest.shape[0], 1))

# Add column of ones to testing data
Xtest = np.hstack((ones_test, Xtest))



# Sigmoid Function

def sigmoid(z):

    # Sigmoid formula
    return 1 / (1 + np.exp(-z))



# Cost Function


def cost_function(X, y, weights):

    # Calculate z = Xw
    z = np.dot(X, weights)

    # Calculate predicted probabilities
    predictions = sigmoid(z)

    # Number of training examples
    m = len(y)

    # Calculate Binary Cross Entropy cost
    cost = -(1 / m) * np.sum(y * np.log(predictions)+ (1 - y) * np.log(1 - predictions))

    return cost



# Gradient Descent


def gradient_descent(X, y, l, iterations):

    # Number of features including intercept
    n = X.shape[1]

    # Initially set all weights to zero
    weights = np.zeros(n)

    # Store cost after each iteration
    cost_history = []

    # Repeat the process
    for i in range(iterations):

        # Calculate z = Xw
        z = np.dot(X, weights)

        # Calculate predicted probabilities
        predictions = sigmoid(z)

        # Calculate gradient   
        gradient = (1 / len(y)) * np.dot( X.T,predictions - y)

        # Update weights
        weights = weights - l * gradient

        # Calculate current cost
        cost = cost_function(X, y, weights)
        
        # Check convergence
        

        # Store cost
        cost_history.append(cost)
        
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < 0.000001:
            break

    return weights, cost_history


# Train Gradient Descent model
gd_weights, gd_cost = gradient_descent(Xtrain,ytrain,0.01,1000)



# Prediction Function


def predict(X, weights):

    # Calculate z = Xw
    z = np.dot(X, weights)

    # Calculate probability using sigmoid
    probabilities = sigmoid(z)

    # If probability >= 0.5, predict 1
    # Otherwise predict 0
    predictions = (probabilities >= 0.5).astype(int)

    return predictions



# Accuracy Function


def accuracy(y_actual, y_predicted):

    # Count correct predictions
    correct = np.sum(y_actual == y_predicted)

    # Calculate accuracy
    return (correct / len(y_actual)) * 100



# Test Gradient Descent


gd_predictions = predict(
    Xtest,
    gd_weights
)

gd_accuracy = accuracy(
    ytest,
    gd_predictions
)


print("\nGradient Descent Results")

print("Accuracy =", gd_accuracy, "%")



# Newton's Method


def newtons_method(X, y, iterations):

    # Number of features including intercept
    n = X.shape[1]

    # Initially set all weights to zero
    weights = np.zeros(n)

    # Store cost values
    cost_history = []

    # Repeat Newton's method
    for i in range(iterations):

        # Calculate z = Xw
        z = np.dot(X, weights)

        # Calculate predicted probabilities
        predictions = sigmoid(z)

        # Calculate Gradient
        gradient = (1 / len(y)) * np.dot(
            X.T,
            predictions - y
        )

        # Calculate p(1-p)
        p = predictions * (1 - predictions)

        # Create diagonal matrix
        R = np.diag(p)

        # Calculate Hessian
        hessian = (1 / len(y)) * np.dot(
            X.T,
            np.dot(R, X)
        )

        # Find inverse of Hessian   
        hessian_inverse = np.linalg.inv(hessian)

        # Calculate H^(-1) * gradient
        update = np.dot(
            hessian_inverse,
            gradient
        )

        # Newton's Method update
        weights = weights - update

        # Calculate cost
        cost = cost_function(X, y, weights)

        # Store cost
        cost_history.append(cost)
        
        # Check convergence
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < 0.000001:
            break

    return weights, cost_history



newton_weights, newton_cost = newtons_method(
    Xtrain,
    ytrain,
    iterations=1000
)



# Test Newton's Method


newton_predictions = predict(
    Xtest,
    newton_weights
)

newton_accuracy = accuracy(
    ytest,
    newton_predictions
)


print("\nNewton's Method Results")

print("Accuracy =", newton_accuracy, "%")




print("\nFINAL COMPARISON")

print("Gradient Descent Accuracy =", gd_accuracy, "%")
print("Newton's Method Accuracy  =", newton_accuracy, "%")

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(gd_cost) + 1),
    gd_cost
)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost vs Iterations")

plt.show()




plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(newton_cost) + 1),
    newton_cost
)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Newton's Method: Cost vs Iterations")

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(gd_cost) + 1),
    gd_cost,
    label="Gradient Descent"
)

plt.plot(
    range(1, len(newton_cost) + 1),
    newton_cost,
    label="Newton's Method"
)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent vs Newton's Method")

plt.legend()

plt.show()