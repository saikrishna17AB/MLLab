import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart-disease.csv")

df = df.fillna(0)

X = df.drop("target", axis=1).values
y = df["target"]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.fit_transform(Xtest)

ytrain = ytrain.values
ytest = ytest.values

onestrain = np.ones((Xtrain.shape[0], 1))
Xtrain = np.hstack((onestrain, Xtrain))

onestest = np.ones((Xtest.shape[0], 1))
Xtest = np.hstack((onestest, Xtest))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(X, y, W):

    z = np.dot(X, W)
    predictions = sigmoid(z)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)

    return (-1 / len(X)) * np.sum(
        y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
    )


a = 0.005


def gradientdescent(X, y, iterations, tolerance):

    n = X.shape[1]
    W = np.zeros(n)
    losshistory = []
    for i in range(iterations):
        z = np.dot(X, W)
        p = sigmoid(z)
        dW = (1 / len(y)) * np.dot(X.T, p - y)
        if np.linalg.norm(dW) < tolerance:
            break
        currentloss = loss(X, y, W)
        losshistory.append(currentloss)
        W = W - (dW * a)

    return W, losshistory


W, losshistory = gradientdescent(Xtrain, ytrain, 10000, np.exp(-20))


def predict(X, W):
    z = np.dot(X, W)
    probabilities = sigmoid(z)
    predictions = (probabilities >= 0.5).astype(int)

    return predictions


def accuracy(predictions, y):

    correct = np.sum(predictions == y)
    return correct / len(predictions)


predictions = predict(Xtest, W)
accuracy = accuracy_score(ytest, predictions)

print("Accuracy: ", accuracy)

cm = confusion_matrix(ytest, predictions)
print(cm)

report = classification_report(ytest, predictions)
print(report)


# Usine netwons method


def newgd(X, y, iterations):
    n = X.shape[1]
    m = X.shape[0]
    W = np.zeros(n)
    losshistory = []

    for i in range(iterations):
        z = np.dot(X, W)
        prediction = sigmoid(z)

        gradient = (1 / m) * np.dot(X.T, prediction - y)
        p = prediction * (1 - prediction)
        R = np.diag(p)

        H = (1 / m) * np.dot(X.T, np.dot(R, X))
        Hinverse = np.linalg.inv(H)

        update = np.dot(Hinverse, gradient)
        if np.linalg.norm(update) < np.exp(-20):
            break
        currentloss = loss(X, y, W)
        losshistory.append(currentloss)
        W = W - update

    return W, losshistory


W1, lh = newgd(Xtrain, ytrain, 1000)
npred = predict(Xtest, W1)
accuracy = accuracy_score(ytest, npred)
cmn = confusion_matrix(ytest, npred)
crn = classification_report(ytest, npred)

print(accuracy)
print(cmn)
print(crn)


plt.figure(figsize=(10, 8))

plt.plot(losshistory, label="Normal GD")
plt.plot(lh, label="Newton's GD")
plt.show()