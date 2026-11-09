import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("50_Startups.csv")

print(df.shape)
print(df.columns)

print("\nMissing values:")
print(df.isna().sum())

df = pd.get_dummies(df, columns=["State"], drop_first=True)

print("\nAfter One-Hot Encoding:")
print(df.head())

print("\nColumns:")
print(df.columns)

features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State_Florida",
    "State_New York",
]

target = "Profit"

X = df[features].copy()
y = df[target].copy()


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
mean = Xtrain.mean()
std = Xtrain.std()

Xtrain = (Xtrain - mean) / std
Xtest = (Xtest - mean) / std

Xtrain = Xtrain.to_numpy(dtype=float)
Xtest = Xtest.to_numpy(dtype=float)
ytrain = ytrain.to_numpy(dtype=float)
ytest = ytest.to_numpy(dtype=float)

Xtrain = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
Xtest = np.c_[np.ones(Xtest.shape[0]), Xtest]

B = np.zeros(Xtrain.shape[1])

iterations = 1000000
alpha = 0.0001
tolerance = 10e-6

n = Xtrain.shape[0]
losshistory = []

for i in range(iterations):
    ypred = Xtrain @ B
    error = ytrain - ypred

    loss = np.mean(error**2)
    losshistory.append(loss)

    gradient = (-2 / n) * (Xtrain.T @ error)

    B = B - (alpha * gradient)

    gradient_norm = np.sqrt(np.sum(gradient**2))

    if gradient_norm < tolerance:
        print("Training converged.")
        print("Iteration:", i)
        break

    if i % 100 == 0:
        print("Iteration:", i)
        print("Loss:", loss)
        print("B:", B)


ytrainpred = Xtrain @ B
trainerror = ytrain - ytrainpred

trainmse = np.mean(trainerror**2)
trainrmse = np.sqrt(trainmse)

ytestpred = Xtest @ B
testerror = ytest - ytestpred

test_mse = np.mean(testerror**2)


test_rmse = np.sqrt(test_mse)

plt.figure(figsize=(8, 6))

plt.scatter(ytrain, ytrainpred, alpha=0.7)


minimum = min(ytrain.min(), ytrainpred.min())

maximum = max(ytrain.max(), ytrainpred.max())


# Perfect prediction line

plt.plot([minimum, maximum], [minimum, maximum])


plt.xlabel("Actual Profit")

plt.ylabel("Predicted Profit")

plt.title("Training Data: Actual vs Predicted")

plt.grid(True)

plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(ytest, ytestpred, alpha=0.7)


minimum = min(ytest.min(), ytestpred.min())

maximum = max(ytest.max(), ytestpred.max())


# Perfect prediction line

plt.plot([minimum, maximum], [minimum, maximum])


plt.xlabel("Actual Profit")

plt.ylabel("Predicted Profit")

plt.title("Testing Data: Actual vs Predicted")

plt.grid(True)

plt.show()
