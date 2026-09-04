import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\SANTOSH KUMAR GOJE\OneDrive\Desktop\SM\MLLab\Week6_28-08-26\Wine.csv"

df = pd.read_csv(file_path)

X = df.drop("Customer_Segment", axis=1).values
y = df["Customer_Segment"].values

np.random.seed(42)
indices = np.random.permutation(len(X))
test_count = int(0.20 * len(X))

test_indices = indices[:test_count]
train_indices = indices[test_count:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

classes = np.unique(y_train)
overall_mean = np.mean(X_train_scaled, axis=0)
n_features = X_train_scaled.shape[1]

sw = np.zeros((n_features, n_features))
sb = np.zeros((n_features, n_features))

for c in classes:
    data = X_train_scaled[y_train == c]
    class_mean = np.mean(data, axis=0)

    centered = data - class_mean
    sw += centered.T @ centered

    difference = (class_mean - overall_mean).reshape(-1, 1)
    sb += len(data) * (difference @ difference.T)

matrix = np.linalg.pinv(sw) @ sb

eigenvalues, eigenvectors = np.linalg.eig(matrix)

eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

components = eigenvectors[:, :2]

X_train_lda = X_train_scaled @ components
X_test_lda = X_test_scaled @ components

print("Original dimensions:", X_train_scaled.shape[1])
print("Reduced dimensions:", X_train_lda.shape[1])
print("Training shape:", X_train_lda.shape)
print("Testing shape:", X_test_lda.shape)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

models = {}

for c in classes:
    target = (y_train == c).astype(int)

    weights = np.zeros(X_train_lda.shape[1])
    bias = 0

    learning_rate = 0.01
    iterations = 5000

    for _ in range(iterations):
        z = X_train_lda @ weights + bias
        prediction = sigmoid(z)

        dw = (X_train_lda.T @ (prediction - target)) / len(X_train_lda)
        db = np.mean(prediction - target)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    models[c] = (weights, bias)

probabilities = []

for c in classes:
    weights, bias = models[c]

    z = X_test_lda @ weights + bias
    probability = sigmoid(z)

    probabilities.append(probability)

probabilities = np.array(probabilities).T

pred = classes[np.argmax(probabilities, axis=1)]

def confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix[i, j] = np.sum(
                (y_true == labels[i]) &
                (y_pred == labels[j])
            )

    return matrix, labels

cm, labels = confusion_matrix(y_test, pred)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print("Class\tPrecision\tRecall\tF1")

for i, c in enumerate(labels):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp

    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    print(f"{c}\t{precision:.4f}\t\t{recall:.4f}\t{f1:.4f}")

accuracy = np.mean(y_test == pred)

print("\nAccuracy:", f"{accuracy:.4f}")

plt.figure(figsize=(9, 7))

for c in np.unique(y_test):
    points = X_test_lda[y_test == c]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        label=f"Class {c}"
    )

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("Wine Dataset - LDA Projection")
plt.legend()
plt.grid(True)
plt.savefig(r"C:\Users\SANTOSH KUMAR GOJE\OneDrive\Desktop\SM\MLLab\Week6_28-08-26\task2_lda_plot.png", dpi=300, bbox_inches="tight")
plt.show()