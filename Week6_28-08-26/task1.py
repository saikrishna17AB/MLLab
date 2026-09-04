import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\SANTOSH KUMAR GOJE\OneDrive\Desktop\SM\MLLab\Week6_28-08-26\inputDataset.txt"

df = pd.read_csv(file_path, sep=r"\s+", header=None)
df.columns = ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol", "age", "chd"]

X = df.drop("chd", axis=1).values
y = df["chd"].values

np.random.seed(42)
indices = np.random.permutation(len(X))
test_count = int(0.30 * len(X))
test_indices = indices[:test_count]
train_indices = indices[test_count:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

classes = np.unique(y_train)
means = {}
priors = {}

for c in classes:
    data = X_train[y_train == c]
    means[c] = np.mean(data, axis=0)
    priors[c] = len(data) / len(X_train)

n_features = X_train.shape[1]
covariance = np.zeros((n_features, n_features))

for c in classes:
    data = X_train[y_train == c]
    centered = data - means[c]
    covariance += centered.T @ centered

covariance = covariance / (len(X_train) - len(classes))
inverse_covariance = np.linalg.pinv(covariance)

pred_lda = []

for x in X_test:
    scores = []

    for c in classes:
        score = (
            x @ inverse_covariance @ means[c]
            - 0.5 * means[c] @ inverse_covariance @ means[c]
            + np.log(priors[c])
        )
        scores.append(score)

    pred_lda.append(classes[np.argmax(scores)])

pred_lda = np.array(pred_lda)

means_qda = {}
covariances_qda = {}
inverse_covariances_qda = {}
log_determinants = {}
priors_qda = {}

for c in classes:
    data = X_train[y_train == c]
    mean = np.mean(data, axis=0)
    centered = data - mean
    covariance = (centered.T @ centered) / (len(data) - 1)

    means_qda[c] = mean
    covariances_qda[c] = covariance
    inverse_covariances_qda[c] = np.linalg.pinv(covariance)
    log_determinants[c] = np.linalg.slogdet(covariance)[1]
    priors_qda[c] = len(data) / len(X_train)

pred_qda = []

for x in X_test:
    scores = []

    for c in classes:
        difference = x - means_qda[c]

        score = (
            -0.5 * log_determinants[c]
            -0.5 * difference @ inverse_covariances_qda[c] @ difference
            + np.log(priors_qda[c])
        )

        scores.append(score)

    pred_qda.append(classes[np.argmax(scores)])

pred_qda = np.array(pred_qda)

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

def metrics(y_true, y_pred):
    matrix, labels = confusion_matrix(y_true, y_pred)

    accuracy = np.trace(matrix) / np.sum(matrix)
    precision_values = []
    recall_values = []

    for i in range(len(labels)):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        precision_values.append(precision)
        recall_values.append(recall)

    precision = np.mean(precision_values)
    recall = np.mean(recall_values)

    return accuracy, precision, recall

def report(y_true, y_pred):
    matrix, labels = confusion_matrix(y_true, y_pred)

    print("Class\tPrecision\tRecall\tF1")

    for i, c in enumerate(labels):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        print(f"{c}\t{precision:.4f}\t\t{recall:.4f}\t{f1:.4f}")

    print("Accuracy:", f"{np.mean(y_true == y_pred):.4f}")

print("LDA Classification Report")
report(y_test, pred_lda)

print("\nQDA Classification Report")
report(y_test, pred_qda)

a1, p1, r1 = metrics(y_test, pred_lda)
a2, p2, r2 = metrics(y_test, pred_qda)

result = pd.DataFrame({
    "Model": ["LDA", "QDA"],
    "Accuracy": [a1, a2],
    "Precision": [p1, p2],
    "Recall": [r1, r2]
})

print("\nPerformance Comparison")
print(result.to_string(index=False))

cm1, labels1 = confusion_matrix(y_test, pred_lda)
cm2, labels2 = confusion_matrix(y_test, pred_qda)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(cm1)
ax[0].set_title("LDA")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")
ax[0].set_xticks(range(len(labels1)))
ax[0].set_yticks(range(len(labels1)))

for i in range(len(labels1)):
    for j in range(len(labels1)):
        ax[0].text(j, i, cm1[i, j], ha="center", va="center")

ax[1].imshow(cm2)
ax[1].set_title("QDA")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")
ax[1].set_xticks(range(len(labels2)))
ax[1].set_yticks(range(len(labels2)))

for i in range(len(labels2)):
    for j in range(len(labels2)):
        ax[1].text(j, i, cm2[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(r"C:\Users\SANTOSH KUMAR GOJE\OneDrive\Desktop\SM\MLLab\Week6_28-08-26\task1_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nLDA False Negatives:", cm1[1, 0])
print("QDA False Negatives:", cm2[1, 0])