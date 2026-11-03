import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

with open("rt-polarity.pos", "r", encoding="latin-1") as f:
    positive_reviews = [line.strip() for line in f if line.strip()]

with open("rt-polarity.neg", "r", encoding="latin-1") as f:
    negative_reviews = [line.strip() for line in f if line.strip()]

texts = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

print("Total reviews:", len(texts))
print("Positive reviews:", len(positive_reviews))
print("Negative reviews:", len(negative_reviews))


X_temp, X_test, y_temp, y_test = train_test_split(
    texts, labels, test_size=1062, random_state=42, stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=1100, random_state=42, stratify=y_temp
)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

vocab = {"<pad>": 0, "<unk>": 1}

for text in X_train:
    for word in tokenize(text):
        if word not in vocab:
            vocab[word] = len(vocab)

PAD_IDX = vocab["<pad>"]
UNK_IDX = vocab["<unk>"]
MAX_LEN = 50

print("Vocabulary size:", len(vocab))


def encode_text(text):
    sequence = [vocab.get(word, UNK_IDX) for word in tokenize(text)]
    sequence = sequence[:MAX_LEN]

    while len(sequence) < MAX_LEN:
        sequence.append(PAD_IDX)

    return sequence

X_train = [encode_text(text) for text in X_train]
X_val = [encode_text(text) for text in X_val]
X_test = [encode_text(text) for text in X_test]

X_train = torch.tensor(X_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

BATCH_SIZE = 64

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

embedding_dim = 128
hidden_dim = 128

rnn_embedding = nn.Embedding(
    len(vocab), embedding_dim, padding_idx=PAD_IDX
).to(device)

rnn_layer = nn.RNN(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    batch_first=True
).to(device)

rnn_fc = nn.Linear(hidden_dim, 1).to(device)

rnn_parameters = (
    list(rnn_embedding.parameters())
    + list(rnn_layer.parameters())
    + list(rnn_fc.parameters())
)

rnn_optimizer = torch.optim.Adam(rnn_parameters, lr=0.001)
criterion = nn.BCEWithLogitsLoss()

rnn_train_loss = []
rnn_val_loss = []
rnn_val_acc = []

best_rnn_loss = float("inf")
patience = 3
counter = 0
epochs = 15

for epoch in range(epochs):

    rnn_embedding.train()
    rnn_layer.train()
    rnn_fc.train()

    total_loss = 0

    for inputs, labels_batch in train_loader:

        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device)

        rnn_optimizer.zero_grad()

        embedded = rnn_embedding(inputs)
        output, hidden = rnn_layer(embedded)

        logits = rnn_fc(hidden[-1]).squeeze(1)

        loss = criterion(logits, labels_batch)
        loss.backward()
        rnn_optimizer.step()

        total_loss += loss.item() * labels_batch.size(0)

    train_loss = total_loss / len(train_loader.dataset)
    rnn_train_loss.append(train_loss)

    # Validation
    rnn_embedding.eval()
    rnn_layer.eval()
    rnn_fc.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels_batch in val_loader:

            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            embedded = rnn_embedding(inputs)
            output, hidden = rnn_layer(embedded)
            logits = rnn_fc(hidden[-1]).squeeze(1)

            loss = criterion(logits, labels_batch)
            total_loss += loss.item() * labels_batch.size(0)

            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

    val_loss = total_loss / total
    val_accuracy = correct / total

    rnn_val_loss.append(val_loss)
    rnn_val_acc.append(val_accuracy)

    print(f"RNN Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}")

    if val_loss < best_rnn_loss:
        best_rnn_loss = val_loss
        counter = 0

        torch.save({
            "embedding": rnn_embedding.state_dict(),
            "rnn": rnn_layer.state_dict(),
            "fc": rnn_fc.state_dict()
        }, "best_rnn.pth")

    else:
        counter += 1

    if counter >= patience:
        print("RNN early stopping")
        break


# Restore best RNN weights
checkpoint = torch.load("best_rnn.pth", map_location=device, weights_only=True)
rnn_embedding.load_state_dict(checkpoint["embedding"])
rnn_layer.load_state_dict(checkpoint["rnn"])
rnn_fc.load_state_dict(checkpoint["fc"])

#LSTM
lstm_embedding = nn.Embedding(
    len(vocab), embedding_dim, padding_idx=PAD_IDX
).to(device)

lstm_layer = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    batch_first=True
).to(device)

lstm_fc = nn.Linear(hidden_dim, 1).to(device)

lstm_parameters = (list(lstm_embedding.parameters())+list(lstm_layer.parameters())+list(lstm_fc.parameters()))

lstm_optimizer = torch.optim.Adam(lstm_parameters, lr=0.001)

lstm_train_loss = []
lstm_val_loss = []
lstm_val_acc = []

best_lstm_loss = float("inf")
counter = 0

for epoch in range(epochs):

    lstm_embedding.train()
    lstm_layer.train()
    lstm_fc.train()

    total_loss = 0

    for inputs, labels_batch in train_loader:

        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device)

        lstm_optimizer.zero_grad()

        embedded = lstm_embedding(inputs)
        output, (hidden, cell) = lstm_layer(embedded)

        logits = lstm_fc(hidden[-1]).squeeze(1)

        loss = criterion(logits, labels_batch)
        loss.backward()
        lstm_optimizer.step()

        total_loss += loss.item() * labels_batch.size(0)

    train_loss=total_loss/len(train_loader.dataset)
    lstm_train_loss.append(train_loss)

    # Validation
    lstm_embedding.eval()
    lstm_layer.eval()
    lstm_fc.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels_batch in val_loader:

            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            embedded = lstm_embedding(inputs)
            output, (hidden, cell) = lstm_layer(embedded)

            logits = lstm_fc(hidden[-1]).squeeze(1)

            loss = criterion(logits, labels_batch)
            total_loss += loss.item() * labels_batch.size(0)

            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

    val_loss = total_loss / total
    val_accuracy = correct / total

    lstm_val_loss.append(val_loss)
    lstm_val_acc.append(val_accuracy)

    print(f"LSTM Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}")

    if val_loss < best_lstm_loss:
        best_lstm_loss = val_loss
        counter = 0

        torch.save({
            "embedding": lstm_embedding.state_dict(),
            "lstm": lstm_layer.state_dict(),
            "fc": lstm_fc.state_dict()
        }, "best_lstm.pth")

    else:
        counter += 1

    if counter >= patience:
        print("LSTM early stopping")
        break


# Restore best LSTM weights
checkpoint = torch.load("best_lstm.pth", map_location=device, weights_only=True)
lstm_embedding.load_state_dict(checkpoint["embedding"])
lstm_layer.load_state_dict(checkpoint["lstm"])
lstm_fc.load_state_dict(checkpoint["fc"])

plt.figure(figsize=(10, 5))

plt.plot(rnn_train_loss, label="RNN Training Loss")
plt.plot(rnn_val_loss, label="RNN Validation Loss")
plt.plot(lstm_train_loss, label="LSTM Training Loss")
plt.plot(lstm_val_loss, label="LSTM Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN vs LSTM: Loss Curves")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(rnn_val_acc, marker="o", label="RNN Validation Accuracy")
plt.plot(lstm_val_acc, marker="s", label="LSTM Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("RNN vs LSTM: Validation Accuracy")
plt.legend()
plt.grid()
plt.show()


def get_predictions(model_type):

    predictions = []
    actual = []

    if model_type == "RNN":
        rnn_embedding.eval()
        rnn_layer.eval()
        rnn_fc.eval()
    else:
        lstm_embedding.eval()
        lstm_layer.eval()
        lstm_fc.eval()

    with torch.no_grad():

        for inputs, labels_batch in test_loader:

            inputs = inputs.to(device)

            if model_type == "RNN":
                embedded = rnn_embedding(inputs)
                output, hidden = rnn_layer(embedded)
                logits = rnn_fc(hidden[-1]).squeeze(1)
            else:
                embedded = lstm_embedding(inputs)
                output, (hidden, cell) = lstm_layer(embedded)
                logits = lstm_fc(hidden[-1]).squeeze(1)

            pred = (torch.sigmoid(logits) >= 0.5).long()

            predictions.extend(pred.cpu().numpy())
            actual.extend(labels_batch.long().numpy())

    return actual, predictions


rnn_actual, rnn_predictions = get_predictions("RNN")
lstm_actual, lstm_predictions = get_predictions("LSTM")


def calculate_metrics(actual, predictions, name):

    print("\n", name, "Test Results")
    print("-" * 35)

    print("Accuracy :", round(accuracy_score(actual, predictions), 4))
    print("Precision:", round(precision_score(actual, predictions, zero_division=0), 4))
    print("Recall   :", round(recall_score(actual, predictions, zero_division=0), 4))
    print("F1-score :", round(f1_score(actual, predictions, zero_division=0), 4))


calculate_metrics(rnn_actual, rnn_predictions, "Simple RNN")
calculate_metrics(lstm_actual, lstm_predictions, "LSTM")


results = pd.DataFrame({
    "Model": ["Simple RNN", "LSTM"],
    "Accuracy": [
        accuracy_score(rnn_actual, rnn_predictions),
        accuracy_score(lstm_actual, lstm_predictions)
    ],
    "Precision": [
        precision_score(rnn_actual, rnn_predictions, zero_division=0),
        precision_score(lstm_actual, lstm_predictions, zero_division=0)
    ],
    "Recall": [
        recall_score(rnn_actual, rnn_predictions, zero_division=0),
        recall_score(lstm_actual, lstm_predictions, zero_division=0)
    ],
    "F1-score": [
        f1_score(rnn_actual, rnn_predictions, zero_division=0),
        f1_score(lstm_actual, lstm_predictions, zero_division=0)
    ]
})

print("\nFinal Model Comparison:")
print(results.round(4).to_string(index=False))