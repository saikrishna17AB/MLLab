import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Set random seeds to make results reproducible
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)


# Check whether GPU is available
gpus = tf.config.list_physical_devices("GPU")

device = "GPU" if len(gpus) > 0 else "CPU"

print("Device:", device)

# Read positive reviews from the positive dataset
with open("rt-polarity.pos", "r", encoding="latin-1") as f:
    positive_reviews = [line.strip() for line in f if line.strip()]


# Read negative reviews from the negative dataset
with open("rt-polarity.neg", "r", encoding="latin-1") as f:
    negative_reviews = [line.strip() for line in f if line.strip()]


# Combine positive and negative reviews
texts = positive_reviews + negative_reviews


# Assign labels
# 1 represents positive sentiment
# 0 represents negative sentiment
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)


# Display dataset information
print("Total reviews:", len(texts))
print("Positive reviews:", len(positive_reviews))
print("Negative reviews:", len(negative_reviews))

# First split: Separate test data
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, labels, test_size=1062, random_state=42, stratify=labels
)


# Second split: Separate training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=1100, random_state=42, stratify=y_temp
)


# Display dataset sizes
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# Function to convert a sentence into individual tokens
def tokenize(text):

    return re.findall(r"\w+|[^\w\s]", text.lower())


# Create vocabulary dictionary
# PAD represents padding
# UNK represents unknown words
vocab = {"<pad>": 0, "<unk>": 1}


# Build vocabulary using only training data
for text in X_train:
    for word in tokenize(text):
        if word not in vocab:
            vocab[word] = len(vocab)


# Get padding and unknown token indices
PAD_IDX = vocab["<pad>"]
UNK_IDX = vocab["<unk>"]


# Maximum number of tokens in each review
MAX_LEN = 50


print("Vocabulary size:", len(vocab))


# Function to convert a review into numerical values
def encode_text(text):

    # Convert words into their corresponding vocabulary indices
    sequence = [vocab.get(word, UNK_IDX) for word in tokenize(text)]

    # Truncate sequences longer than MAX_LEN
    sequence = sequence[:MAX_LEN]

    # Add padding if the sequence is shorter than MAX_LEN
    while len(sequence) < MAX_LEN:
        sequence.append(PAD_IDX)

    return sequence


# Encode training, validation and testing reviews
X_train = [encode_text(text) for text in X_train]

X_val = [encode_text(text) for text in X_val]

X_test = [encode_text(text) for text in X_test]


# Convert input sequences into NumPy arrays
X_train = np.array(X_train, dtype=np.int32)

X_val = np.array(X_val, dtype=np.int32)

X_test = np.array(X_test, dtype=np.int32)


# Convert labels into NumPy arrays
y_train = np.array(y_train, dtype=np.float32)

y_val = np.array(y_val, dtype=np.float32)

y_test = np.array(y_test, dtype=np.float32)


# Embedding dimension
embedding_dim = 128

# Number of hidden units in RNN and LSTM
hidden_dim = 128

# Number of samples processed in one batch
BATCH_SIZE = 64

# Learning rate of Adam optimizer
learning_rate = 0.001

# Maximum number of training epochs
epochs = 15

# Number of epochs to wait before early stopping
patience = 3


# Function to create the Simple RNN model
def build_rnn_model():

    # Create a sequential model
    model = tf.keras.Sequential(
        [
            # Input: Each review contains MAX_LEN tokens
            tf.keras.layers.Input(shape=(MAX_LEN,), dtype="int32"),
            # Embedding layer
            # Converts each word index into a 128-dimensional vector
            # mask_zero=True tells the RNN to ignore padding tokens
            tf.keras.layers.Embedding(
                input_dim=len(vocab), output_dim=embedding_dim, mask_zero=True
            ),
            # Simple RNN layer
            # Contains 128 hidden units
            tf.keras.layers.SimpleRNN(units=hidden_dim, activation="tanh"),
            # Output layer
            # One neuron for binary sentiment classification
            # Sigmoid produces probability between 0 and 1
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the model
    model.compile(
        # Adam optimizer
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # Binary cross entropy for binary classification
        loss="binary_crossentropy",
        # Accuracy metric
        metrics=["accuracy"],
    )

    return model


# Create RNN model
rnn_model = build_rnn_model()


# Display model architecture
rnn_model.summary()


# Save the best model based on validation loss
rnn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_rnn.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)


# Early stopping callback
rnn_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
)


# Train the RNN model
rnn_history = rnn_model.fit(
    # Training inputs and labels
    X_train,
    y_train,
    # Validation data
    validation_data=(X_val, y_val),
    # Batch size
    batch_size=BATCH_SIZE,
    # Maximum epochs
    epochs=epochs,
    # Shuffle training samples
    shuffle=True,
    # Apply callbacks
    callbacks=[rnn_checkpoint, rnn_early_stopping],
    verbose=1,
)


# Load the best RNN model
rnn_model = tf.keras.models.load_model("best_rnn.keras")


# Function to create the LSTM model
def build_lstm_model():

    # Create sequential model
    model = tf.keras.Sequential(
        [
            # Input layer
            tf.keras.layers.Input(shape=(MAX_LEN,), dtype="int32"),
            # Embedding layer
            tf.keras.layers.Embedding(
                input_dim=len(vocab), output_dim=embedding_dim, mask_zero=True
            ),
            # LSTM layer with 128 hidden units
            tf.keras.layers.LSTM(units=hidden_dim, activation="tanh"),
            # Output layer for binary classification
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the LSTM model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Create LSTM model
lstm_model = build_lstm_model()


# Display model architecture
lstm_model.summary()


# Save the best LSTM model
lstm_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_lstm.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)


# Early stopping callback
lstm_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
)


# Train the LSTM model
lstm_history = lstm_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=epochs,
    shuffle=True,
    callbacks=[lstm_checkpoint, lstm_early_stopping],
    verbose=1,
)


# Load the best LSTM model
lstm_model = tf.keras.models.load_model("best_lstm.keras")


# Extract RNN training and validation loss
rnn_train_loss = rnn_history.history["loss"]

rnn_val_loss = rnn_history.history["val_loss"]


# Extract LSTM training and validation loss
lstm_train_loss = lstm_history.history["loss"]

lstm_val_loss = lstm_history.history["val_loss"]


# Plot loss curves
plt.figure(figsize=(10, 5))


# RNN loss
plt.plot(rnn_train_loss, label="RNN Training Loss")

plt.plot(rnn_val_loss, label="RNN Validation Loss")


# LSTM loss
plt.plot(lstm_train_loss, label="LSTM Training Loss")

plt.plot(lstm_val_loss, label="LSTM Validation Loss")


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("RNN vs LSTM: Loss Curves")

plt.legend()

plt.grid()

plt.show()

# Extract validation accuracy
rnn_val_acc = rnn_history.history["val_accuracy"]

lstm_val_acc = lstm_history.history["val_accuracy"]


# Plot validation accuracy
plt.figure(figsize=(8, 5))


plt.plot(rnn_val_acc, marker="o", label="RNN Validation Accuracy")


plt.plot(lstm_val_acc, marker="s", label="LSTM Validation Accuracy")


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("RNN vs LSTM: Validation Accuracy")

plt.legend()

plt.grid()

plt.show()


# Function to generate predictions
def get_predictions(model):

    # Get predicted probabilities
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

    # Convert probabilities into binary predictions
    # Probability >= 0.5 represents positive sentiment
    predictions = (probabilities >= 0.5).astype(int)

    # Return actual labels and predicted labels
    return y_test.astype(int), predictions.flatten()


# Generate RNN predictions
rnn_actual, rnn_predictions = get_predictions(rnn_model)


# Generate LSTM predictions
lstm_actual, lstm_predictions = get_predictions(lstm_model)


# Function to calculate evaluation metrics
def calculate_metrics(actual, predictions, name):

    print("\n", name, "Test Results")

    print("-" * 35)

    # Calculate accuracy
    print("Accuracy :", round(accuracy_score(actual, predictions), 4))

    # Calculate precision
    print("Precision:", round(precision_score(actual, predictions, zero_division=0), 4))

    # Calculate recall
    print("Recall   :", round(recall_score(actual, predictions, zero_division=0), 4))

    # Calculate F1-score
    print("F1-score :", round(f1_score(actual, predictions, zero_division=0), 4))


# Evaluate RNN
calculate_metrics(rnn_actual, rnn_predictions, "Simple RNN")


# Evaluate LSTM
calculate_metrics(lstm_actual, lstm_predictions, "LSTM")

# Create comparison dataframe
results = pd.DataFrame(
    {
        "Model": ["Simple RNN", "LSTM"],
        "Accuracy": [
            accuracy_score(rnn_actual, rnn_predictions),
            accuracy_score(lstm_actual, lstm_predictions),
        ],
        "Precision": [
            precision_score(rnn_actual, rnn_predictions, zero_division=0),
            precision_score(lstm_actual, lstm_predictions, zero_division=0),
        ],
        "Recall": [
            recall_score(rnn_actual, rnn_predictions, zero_division=0),
            recall_score(lstm_actual, lstm_predictions, zero_division=0),
        ],
        "F1-score": [
            f1_score(rnn_actual, rnn_predictions, zero_division=0),
            f1_score(lstm_actual, lstm_predictions, zero_division=0),
        ],
    }
)


# Display final comparison
print("\nFinal Model Comparison:")

print(results.round(4).to_string(index=False))
