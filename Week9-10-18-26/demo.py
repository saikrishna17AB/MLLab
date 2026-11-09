from collections import Counter

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import *

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

positivefile = "rt-polarity.pos"
negativefile = "rt-polarity.neg"

with open(positivefile, "r", encoding="latin-1") as file:
    positivereviews = file.readlines()

with open(negativefile, "r", encoding="latin-1") as file:
    negativereviews = file.readlines()

positivelabels = [1] * len(positivereviews)
negativelabels = [0] * len(negativereviews)

reviews = positivereviews + negativereviews
labels = positivelabels + negativelabels

reviews = [review.strip() for review in reviews]

# Split the dataset

Xtrain, Xtemp, ytrain, ytemp = train_test_split(
    reviews, labels, train_size=8500, stratify=labels, random_state=42
)
Xval, Xtest, yval, ytest = train_test_split(
    Xtemp, ytemp, train_size=1100, stratify=ytemp, random_state=42
)


def tokenize(sentence):
    sentence = sentence.lower()
    tokens = sentence.split()

    return tokens  # ['the', 'movie', 'was', 'really', 'good']


# Step 1.5 – Build the vocabulary
# This is an important step.

# A neural network cannot directly understand words. We need to assign a unique integer ID to every word.

# For example:
# Word	ID
# <pad>	0
# <unk>	1
# the	2
# movie	3
# was	4
# good	5
# bad	6

# Two special tokens are required:
# - <pad>: Used when a sentence has fewer than 50 words.
# - <unk>: Used when a word is not present in the vocabulary.


wordcounts = Counter()

for review in Xtrain:
    tokens = tokenize(review)
    wordcounts.update(tokens)

print(wordcounts)

vocab = {"<pad>": 0, "<unk>": 1}
for word in wordcounts:
    if word not in vocab:
        vocab[word] = len(vocab)


# Reverse vocabulary: ID to word
idtoword = {idx: word for word, idx in vocab.items()}

# tep 1.6 – Convert sentences into integer sequences
# Now we use the vocabulary to convert each review into numerical form.

# For example:
# Review:
# "The movie is good"

# Integer sequence:
# [2, 3, 4, 5]

maxlen = 50


def encodereview(review):
    tokens = tokenize(review)
    sequence = [vocab.get(word, vocab["<unk>"]) for word in tokens]

    return sequence


def padsequence(sequence):
    if len(sequence) > maxlen:
        sequence = sequence[:maxlen]
    else:
        padding = maxlen - len(sequence)
        sequence = sequence + [vocab["<pad>"]] * padding
    return sequence


Xtrainencoded = [padsequence(encodereview(review)) for review in Xtrain]
Xvalencoded = [padsequence(encodereview(review)) for review in Xval]
Xtestencoded = [padsequence(encodereview(review)) for review in Xtest]

Xtrainlengths = [min(maxlen, encodereview(review)) for review in Xtrain]
Xvallengths = [min(maxlen, encodereview(review)) for review in Xval]
Xtestlengths = [min(maxlen, encodereview(review)) for review in Xtest]

# Convert training sequences into an integer tensorflow tensor

Xtrain = tf.constant(Xtrainencoded, dtype=tf.int32)
Xval = tf.constant(Xvalencoded, dtype=tf.int32)
Xtest = tf.constant(Xtestencoded, dtype=tf.int32)

# Convert labels into float tensors. Float tensors are required for binary cross entropy
ytrain = tf.constant(ytrain, dtype=tf.float32)
yval = tf.constant(yval, dtype=tf.float32)
ytest = tf.constant(ytest, dtype=tf.float32)

# Convert the actual sequence lengths into Tensorflow integer tensors

Xtrainlengths = tf.constant(Xtrainlengths, dtype=tf.int32)
Xvallengths = tf.constant(Xvallengths, dtype=tf.int32)
Xtestlengths = tf.constant(Xtestlengths, dtype=tf.int32)

# Model Parameters

vocabsize = len(vocab)  # Number of unique words

embeddingdim = 128  # Every word will be represented by a vector of 128 members

hiddensize = 128  # Number of neuorns in the hidden layer of RNN

outputsize = 1  # One output becz this is binary classification


def initweights(shape):  # Returns a trainable tensorflow variable
    return tf.Variable(tf.random.normal(shape, stddev=0.1))


# Initialize model weights

# Embedding matrix

# Each row represents one word in the vocabulary.
# Each row contains 128 embedding values.
Wemb = initweights([vocabsize, embeddingdim])  # Shape=(vocab size,embedding dim)


# B. RNN WEIGHTS

# Input-to-hidden weight matrix.
# Converts the current word embedding into hidden-layer information.

Wxh = initweights([embeddingdim, hiddensize])  # Shape= (embeddingdim,hiddensize)

# Hidden-to-hidden weight matrix.
# Transfers information from the previous hidden state.
#
# Shape = (hidden_size, hidden_size).
Whh = initweights([hiddensize, hiddensize])

# Bias added while calculating the new hidden state.
#
# Shape = (1, hidden_size).

bh = tf.Variable(tf.zeros([1, hiddensize]))

# Output layer

# Converts the final hidden state into a single output score.
#
# Shape = (hidden_size, output_size).

Why = initweights([hiddensize, outputsize])

# Bias of the output layer.
#
# Shape = (1, output_size).
bhy = tf.Variable(tf.zeros([1, outputsize]))

# Group all trainable weights

runweights = [Wemb, Wxh, Whh, bh, Why, bhy]

# Weight	Shape	Purpose
# W_emb	(vocab_size, 128)	Embedding lookup
# W_xh	(128, 128)	Input-to-hidden
# W_hh	(128, 128)	Previous-hidden-to-current-hidden
# b_h	(1, 128)	Hidden-layer bias
# W_hy	(128, 1)	Hidden-to-output
# b_y	(1, 1)	Output bias

# Forward pass

# This function takes a batch of encoded reviews
# and calculates their prediction probabilities.


def forward(sequences, lengths):

    # Replace every integer word ID with its embedding vector.
    #
    # Input shape:  (batch_size, maxlen)
    # Output shape: (batch_size, maxlen, embedding_dim)

    embeddedwords = tf.nn.embedding_lookup(Wemb, sequences)

    # initialize hidden state

    currentbatchsize = tf.shape(sequences)[0]  # Number of reviews in current batch size

    # Initialize the hidden state with zeros.
    #
    # Each review gets its own hidden state.
    #
    # Shape = (batch_size, hidden_size).

    h = tf.zeros([currentbatchsize, hiddensize])

    # Process the words sequentially

    # Loop through all 50 positions of each review.
    # Each position represents one time step.

    for step in range(maxlen):
        # Extract the embedding vector of the current word
        # from every review in the batch.
        #
        # Shape = (batch_size, embedding_dim).

        Xt = embeddedwords[:, step, :]

        hnew = tf.tanh(tf.matmul(Xt, Wxh) + tf.matmul(h, Whh) + bh)

        # ==========================================
        # F. FINAL CLASSIFICATION LAYER
        # ==========================================

        # After processing the complete review,
        # h contains its final hidden representation.
        #
        # Convert this representation into one raw prediction score.
        #
        # Shape = (batch_size, 1).

    logits = tf.matmul(h, Why) + bhy

    probabilities = tf.sigmoid(logits)

    return logits, probabilities


# Loss function and optimizer

lossfunction = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Adam optimizer updates the trainable weights using their gradients.
#
# learning_rate=0.001 controls the size of each weight update.

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Train the RNN

epochs = 15
batchsize = 32

trainlosses = []
vallosses = []
valaccuracies = []

bestvalloss = float("inf")

bestweights = None

print("Training simple RNN")

for epoch in range(epochs):
    totaltrainloss = 0

    indices = tf.random.shuffle(tf.range(tf.shape(Xtrain)[0]))

    for start in range(0, len(Xtrain), batchsize):
        batchindices = indices[start : start + batchsize]

        batchX = tf.gather(Xtrain, batchindices)
        batchlengths = tf.gather(Xtrainlengths, batchindices)

        batchy = tf.gather(ytrain, batchindices)

        batchy = tf.expand_dims(batchy, axis=1)

        # Forward pass and loss calculations

        # GradientTape records the operations

        with tf.GradientTape() as tape:
            logits, probabilities = forward(batchX, batchlengths)

            loss = lossfunction(batchy, logits)

        # Backward propagation

        gradients = tape.gradient(loss, runweights)
        optimizer.apply_gradients(zip(gradients, runweights))

        totaltrainloss += loss.numpy() * int(tf.shape(batchX)[0])

    avgtrainloss = totaltrainloss / len(Xtrain)

    # Initialize validation loss.
    total_val_loss = 0.0

    # Initialize number of correctly classified validation reviews.
    correct = 0

    # Process validation data in batches.
    for start in range(0, len(Xval), batch_size):
        # Select the current validation batch.
        batch_X = Xval[start : start + batch_size]

        # Get the actual lengths of these reviews.
        batch_lengths = Xvallengths[start : start + batch_size]

        # Get the actual labels.
        batch_y = yval[start : start + batch_size]

        # Change label shape from (batch_size,) to (batch_size, 1).
        batch_y = tf.expand_dims(batch_y, axis=1)

        # Perform forward propagation.
        #
        # We do not need GradientTape here because
        # validation data is not used to update model weights.
        logits, probabilities = rnn_forward(batch_X, batch_lengths)

        # Calculate validation loss.
        loss = loss_function(batch_y, logits)

        # Accumulate validation loss weighted by batch size.
        total_val_loss += loss.numpy() * int(tf.shape(batch_X)[0])

        # Convert probabilities into binary predictions.
        #
        # Probability >= 0.5 means positive (1).
        # Probability < 0.5 means negative (0).
        predictions = tf.cast(probabilities >= 0.5, tf.float32)

        # Compare predictions with actual labels.
        # Count the number of correct predictions.
        correct += tf.reduce_sum(tf.cast(predictions == batch_y, tf.int32)).numpy()

    # Calculate average validation loss.
    avg_val_loss = total_val_loss / len(Xval)

    # Calculate validation accuracy.
    val_accuracy = correct / len(Xval)

    # ==========================================
    # C. STORE EPOCH RESULTS
    # ==========================================

    # Store training loss for plotting.
    train_losses.append(avg_train_loss)

    # Store validation loss for plotting.
    val_losses.append(avg_val_loss)

    # Store validation accuracy for plotting.
    val_accuracies.append(val_accuracy)

    # ==========================================
    # D. SAVE THE BEST MODEL
    # ==========================================

    # Check whether the current validation loss
    # is lower than the best validation loss so far.
    if avg_val_loss < best_val_loss:
        # Update the best validation loss.
        best_val_loss = avg_val_loss

        # Save copies of all trainable weights.
        #
        # tf.identity creates tensor copies of the current values.
        # This prevents later weight updates from changing the saved values.
        best_weights = [tf.identity(weight) for weight in rnn_weights]

    # Display the results after each epoch.
    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Val Loss: {avg_val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # ==========================================
# 6. RESTORE BEST MODEL
# ==========================================

# Traverse the original trainable weights and their saved values.
for weight, best_weight in zip(rnn_weights, best_weights):
    # Replace the current weight value with the saved best value.
    weight.assign(best_weight)


# Display the lowest validation loss obtained during training.
print("\nBest Validation Loss:", best_val_loss)

# ==========================================
# 7. TESTING
# ==========================================

# Store actual test labels.
y_actual = []

# Store model-predicted labels.
y_predicted = []


# Process the test dataset in batches.
for start in range(0, len(Xtest), batch_size):
    # Select the current batch of test reviews.
    batch_X = Xtest[start : start + batch_size]

    # Retrieve their actual sequence lengths.
    batch_lengths = Xtestlengths[start : start + batch_size]

    # Retrieve the actual sentiment labels.
    batch_y = ytest[start : start + batch_size]

    # Perform the forward pass using the trained model.
    logits, probabilities = rnn_forward(batch_X, batch_lengths)

    # Convert probabilities into binary predictions.
    #
    # >= 0.5 gives class 1 (positive).
    # < 0.5 gives class 0 (negative).
    predictions = tf.cast(probabilities >= 0.5, tf.int32)

    # Append actual labels to the list.
    y_actual.extend(batch_y.numpy().astype(int).tolist())

    # Remove the second dimension from predictions
    # and append them to the predicted-label list.
    y_predicted.extend(tf.squeeze(predictions, axis=1).numpy().tolist())


# ==========================================
# CALCULATE EVALUATION METRICS
# ==========================================

# Percentage of all predictions that are correct.
accuracy = accuracy_score(y_actual, y_predicted)


# Of all reviews predicted as positive,
# how many were actually positive?
precision = precision_score(y_actual, y_predicted, zero_division=0)


# Of all actual positive reviews,
# how many did the model correctly identify?
recall = recall_score(y_actual, y_predicted, zero_division=0)


# Harmonic mean of precision and recall.
f1 = f1_score(y_actual, y_predicted, zero_division=0)


# Display final test results.
print("\n========== RNN TEST RESULTS ==========")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)


# ==========================================
# 8. PLOT TRAINING AND VALIDATION LOSS
# ==========================================

# Create a figure of size 8 by 5 inches.
plt.figure(figsize=(8, 5))


# Plot training loss for all epochs.
plt.plot(train_losses, label="Training Loss")


# Plot validation loss for all epochs.
plt.plot(val_losses, label="Validation Loss")


# Label the horizontal axis.
plt.xlabel("Epochs")


# Label the vertical axis.
plt.ylabel("Loss")


# Give the graph a title.
plt.title("RNN Training and Validation Loss")


# Display the legend.
plt.legend()


# Display grid lines.
plt.grid()


# Display the graph.
plt.show()


# ==========================================
# 9. PLOT VALIDATION ACCURACY
# ==========================================

# Create another figure.
plt.figure(figsize=(8, 5))


# Plot validation accuracy for every epoch.
plt.plot(val_accuracies, label="Validation Accuracy")


# Label the horizontal axis.
plt.xlabel("Epochs")


# Label the vertical axis.
plt.ylabel("Accuracy")


# Give the graph a title.
plt.title("RNN Validation Accuracy")


# Display the legend.
plt.legend()


# Display grid lines.
plt.grid()


# Display the graph.
plt.show()
