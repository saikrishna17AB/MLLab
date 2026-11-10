# ANN are not used to image classification becz:
#       Too much computation
#       Treats local pixels same as pixels far apart
#       Sensitive to location of an object in an image

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

datasetdir = Path("shoeTypeClassifierDataset")

outputdir = datasetdir / "dataset_split"

sourcedir = [datasetdir / "training", datasetdir / "validation"]
imageextensions = [".png", ".jpeg", ".jpg"]

classes = sorted([folder.name for folder in sourcedir[0].iterdir() if folder.is_dir()])
# print(classes)

# #Create separate folders for different images
# for split in ["train","test","validation"]:
#     for cls in classes:
#         (outputdir / split / cls).mkdir(parents=True,exist_ok=True)


# #Process each image independently

# for cls in classes:
#     images=[]
#     for source in sourcedir:
#         classdir=source / cls

#         for img in classdir.iterdir():
#             if img.suffix.lower() in imageextensions:
#                 images.append(img)

#     #First split: 70% training, 30% remaining
#     trainimages, remainingimages=train_test_split(images,test_size=0.30,random_state=42)

#     #Second split: 15% training, 15% validation
#     validationimages, testimages=train_test_split(remainingimages,test_size=0.50,random_state=42)

#     #Copy the images into their respective folders

#     for img in trainimages:
#         shutil.copy2(img,outputdir / "train" / cls / img.name)

#     for img in validationimages:
#         shutil.copy2(img,outputdir / "validation" / cls / img.name)

#     for img in testimages:
#         shutil.copy2(img,outputdir / "test" / cls / img.name)

#     print("Class: ",cls)
#     print("Train images: ",len(trainimages))
#     print("Validation images: ",len(validationimages))
#     print("Test images: ",len(testimages))

# print("Dataset splitting completed")


imgheight = 150
imgwidth = 150
batchsize = 32
epochs = 5

seed = 42  # Set random seed

trainds = tf.keras.utils.image_dataset_from_directory(
    outputdir / "train",
    image_size=(imgheight, imgwidth),
    batch_size=batchsize,
    label_mode="int",
    color_mode="rgb",
    shuffle=True,
    seed=seed,
)
validationds = tf.keras.utils.image_dataset_from_directory(
    outputdir / "validation",
    image_size=(imgheight, imgwidth),
    batch_size=batchsize,
    label_mode="int",
    color_mode="rgb",
    shuffle=False,
    seed=seed,
)
testds = tf.keras.utils.image_dataset_from_directory(
    outputdir / "test",
    image_size=(imgheight, imgwidth),
    batch_size=batchsize,
    label_mode="int",
    color_mode="rgb",
    shuffle=False,
    seed=seed,
)

# Use prefetch to improve data performance loading
autotune = tf.data.AUTOTUNE

trainds = trainds.prefetch(buffer_size=autotune)
validationds = validationds.prefetch(buffer_size=autotune)
testds = testds.prefetch(buffer_size=autotune)

# Plot the first 9 images in the training dataset

# Get one batch of training images and their labels
# images,labels=next(iter(trainds))

# plt.figure(figsize=(10,10))

# for i in range(9):
#     ax=plt.subplot(3,3,i+1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(classes[labels[i].numpy()])

#     plt.axis("off")

# plt.show()


# CNN pipeline

model = models.Sequential(
    [
        # Define the input size and color channels
        layers.Input(shape=(150, 150, 3)),
        # #Normalize pixel values from range 0-255 to 0-1
        # layers.Rescaling(1./255),
        # #Randomly flip the images horizontally while training
        # layers.RandomFlip("horizontal"),
        # # #Rotate the images slightly
        # layers.RandomRotation(0.03),
        # First CNN layer
        layers.Conv2D(
            32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        # Max POOL layer8_
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        # Seocnd CNN layer with 64 filters
        layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        # Max POOL layer
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        # Third CNN layer
        layers.Conv2D(
            128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        # GLobal Max pool layer
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        # Fourth CNN layer
        layers.Conv2D(
            256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        # Fully connected layer containing 128 layers
        layers.Dense(128, activation="relu"),
        # Output layer containing 6 neurons , one for each class
        layers.Dense(len(classes), activation="softmax"),
    ]
)

model.summary()


# Compile the model: Define optimizer, loss function, evaluation metrics

model.compile(
    # Adam's optimizer updates the learning rate using adaptive learning rates
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    # Sparse categorical cross entropy loss function
    loss="sparse_categorical_crossentropy",
    # Track the accuracy during training and validation
    metrics=["accuracy"],
)
print("Model compiled successfully")

# Train the model

history = model.fit(trainds, validation_data=validationds, epochs=epochs)
print(history)

# Plot the curve

trainloss = history.history["loss"]
valloss = history.history["val_loss"]

trainaccuracy = history.history["accuracy"]
valaccuracy = history.history["val_accuracy"]

epochsrange = range(1, len(trainloss) + 1)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(epochsrange, trainloss, label="Training loss", marker="o")
ax1.plot(epochsrange, valloss, label="Validation loss", marker="o")

ax1.set_xlabel("Epochs")
ax1.set_ylabel("LOSS")

fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(epochsrange, trainaccuracy, label="Training accuracy", marker="s")
ax2.plot(epochsrange, valaccuracy, label="Validation accuracy", marker="s")

ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")


plt.show()

# Evaluate the model on the test dataset

testloss, testaccuracy = model.evaluate(testds)
print("Test loss: ", testloss)
print("Test accuracy: ", testaccuracy * 100)

# Generate prediction scores for the test dataset

# Generate prediciton probabilities for all classes
yprobability = model.predict(testds)

# Select the class with the maximum probability
ypred = np.argmax(yprobability, axis=1)

# Extract the actual class labels from the testing dataset

ytrue = np.concatenate([labels.numpy() for images, labels in testds])

print(classification_report(ytrue, ypred, target_names=classes, digits=4))

# Generate the confusion matrix

cm = confusion_matrix(ytrue, ypred)

print(cm)

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
)

plt.xlabel("Predicated labels")
plt.ylabel("Actual labels")

plt.tight_layout()
plt.show()
