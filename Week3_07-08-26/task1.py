import math

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

traindata = pd.read_csv("SalaryData_Train.csv")
testdata = pd.read_csv("SalaryData_Test.csv")
X = traindata.iloc[:, :-1]
y = traindata.iloc[:, -1]

numericalfeatures = ["age", "educationno", "capitalgain", "capitalloss", "hoursperweek"]
categoricalfeatures = [
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native",
]

for feature in categoricalfeatures:
    print(feature, ":", traindata[feature].nunique())

# Missing values in each column

# print(traindata.isnull().sum())

# total number of missing values

totalmissing = traindata.isnull().sum().sum()
print(totalmissing)

print(traindata["Salary"].unique())


# calculating prior probability
less = 0
more = 0
for salary in traindata["Salary"]:
    if salary == " <=50K":
        less = less + 1
    elif salary == " >50K":
        more = more + 1

total = len(traindata)

lessprobability = less / total
print("Less than 50K:", lessprobability)
moreprobability = more / total
print("More than 50K:", moreprobability)


# Preprocessing data

Xtrain = traindata.drop("Salary", axis=1)
ytrain = traindata["Salary"]

Xtest = testdata.drop("Salary", axis=1)
ytest = testdata["Salary"]

print("Missing values in training data:", Xtrain.isnull().sum())
print("Missing values in test data:", Xtest.isnull().sum())

# Encoding it

mappings = {}
for column in categoricalfeatures:
    categories = Xtrain[column].unique()

    mapping = {}
    for i in range(len(categories)):
        mapping[categories[i]] = i

    mappings[column] = mapping

# Apply the mapping to both datasets
for column in categoricalfeatures:
    Xtrain[column] = Xtrain[column].map(mappings[column])
    Xtest[column] = Xtest[column].map(mappings[column])


Xless = Xtrain[ytrain == " <=50K"]
Xmore = Xtrain[ytrain == " >50K"]

print(Xless)
print(Xmore)

# Calculate mean for both

means = {}
means[" <=50K"] = {}
means[" >50K"] = {}

for feature in Xtrain.columns:
    total = 0
    for value in Xless[feature]:
        total += value

    meanless = total / len(Xless)
    means[" <=50K"][feature] = meanless

    # Mean for >50K

    total = 0
    for value in Xmore[feature]:
        total += value
    meanmore = total / len(Xmore)

    means[" >50K"][feature] = meanmore

print("\nMeans for <=50K:")
print(means[" <=50K"])

print("\nMeans for >50K:")
print(means[" >50K"])

# Varience

variances = {}

variances[" <=50K"] = {}
variances[" >50K"] = {}


for feature in Xtrain.columns:
    mean = means[" <=50K"][feature]

    total = 0

    for value in Xless[feature]:
        total += (value - mean) ** 2

    variance_less = total / len(Xless)

    variances[" <=50K"][feature] = variance_less

    mean = means[" >50K"][feature]

    total = 0

    for value in Xmore[feature]:
        total += (value - mean) ** 2

    variance_more = total / len(Xmore)

    variances[" >50K"][feature] = variance_more

print("\nVariances for <=50K:")
print(variances[" <=50K"])

print("\nVariances for >50K:")
print(variances[" >50K"])


def gaussian_probability(x, mean, variance):

    numerator = math.exp(-((x - mean) ** 2) / (2 * variance))

    denominator = math.sqrt(2 * math.pi * variance)

    return numerator / denominator


def calculate_likelihood(row, class_value):

    # Initially likelihood is 1
    likelihood = 1
    # Go through every feature
    for feature in Xtrain.columns:
        # Value of this feature for the test employee
        x = row[feature]

        # Mean of this feature for the particular class
        mean = means[class_value][feature]

        # Variance of this feature for the particular class
        variance = variances[class_value][feature]

        probability = gaussian_probability(x, mean, variance)

        likelihood *= probability

    return likelihood


priors = {" <=50K": lessprobability, " >50K": moreprobability}


def calculatescore(row, classvalue):
    likelihood = calculate_likelihood(row, classvalue)

    prior = priors[classvalue]
    score = prior * likelihood
    return score


# Make predicitons

predictions = []

for i in range(len(Xtest)):
    row = Xtest.iloc[i]

    scoreless = calculatescore(row, " <=50K")
    scoremore = calculatescore(row, " >50K")

    # choose the class with higher score
    if scoreless > scoremore:
        prediciton = " <=50K"
        predictions.append(prediciton)
    else:
        prediciton = " >50K"
        predictions.append(prediciton)

# for i in range(len(Xtest)):
#     print("Test sample: ",i+1,"Actual value: ",ytest.iloc[i],"Predicted: ",predictions[i])

accuracy = accuracy_score(ytest, predictions)
precision = precision_score(ytest, predictions, pos_label=" >50K")
recall = recall_score(ytest, predictions, pos_label=" >50K")
f1score = recall_score(ytest, predictions, pos_label=" >50K")

confusion = confusion_matrix(ytest, predictions, labels=[" <=50K", " >50K"])

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1score)

print("\nConfusion Matrix:")
print(confusion)
