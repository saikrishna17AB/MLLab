import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data=pd.read_csv("Social_Network_Ads.csv")

data=data.drop("User ID",axis=1)

data["Gender"] = data["Gender"].map({    "Male": 0,"Female": 1})
X = data.drop("Purchased", axis=1)
y = data["Purchased"]


Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.20,random_state=42)

print("\nTraining samples:", len(Xtrain))
print("Testing samples:", len(Xtest))

#Feature Scaling


scaler=StandardScaler()

Xtrain=pd.DataFrame(scaler.fit_transform(Xtrain),columns=Xtrain.columns,index=Xtrain.index)

Xtest=pd.DataFrame(scaler.transform(Xtest),columns=Xtest.columns,index=Xtest.index)

Xnotpurchased = Xtrain[ytrain == 0]
Xpurchased = Xtrain[ytrain == 1]

notpurchased = 0
purchased = 0

for value in ytrain:

    if value == 0:
        notpurchased += 1

    elif value == 1:
        purchased += 1


total = len(ytrain)

priornotpurchased = notpurchased / total
priorpurchased = purchased / total

means={}

means[0]={}
means[1]={}


for feature in Xtrain.columns:
    # Mean for Purchased = 0
    total_value = 0

    for value in Xnotpurchased[feature]:
        total_value += value

    mean=total_value/len(Xnotpurchased)

    means[0][feature] = mean

    # Mean for Purchased = 1

    total_value = 0
    for value in Xpurchased[feature]:
        total_value += value

    mean=total_value / len(Xpurchased)

    means[1][feature]=mean

    variances = {}

variances[0] = {}
variances[1] = {}

#Varience
for feature in Xtrain.columns:

    # Variance for Purchased = 0

    mean = means[0][feature]

    total_value = 0

    for value in Xnotpurchased[feature]:
        total_value += (value - mean) ** 2

    variance = total_value / len(Xnotpurchased)

    variances[0][feature] = variance


    # Variance for Purchased = 1

    mean = means[1][feature]

    total_value = 0

    for value in Xpurchased[feature]:
        total_value += (value - mean) ** 2

    variance = total_value / len(Xpurchased)

    variances[1][feature] = variance

def gaussian_probability(x, mean, variance):

    numerator = math.exp( -((x - mean) ** 2) /(2 * variance))

    denominator = math.sqrt(2 * math.pi * variance)
    return numerator / denominator


def calculate_likelihood(row, class_value):

    likelihood = 1

    # Calculate probability for every feature
    for feature in Xtrain.columns:

        x = row[feature]

        mean = means[class_value][feature]

        variance = variances[class_value][feature]

        probability = gaussian_probability(x,mean,variance)

        likelihood *= probability

    return likelihood

priors = { 0: priornotpurchased,1: priorpurchased
}

def calculate_score(row, class_value):
    likelihood = calculate_likelihood(row,class_value)

    prior=priors[class_value]

    score=prior*likelihood

    return score

predictions = []


for i in range(len(Xtest)):

    row = Xtest.iloc[i]

    score_notpurchased = calculate_score(row,0)

    score_purchased = calculate_score(row,1)

    if score_notpurchased > score_purchased:
        prediction = 0

    else:
        prediction = 1
    predictions.append(prediction)


accuracy = accuracy_score(
    ytest,
    predictions
)

precision = precision_score(
    ytest,
    predictions,
    pos_label=1
)

recall = recall_score(
    ytest,
    predictions,
    pos_label=1
)

f1score = f1_score(
    ytest,
    predictions,
    pos_label=1
)

confusion = confusion_matrix(
    ytest,
    predictions
)


print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1score)

print("\nConfusion Matrix:")
print(confusion)


