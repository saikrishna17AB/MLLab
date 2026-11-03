import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

traindata=pd.read_csv("ds1_train.csv")
validationdata=pd.read_csv("ds1_valid.csv")
testdata=pd.read_csv("ds1_test.csv")

Xtrain=traindata[["x_1","x_2"]]
ytrain=traindata["y"]


Xvalidation=validationdata[["x_1","x_2"]]
yvalidation=validationdata["y"]

Xtest=testdata[["x_1","x_2"]]
ytest=testdata["y"]

#Feature scaling

scaler=StandardScaler()

Xtrain=scaler.fit_transform(Xtrain)
Xvalidation=scaler.transform(Xvalidation)
Xtest=scaler.transform(Xtest)


def calculatedistance(p1,p2):
    distance=math.sqrt(((p2[0]-p1[0])**2) + ((p2[1]-p1[1])**2))
    return distance

def predict(point,k):
    distances=[]
    for i in range(len(Xtrain)):
        d=calculatedistance(point,Xtrain[i])
        distances.append((d,ytrain.iloc[i]))
        
    distances.sort(key=lambda x: x[0])
    
    nearestneighbors=distances[:k]
    
    votes={}
    
    for distance,label in nearestneighbors:
        if label not in votes:
            votes[label]=0
        votes[label]+=1
        
    predictedclass=max(votes,key=votes.get)
    return predictedclass

def calculateaccuracy(X,y,k):
    correct=0
    
    # Predict every point in the given dataset
    for i in range(len(X)):

        prediction = predict(X[i], k)

        # Compare prediction with actual class
        if prediction == y[i]:
            correct += 1


    accuracy = correct / len(X)

    return accuracy

validationaccuracies=[]
bestk=1
bestaccuracy=0

print("validation results:")

for k in range(1,21):
    accuracy=calculateaccuracy(Xvalidation,yvalidation,k)
        
    validationaccuracies.append(accuracy)
    
    print("K =", k, "Validation Accuracy =", accuracy)


    # Check whether this K is better
    # than the previous best K
    if accuracy > bestaccuracy:

        bestaccuracy = accuracy
        bestk = k
        

print("\nBest K =", bestk)

print("Best Validation Accuracy =", bestaccuracy)
test_accuracy = calculateaccuracy(Xtest, ytest, bestk)


print("\nFinal Test Results")


print("Selected K =", bestk)

print("Test Accuracy =", test_accuracy)
    
    
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 21),
    validationaccuracies,
    marker='o'
)

plt.xlabel("K")
plt.ylabel("Validation Accuracy")
plt.title("K vs Validation Accuracy")

plt.xticks(range(1, 21))

plt.grid(True)

plt.show()


# Confusion Matrix

test_predictions = []

for i in range(len(Xtest)):

    prediction = predict(
        Xtest[i],
        bestk
    )

    test_predictions.append(prediction)



cm = confusion_matrix(
    ytest,test_predictions

)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("KNN Confusion Matrix - Test Data")

plt.show()
    







