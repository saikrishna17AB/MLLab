import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

k=11
df=pd.read_csv("ds1_test.csv")

scaler=StandardScaler()

Xtest=df.drop("y",axis=1)
ytest=df["y"]

Xtest=scaler.fit_transform(Xtest)

def euclediandistance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def predictpoint(x1,y1):
    distances=[]
    for i in range(len(Xtest)):
        distance=euclediandistance(x1,y1,Xtest[i][0],Xtest[i][1])
        if distance==0:
            continue
        distances.append((distance,ytest[i]))

    distances.sort(key=lambda x:x[0])
    neighbors=distances[:k]
    labels=[]
    for _,label in neighbors:
        labels.append(label)

    prediction=max(set(labels),key=labels.count)
    return prediction

def predictions():
    predictions=[]
    correct=[]
    for i in range(len(Xtest)):
        prediction=predictpoint(Xtest[i][0],Xtest[i][1])
        predictions.append(prediction)
        correct.append(prediction==ytest[i])

    print(predictions)
    print(correct)
    ncorrect=0
    for i in range(len(correct)):
        if(correct[i]==True):
            ncorrect+=1
    return ncorrect/len(correct)

accuracy=predictions()
print(accuracy)                   

