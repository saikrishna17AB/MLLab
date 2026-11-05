import numpy as np
import pandas as pd

def entropy(y):
    unique,counts=np.unique(y,return_counts=True)
    total=len(unique)
    e=0
    for i in counts:
        p=i/total
        e-=(p*np.log2(p))

    return e

def informationgain(y,lefty,righty):
    parententropy=entropy(y)
    n=len(y)

    leftn=len(lefty)
    rightn=len(righty)

    we=((leftn/n)*entropy(lefty))+((rightn/n)*entropy(righty))
    return parententropy-we

def findbestsplit(X,y):
    bestfeature=None
    bestgain=-1
    bestiscategorical=False
    bestthreshold=None

    for feature in X.columns:
        c=X[feature]

        if c.dtype=="object":

            uniquevalues=c.unique()
            for i in uniquevalues:
                leftmask=c==i
                rightmask=c!=i

                yleft=y[leftmask]
                yright=y[rightmask]

                if(len(yleft)==0 or len(yright)==0):
                    continue
                gain=informationgain(y,yleft,yright)

                if gain>bestgain:
                    bestgain=gain
                    bestiscategorical=True
                    bestfeature=feature
                    bestthreshold=i

        else:
            values=np.sort(c.unique())
            if(len(values)<2):
                continue

            for i in range(len(values)-1):
                threshold=(values[i]+values[i+1])/2
                leftmask=c<=i
                rightmask=c>i

                yleft=y[leftmask]
                yright=y[rightmask]

                if(len(yleft)==0 or len(yright)==0):
                    continue

                gain=informationgain(y,yleft,yright)
                if gain>bestgain:
                    bestgain=gain
                    bestfeature=feature
                    bestthreshold=threshold
                    bestiscategorical=False

    return (bestfeature,bestthreshold,bestgain,bestiscategorical)

def majorityclass(y):
    return y.value_counts().idxmax()

def buildtree(X,y,depth=0,maxdepth=4,minsamplesplit=2):
    if(len(y.unique())==1):
        return {
            "leaf":True,
            "prediction":y.iloc[0]
        }

    if depth>=maxdepth:
        return {
            "leaf":True,
            "prediction":majorityclass(y)
        }    

    if len(y)<minsamplesplit:
        return {
                "leaf":True,
                "prediction":majorityclass(y)
        }

    (bestfeature,bestthreshold,bestgain,bestiscategorical)=findbestsplit(X,y)

    if bestfeature is None or bestgain<0:
        return {
            "leaf":True,
            "prediction":majorityclass(y)
        }

    #Split the data
    
    if bestiscategorical:
        leftmask=(X[bestfeature]==bestthreshold)
        rightmask=(X[bestfeature]!=bestthreshold)

    else:
        leftmask=(X[bestfeature]<=bestthreshold)
        rightmask=(X[bestfeature]>bestthreshold)

    Xleft=X[leftmask]
    yleft=y[leftmask]

    Xright=X[rightmask]
    yright=y[rightmask]

    #Recursively build the tree

    leftsubtree=buildtree(Xleft,yleft,depth=depth+1,maxdepth=maxdepth,minsamplesplit=minsamplesplit)
    rightsubtree=buildtree(Xright,yright,depth=depth+1,maxdepth=maxdepth,minsamplesplit=minsamplesplit)

    #Return the current internal node

    return {
        "leaf":False,
        "feature":bestfeature,
        "threshold":bestthreshold,
        "iscategorical":bestiscategorical,
        "gain":bestgain,
        "left":leftsubtree,
        "right":rightsubtree
    }


def predictone(row,tree):
    if tree["leaf"]:
        return tree["prediction"]

    feature=tree["feature"]
    threshold=tree["threshold"]

    if tree["iscategorical"]:
        if row[feature]==threshold:
            return predictone(row,tree["left"])
        else:
            return predictone(row,tree["right"])

    else:
        if row[feature]<=threshold:
            return predictone(row,tree["left"])
        else:
            return predictone(row,tree["right"])

def predict(X,tree):
    predictions=[]
    for _,row in X.iterrows():

        prediction=predictone(row,tree)
        predictions.append(prediction)

    return predictions

df=pd.read_csv("drug200.csv")

X=df.drop("Drug",axis=1)
y=df["Drug"]

splitindex=int(0.67*len(df))
Xtrain=X.iloc[:splitindex]
ytrain=y.iloc[:splitindex]

Xtest=X.iloc[:splitindex]
ytest=y.iloc[:splitindex]

tree=buildtree(Xtrain,ytrain,0,10,2)
predictions=predict(Xtest,tree)

total=len(ytest)
corr=0
for predicted,correct in zip(predictions,ytest):
    if predicted==correct:
        corr+=1    

accuracy=corr/total
print(predictions)

print(accuracy)

