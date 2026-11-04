import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load directly as a structured Pandas DataFrame
iris_data = load_iris(as_frame=True)
df = iris_data.frame

# Display the continuous features to be discretized
print(df.columns)

features=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
df=df[features].copy()

#Divie the datagframe into ranges
discreteddf=pd.DataFrame()

for column in df.columns:
    discreteddf[column]=pd.cut(df[column],bins=10,labels=False,include_lowest=True)
print(discreteddf.head())

def calculatepmf(df,x):
    freq=np.bincount(df[x],minlength=10)
    pmf=freq/len(df)
    return pmf

pmfs={}
for column in discreteddf.columns:
    pmfs[column]=calculatepmf(discreteddf,column)

for column in pmfs:
    print("\nPMF of", column)
    print(pmfs[column])
    print("Sum =", np.sum(pmfs[column]))


# Check the PMF of the first feature
first_feature = features[0]

print("\nFirst Feature PMF Check:")
print("Feature:", first_feature)
print("PMF:", pmfs[first_feature])
print("Sum of PMF:", np.sum(pmfs[first_feature]))

def entropy(p):
    #Remove zero probabilities
    p=p[p>0]

    h=-np.sum(p* np.log2(p))
    return h

def jointentropy(x,y):

    #calculate joint proabbility frequency table

    jointfreq=pd.crosstab(df[x],df[y])

    jointpmf=jointfreq/len(df)

    p=jointpmf.to_numpy()
    p=p[p>0]

    hxy=-np.sum(p*np.log2(p))
    return hxy
def mutualinformation(x,y):
    px=calculatepmf(discreteddf,x)
    py=calculatepmf(discreteddf,y)

    hx=entropy(px)
    hy=entropy(py)

    hxy=jointentropy(x,y)
    m1=hx+hy-hxy

    return m1


mi_matrix = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        mi_matrix[i][j] = mutualinformation(features[i],features[j])

# Convert matrix to DataFrame for better display
mi_df = pd.DataFrame(
    mi_matrix,
    index=features,
    columns=features
)

print("\nMutual Information Matrix:")
print(mi_df)



max_mi = -1
feature1 = ""
feature2 = ""

for i in range(4):

    for j in range(i + 1, 4):

        if mi_matrix[i][j] > max_mi:

            max_mi = mi_matrix[i][j]
            feature1 = features[i]
            feature2 = features[j]


print("\nMost Predictive Feature Pair:")
print(feature1, "and", feature2)

print("Mutual Information:", max_mi)

plt.figure(figsize=(8, 6))

sns.heatmap(
    mi_df,
    annot=True,
    fmt=".4f"
)

plt.title("Mutual Information Matrix")
plt.xlabel("Features")
plt.ylabel("Features")

plt.tight_layout()
plt.show()
