import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

scaler = StandardScaler()

df = pd.read_csv("Wine.csv")

X = df.drop("Customer_Segment", axis=1).values
y = df["Customer_Segment"]

Xtrain, ytrain, Xtest, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrainscaled = scaler.fit_transform(Xtrain)
Xtestscaled = scaler.fit_transform(Xtest)

#LDA Dimensionality reduction

classes=np.unique(ytrain)

overallmean=np.mean(Xtrainscaled,axis=1)
nfeatures=Xtrainscaled.shape[1]

Sw=np.zeros((nfeatures,nfeatures))
Sb=np.zeros((nfeatures,nfeatures))
for c in classes:
    data=Xtrainscaled[ytrain==c]
    classmean=np.mean(data,axis=0)

    centered=data-classmean
    sw+=centered.T @ centered

    #Between scattered matrix
    difference=centered-overallmean
    sb+=len(data)*(difference @ difference.T)

matrix=np.linalg.solve(sw,sb)

eigenvalues,eigenvectors=np.linalg.eig(matrix)

eigenvalues=np.real_if_close(eigenvalues)
eigenvectors=np.real_if_close(eigenvectors)

#Sort eigenvalues in desc order
indexes=np.argsort(eigenvalues.real)[::-1]
eigenvalues=eigenvalues[indexes]
eigenvectors=eigenvectors[:indexes]

ncomponents=2

components=eigenvectors[:ncomponents]

#Reduced dimension matrix
Xtrainedscaled=Xtrainedscaled @ components
Xtestscaled=Xtestscaled @ components

print("Reduced dimension data :")
print(Xtrainedscaled.shape())
print(Xtestscaled.shape())

nclasses=len(classes)

ytrainencoded=np.searchsorted(classes,ytrain)


