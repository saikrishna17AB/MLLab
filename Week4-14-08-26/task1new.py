import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("kc_house_data.csv")
print(df.shape)
print(df.columns)

print(df.isna().sum().sum())

features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]
target = "price"

X = df[features]
y = df[target]
Xtrain, ytrain, Xtest, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.fit_transform(Xtest)

Xtrain = np.array(Xtrain, dtype=float)
Xtest = np.array(Xtest, dtype=float)

ytrain = ytrain.to_numpy(dtype=float)
ytest = ytest.to_numpy(dtype=float)

Xtrain = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
Xtest = np.c_[np.ones(Xtest.shape[0]), Xtest]

B = np.zeros(Xtrain.shape[1])
print(B)
