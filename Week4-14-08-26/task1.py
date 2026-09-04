import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("kc_house_data.csv")
print(df.shape)
print(df.columns)

print(df.isna().sum().sum())

features=["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","lat","long","sqft_living15","sqft_lot15"]
target="price"

X=df[features].copy()
y=df[target].copy()


np.random.seed(42)
indices=np.random.permutation(len(df))
X=X.iloc[indices]
y=y.iloc[indices]

split=int(0.8*len(X))

Xtrain=X.iloc[:split].copy()
ytrain=y.iloc[:split].copy()

Xtest=X.iloc[split:].copy()
ytest=y.iloc[split:].copy()

mean=Xtrain.mean()
std=Xtrain.std()

#Scaling
Xtrain=(Xtrain-mean)/std
Xtest=(Xtest-mean)/std

Xtrain=Xtrain.to_numpy(dtype=float)
ytrain=ytrain.to_numpy(dtype=float)

Xtest=Xtest.to_numpy(dtype=float)
ytest=ytest.to_numpy(dtype=float)

Xtrain = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
Xtest = np.c_[np.ones(Xtest.shape[0]), Xtest]

#initlaize weights to zero

B=np.zeros(Xtrain.shape[1])

print("Xtrain shape: ",Xtrain.shape)
print("Xtest shape",Xtest.shape)
print("Number of weights:",len(B))


iterations=10000
alpha=0.001
tolerance=1e-6

n=Xtrain.shape[0]

losshistory=[]
for i in range(iterations):
    ypred= Xtrain @ B
    
    error=ytrain-ypred
    loss=np.mean(error**2)
    
    losshistory.append(loss)
    
    gradient=(-2/n)*(Xtrain.T @ error)
    
    B=B-(alpha*gradient)
    
    gradientnorm=np.sqrt(np.sum(gradient**2))
    if gradientnorm<tolerance:
        print("COnverged")
        break
    
    if i%100==0:
        print("Iteration: ",i)
        print("Loss: ",loss)
        print("Parameters: ",B)
        
        
#training predicitons
    
ytrainpred=Xtrain @ B
trainerror=ytrain-ytrainpred
    
trainmse=np.mean(trainerror**2)
print("Training MSE: ",trainmse)
    
    
    #Testing the data
    
ytestpred=Xtest@B
testerror=ytest-ytestpred
    
testmse=np.mean(testerror**2)
print("Testing MSE: ",testmse)
    
plt.figure(figsize=(8, 6))

plt.scatter(ytrain, ytrainpred, alpha=0.3)

minimum = min(ytrain.min(),ytrainpred.min())
maximum = max(ytrain.max(),ytrainpred.max())

plt.plot([minimum, maximum],[minimum, maximum])

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Training Data: Actual vs Predicted")

plt.grid(True)
plt.show()
    
plt.figure(figsize=(8, 6))

plt.scatter(ytest, ytestpred, alpha=0.3)

minimum = min(ytest.min(), ytestpred.min())
maximum = max(ytest.max(), ytestpred.max())

plt.plot([minimum, maximum],
            [minimum, maximum])

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Testing Data: Actual vs Predicted")

plt.grid(True)
plt.show()