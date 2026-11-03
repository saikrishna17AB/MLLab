import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv("Xor_Dataset.csv")
X=data.iloc[:,0:2].to_numpy(dtype=float)
Y=data.iloc[:,2:3].to_numpy(dtype=float)

print(X.shape)
print(Y.shape)

np.random.seed(42)

W1=np.random.randn(2,2)     #Weights for first hidden layer
b1=np.zeros((1,2))

W2=np.random.randn(2,1)
b2=np.zeros((1,1))

print("W1: \n",W1)
print("b1: \n",b1)

print("W2: \n",W2)
print("b2: \n",b2)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidderivative(z):
    s=sigmoid(z)
    return s*(1-s)

def forward(X):
    #Apply the input to the first hidden layer
    Z1=np.dot(X,W1)+b1
    A1=sigmoid(Z1)

    #Forward propagate it to output layer
    Z2=np.dot(A1,W2)+b2
    A2=sigmoid(Z2)

    return Z1,A1,Z2,A2

def mseloss(Y,Ypred):
    error=(Y-Ypred)**2
    return np.mean(error)


def backward(X,Y,Z1,A1,Z2,A2):
    m=X.shape[0]
    dZ2=(2/m)*(A2-Y)*sigmoidderivative(Z2)
    dW2=np.dot(A1.dot,dZ2)
    db2=np.sum(dZ2,axis=0,keepdims=True)

    #Hidden layer gradients
    dA1=np.dot(dZ2,W2.T)
    dZ1=dA1*sigmoidderivative(Z1)
    dW1=np.dot(X.T,dZ1)
    db1=np.sum(dZ1,axis=0,keepdims=True)

    return dW1,db1,dW2,db2


lr=0.1
def updateparameters(dW1,db1,dW2,db2):
    global W1,b1,W2,b2
    W1=W1-(lr*dW1)
    b1=b1-(lr*b1)
    W2=W2-(lr*dW2)
    b2=b2-(lr*b2)

#Training the neural network
epochs=1000
losshistory=[]

for i in range(epochs):
    Z1,A1,Z2,A2=forward(X)

    #Loss calculation
    loss=mseloss(Y,A2)
    losshistory.append(loss)

    #Backward propagation

    #Updating the parameters
    updateparameters(W1,b1,W2,b2)


#Calculate the loss of one more forward propagation using the updated parameters

_,_,_,finalprediction=forward(X)
finalloss=mseloss(Y,finalprediction)

#Plotting the loss curve
plt.figure(figsize=(10,8))
plt.plot(losshistory)

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.grid(True)
plt.show()


#Testing the model

testX=np.array([[0,0],[0,1],[1,0],[1,1]],dtype="float")
_,_,_,predictions=forward(testX)

expected=np.array([0,1,1,0])
outputs=[]
for i in range(4):
    probability=predictions[i,0]
    if(probability>=0.5):
        outputs.append(1)
    else:
        outputs.append(0)

    print("Input: ",testX[i])
    print("Expected: ",expected[i])
    print("Actual: ",outputs[i])
    print("Probability: ",probability)

