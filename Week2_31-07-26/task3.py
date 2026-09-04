import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import statistics
from scipy.stats import norm,laplace,gamma,beta,poisson,binom
df=pd.read_csv("thyroid_cancer_risk_data.csv")


def gaussian_pdf(x,mean,std):
    return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-((x - mean) ** 2) / (2 * std ** 2))

def laplace_pdf(x, mu, b):
    return (1 / (2 * b)) * math.exp(-abs(x - mu) / b)

def gamma_pdf(x, k, theta):
    if x <= 0:
        return 0

    numerator = (x ** (k - 1)) * math.exp(-x / theta)
    denominator = math.gamma(k) * (theta ** k)
    return numerator / denominator

def beta_pdf(x, alpha, beta):

    if x <= 0 or x >= 1:
        return 0

    numerator = (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))

    denominator = (math.gamma(alpha) * math.gamma(beta)) / math.gamma(alpha + beta)

    return numerator / denominator

def poisson_pmf(x, lam):

    if x < 0:
        return 0

    return (math.exp(-lam) * (lam ** x)) / math.factorial(int(x))

def binomial_pmf(x, n, p):

    if x < 0 or x > n:
        return 0

    combination = math.comb(n, int(x))

    return combination * (p ** x) * ((1 - p) ** (n - x))

def calculateerror(hist,t):
    sse=0
    for i in range(len(hist)):
        sse+=(hist[i]-t[i])**2
    return sse

#calculating the feature



df["Risk_Factor_Count"]=0
for i in range(len(df)):
    count=0
    if df.loc[i,"Radiation_Exposure"]=="Yes":
        count+=1
    if df.loc[i,"Iodine_Deficiency"]=="Yes":
        count+=1
    if df.loc[i,"Smoking"]=="Yes":
        count+=1
    if df.loc[i,"Obesity"]=="Yes":
        count+=1
    df.loc[i,"Risk_Factor_Count"]=count

    
        
t1=df["Risk_Factor_Count"]
print(t1)


print("Minimin value: ",t1.min())
print("Maximum value: ",t1.max())

uniquevalues=set()
for value in t1:
    uniquevalues.add(value)

length=len(uniquevalues)
datatype=t1.dtype
if(np.issubdtype(datatype,np.integer) and length<50):
    print("Discerte data")
else:
    print("Continous data")

#calculating parameters
mean=sum(t1)/len(t1)
varience=0
for x in t1:
    varience+=((x-mean)**2)

varience/=len(t1)
std=math.sqrt(varience)

X=[]
for x in sorted(uniquevalues):
    X.append(x)

observed=[]
for x in X:
    count=0
    for val in t1:
        if x==val:
            count+=1
    observed.append(count/len(t1))

lam=mean
pmfpoisson=[]
for x in X:
    pmfpoisson.append(poisson_pmf(x,lam))

n=len(uniquevalues)-1
p=mean/n
pmfbinomial=[]
for x in X:
    pmfbinomial.append(binomial_pmf(x,n,p))

errorpoission=calculateerror(observed,pmfpoisson)
errorbinomial=calculateerror(observed,pmfbinomial)

errors={"Poisson":errorpoission,"Binomial":errorbinomial}

bestdistribution=min(errors,key=errors.get)

print("Closest distribution: ",bestdistribution)
print("Error: ",errors[bestdistribution])

if bestdistribution=="Poisson":
    bestpmf=pmfpoisson
elif bestdistribution=="Binomial":
    bestpmf=pmfbinomial




plt.figure(figsize=(8,5))

plt.bar(X,observed)
plt.plot(X,bestpmf,color='red',label=bestdistribution)
plt.xlabel("RiskLevel")
plt.ylabel("Density")
plt.show()







# mean=t1.mean()
# sd=t1.std()

# plt.hist(t1,bins=30,density=True,edgecolor='black')

# x=np.linspace(t1.min(),t1.max(),1000)
# y=norm.pdf(x,loc=mean,scale=sd)
# plt.plot(x,y,'r',linewidth=2)
# plt.show()

# feature="T3_Level"

# plt.figure(figsize=(8,5))

# plt.hist(df[feature],bins=30,density=True,edgecolor='black')
# plt.xlabel(feature)
# plt.ylabel("Density")
# plt.title(feature)
# plt.show()