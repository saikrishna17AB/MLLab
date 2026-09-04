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




t1=df["T3_Level"]
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

hist,bin_edges=np.histogram(t1,bins=30,density=True)
print(hist)
print(bin_edges)
bin_center=[]

for i in range(len(bin_edges)-1):
    bin_center.append((bin_edges[i]+bin_edges[i+1])/2)

print(bin_center)

pdfgaussian=[]

#Gaussian
for x in bin_center:
    val=gaussian_pdf(x,mean,std)
    pdfgaussian.append(val)

#Gamma

k=(mean**2)/varience
theta=varience/mean
pdfgamma=[]
for x in bin_center:
    val=gamma_pdf(x,k,theta)
    pdfgamma.append(val)

#Beta 

v=(mean*(1-mean)/varience)-1
alpha=mean*v
beta=(1-mean)*v
pdfbeta=[]
for x in bin_center:
    val=beta_pdf(x,alpha,beta)
    pdfbeta.append(val)

errorgaussian=calculateerror(hist,pdfgaussian)
errorgamma=calculateerror(hist,pdfgamma)
errorbeta=calculateerror(hist,pdfbeta)

errors={"Gaussian":errorgaussian,"Gamma":errorgamma,"Beta":errorbeta}
bestdistribution=min(errors,key=errors.get)

print("Closest distribution: ",bestdistribution)
print("Error: ",errors[bestdistribution])

if bestdistribution=="Gaussian":
    bestpdf=pdfgaussian
elif bestdistribution=="Gamma":
    bestpdf=pdfgamma
elif bestdistribution=="Beta":
    bestpdf=pdfbeta

plt.figure(figsize=(8,5))

plt.hist(t1,bins=30,density=True,edgecolor='black')
plt.plot(bin_center,bestpdf,color='red',label=bestdistribution)
plt.xlabel("T3_Level")
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