import pandas as pd
import numpy as np
import os
import random
folder="wine+quality"

for file in os.listdir(folder):


    if(file.endswith(".csv")==False):
        continue

    filepath=os.path.join(folder,file)




    df=pd.read_csv(filepath,sep=';')

    
    rows=len(df)

    missing=False
    for i in range(rows):
        for column in df.columns:
            if pd.isna(df.loc[i,column]):
                df=df.drop(i)
                break
    df=df.reset_index(drop=True)
                

    trainsize=int(rows*0.8)
    testsize=rows-trainsize

    #Split the dataset

    randomindices=random.sample(range(rows),rows)
    # print(randomindices)

    traindata=pd.DataFrame(columns=df.columns)
    testdata=pd.DataFrame(columns=df.columns)

    for i in range(trainsize):
        traindata.loc[i]=df.loc[randomindices[i]]        

    for i in range(trainsize,len(randomindices)):
        testdata.loc[i]=df.loc[randomindices[i]]

    # traindata=df.iloc[:trainsize]
    # testdata=df.iloc[trainsize:]

    print("Training data")
    print(traindata)

    print()
    print("Testing data")
    print(testdata)