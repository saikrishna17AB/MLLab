import pandas as pd
import matplotlib.pyplot as plt
import os
folder="CustomerDataset"

for file in os.listdir(folder):
    filepath=os.path.join(folder,file)

    df=pd.read_csv(filepath)

    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.hist(df[df.columns[0]])
    plt.title(df.columns[0])

    plt.subplot(2,2,2)
    plt.hist(df[df.columns[1]])
    plt.title(df.columns[1])

    plt.subplot(2,2,3)
    plt.hist(df[df.columns[2]])
    plt.title(df.columns[2])

    plt.subplot(2,2,4)
    plt.hist(df[df.columns[3]])
    plt.title(df.columns[3])
    
    
    plt.tight_layout()
    plt.show()
        
