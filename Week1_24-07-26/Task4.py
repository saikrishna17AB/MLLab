import numpy as np
from pathlib import Path
import os
import pandas as pd

folder="CustomerDataset"

matrix=np.random.rand(5,5)
print(matrix)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if(matrix[i][j]>0.5):
            matrix[i][j]=1

print(matrix)



# for file in os.listdir(folder):
#     filepath=os.path.join(folder,file)
#     f=open(filepath)
#     print(f.read())
#     f.close()