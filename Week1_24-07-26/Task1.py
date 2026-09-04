import pandas as pd
import os
folder="wine+quality"

for file in os.listdir(folder):
    filepath=os.path.join(folder,file)

    df=pd.read_csv(filepath)
    for i in range(rows):
        for column in df["TotalCharges"]:
            if df.loc[i,column]=="One Year":
                df.loc[i,column]=1
            elif df.loc[i,column]=="Month-to-month":
                df.loc[i,column]=2
            

                        
