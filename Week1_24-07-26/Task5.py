import pandas as pd
import os
folder="CustomerDataset"

for file in os.listdir(folder):
    filepath=os.path.join(folder,file)

    df=pd.read_csv(filepath)
    # print(df)
    # print()
    # print(df.describe())

    rows=len(df)

    for column in df.columns:
        missing=0

        for value in df[column]:
            if pd.isna(value):
                missing=missing+1

        print(missing)
        if missing> (rows/2):
            df=df.drop(column,axis=1)
    print(df)
    print(df.describe())
