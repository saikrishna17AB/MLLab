import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

df=pd.read_csv("udemy_courses_with_student_ids_and_ratings.csv")

print(df.head())
print("Dataset shape: ",df.shape)

print(df.columns.tolist())
R=df.pivot(index='student_id',columns='course_id',values='course_rating')

print(R)    
print("Interaction Matrix shape: ",R.shape)

#Calculating mean

studentmeans=R.mean(axis=1,skipna=True)

#Subtract the mean rating of each student from all of their ratings

Rcenteredmatrix=R.sub(studentmeans,axis=0)

#Replace missing values with 0
Rcenteredmatrix=R.fillna(0)

#Convert it to numpy array
R=Rcenteredmatrix.to_numpy(dtype=float)

#Performing singular value decomposition

#Finding AT A

A=R.T @ R
print(A)