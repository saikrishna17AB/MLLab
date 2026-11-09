import pandas as pd

df = pd.read_csv("50_Startups.csv")

print(df.shape)
print(df.columns)

print("\nMissing values:")
print(df.isna().sum())

df = pd.get_dummies(df, columns=["State"], drop_first=True)

print("\nAfter One-Hot Encoding:")
print(df.head())

print("\nColumns:")
print(df.columns)

features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State_Florida",
    "State_New York",
]

target = "Profit"

X = df[features].copy()
y = df[target].copy()


