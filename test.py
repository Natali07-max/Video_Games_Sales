import pandas as pd

#load dataset
df = pd.read_csv("Dataset/vgsales.csv")

print("First 5 Rows:")
print(df.head())

print("Last 5 Rows:")
print(df.tail())

print("Columns Names:")
print(df.columns)

print("Information display")
print(df.info())

print("Missing Values")
print(df.isnull().sum())

print("Duplicate values")
print(df.duplicated().sum)

print("Statistical Summary")
print(df.describe())

