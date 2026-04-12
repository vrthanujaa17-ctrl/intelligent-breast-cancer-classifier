import pandas as pd

df = pd.read_csv("/Users/thanujaavr/intelligent-breast-cancer-classifier/data/raw/metabric.csv")
print(df.head())
print(df.shape)
print(df.columns)