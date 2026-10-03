import pandas as pd

df = pd.read_parquet("data/raw_data/ETH_4h.parquet")

print("Number of 4-hour candles:", len(df))
print()
print("First 5 candles:")
print(df.head())
print()
print("Last 5 candles:")
print(df.tail())
print()
print("Data types:")
print(df.dtypes)

print("First timestamp:", df.index.min())
print("Last timestamp:", df.index.max())
print("Number of candles:", len(df))
