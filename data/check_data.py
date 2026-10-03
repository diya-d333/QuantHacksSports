import pandas as pd

df = pd.read_parquet("data/raw_data/BTC_4h.parquet")

print("Number of candles:", len(df))
print()

print("First 10 timestamps:")
print(df.index[:10])
print()

print("Time differences between candles:")
print(df.index.to_series().diff().value_counts())
