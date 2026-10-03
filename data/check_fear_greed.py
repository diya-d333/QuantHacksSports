import pandas as pd

df = pd.read_parquet("data/raw_data/fear_greed_daily.parquet")

print("=" * 50)
print("FEAR & GREED DATA CHECK")
print("=" * 50)

print("Number of rows:", len(df))
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())
print("Duplicate timestamps:", df["timestamp"].duplicated().sum())

print()
print("First 5 rows:")
print(df.head())

print()
print("Last 5 rows:")
print(df.tail())

print()
print("Data types:")
print(df.dtypes)