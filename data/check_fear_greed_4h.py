import pandas as pd

df = pd.read_parquet("data/raw_data/fear_greed_4h.parquet")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

print("=" * 50)
print("FEAR & GREED 4H CHECK")
print("=" * 50)

print("Rows:", len(df))
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())
print("Duplicate timestamps:", df["timestamp"].duplicated().sum())

print()
print("Timestamp differences:")
print(df["timestamp"].diff().value_counts().head())

print()
print("First 10 rows:")
print(df.head(10))

print()
print("Columns:")
print(df.columns.tolist())