import pandas as pd

INPUT_FILE = "data/raw_data/fear_greed_daily.parquet"
OUTPUT_FILE = "data/raw_data/fear_greed_4h.parquet"

print("Loading Fear & Greed data...")

df = pd.read_parquet(INPUT_FILE)

# Make sure timestamp is the index
df = df.set_index("timestamp").sort_index()

# Shift by one day to avoid using a daily value
# before we can safely assume it is available.
df["fear_greed_score"] = df["fear_greed_score"].shift(1)
df["fear_greed_classification"] = df[
    "fear_greed_classification"
].shift(1)

# Create 4-hour timestamps and carry the most recently
# available sentiment value forward.
df = df.resample("4h").ffill()

# Remove rows where there is no previous-day sentiment yet.
df = df.dropna(subset=["fear_greed_score"])

# Add useful features
df["fear_greed_score"] = df["fear_greed_score"].astype(int)

df["fear_greed_change_1d"] = df["fear_greed_score"].diff(6)

df["fear_greed_7d_avg"] = (
    df["fear_greed_score"]
    .rolling(42)
    .mean()
)

df["fear_greed_30d_avg"] = (
    df["fear_greed_score"]
    .rolling(180)
    .mean()
)

df["extreme_fear_flag"] = (
    df["fear_greed_score"] <= 24
).astype(int)

df = df.reset_index()

df.to_parquet(OUTPUT_FILE, index=False)

print()
print("=" * 50)
print("FEAR & GREED 4H DATA CREATED")
print("=" * 50)
print("Rows:", len(df))
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())
print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
print("Saved to:", OUTPUT_FILE)

print()
print(df.head(10))