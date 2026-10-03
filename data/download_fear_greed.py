import requests
import pandas as pd
from pathlib import Path

URL = "https://api.alternative.me/fng/?limit=0"

print("Downloading Crypto Fear & Greed Index...")

response = requests.get(URL, timeout=30)
response.raise_for_status()

data = response.json()["data"]

df = pd.DataFrame(data)

# Convert columns to appropriate types
df["value"] = pd.to_numeric(df["value"])
df["timestamp"] = pd.to_datetime(
    pd.to_numeric(df["timestamp"]),
    unit="s",
    utc=True
)

# Rename value to something descriptive
df = df.rename(columns={
    "value": "fear_greed_score",
    "value_classification": "fear_greed_classification"
})

# Keep only the columns we need
df = df[
    [
        "timestamp",
        "fear_greed_score",
        "fear_greed_classification"
    ]
]

# Sort chronologically
df = df.sort_values("timestamp")

# Remove duplicate timestamps if any
df = df.drop_duplicates(subset=["timestamp"])

output_dir = Path("data/raw_data")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "fear_greed_daily.parquet"
df.to_parquet(output_file, index=False)

print()
print("=" * 50)
print("FEAR & GREED DOWNLOAD COMPLETE")
print("=" * 50)
print(f"Rows: {len(df)}")
print(f"First timestamp: {df['timestamp'].min()}")
print(f"Last timestamp: {df['timestamp'].max()}")
print(f"Saved to: {output_file}")
print()
print(df.head())
print()
print(df.tail())