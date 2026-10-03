import os
from pathlib import Path

import databento as db
import pandas as pd


# =========================
# SETTINGS
# =========================

ASSET = "SOL"
SYMBOL = "SOL.v.0"

START_YEAR = 2025
END_YEAR = 2025


# =========================
# SETUP
# =========================

api_key = os.environ["DATABENTO_API_KEY"]
client = db.Historical(api_key)

output_dir = Path("data/raw_data")
output_dir.mkdir(parents=True, exist_ok=True)

all_data = []


# =========================
# DOWNLOAD EACH YEAR
# =========================

for year in range(START_YEAR, END_YEAR + 1):

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    print()
    print("=" * 50)
    print(f"Downloading {ASSET} data for {year}")
    print(f"Date range: {start_date} → {end_date}")
    print("=" * 50)

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="ohlcv-1h",
        symbols=SYMBOL,
        stype_in="continuous",
        start=start_date,
        end=end_date,
    )

    df = data.to_df()

    print(f"Downloaded {len(df)} hourly candles")

    # Make sure timestamps are UTC
    df.index = pd.to_datetime(df.index, utc=True)

    # Convert 1-hour candles → 4-hour candles
    df_4h = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Remove empty 4-hour periods
    df_4h = df_4h.dropna(
        subset=["open", "high", "low", "close"]
    )

    print(f"Created {len(df_4h)} four-hour candles")

    all_data.append(df_4h)


# =========================
# COMBINE ALL YEARS
# =========================

print()
print("Combining all years...")

final_df = pd.concat(all_data)

# Remove any duplicate timestamps
final_df = final_df[~final_df.index.duplicated(keep="first")]

# Sort chronologically
final_df = final_df.sort_index()


# =========================
# SAVE
# =========================

output_file = output_dir / f"{ASSET}_4h.parquet"

final_df.to_parquet(output_file)

print()
print("=" * 50)
print("DOWNLOAD COMPLETE")
print("=" * 50)
print(f"Asset: {ASSET}")
print(f"Total 4-hour candles: {len(final_df)}")
print(f"First timestamp: {final_df.index.min()}")
print(f"Last timestamp: {final_df.index.max()}")
print(f"Saved to: {output_file}")
print("=" * 50)