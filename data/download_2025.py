import os
from pathlib import Path

import databento as db
import pandas as pd

api_key = os.environ["DATABENTO_API_KEY"]
client = db.Historical(api_key)

output_dir = Path("data/raw_data")
output_dir.mkdir(parents=True, exist_ok=True)

assets = {
    "BTC": "BTC.v.0",
    "ETH": "ETH.v.0",
}

for asset, symbol in assets.items():

    print()
    print("=" * 50)
    print(f"Downloading {asset} data for 2025")
    print("=" * 50)

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="ohlcv-1h",
        symbols=symbol,
        stype_in="continuous",
        start="2025-01-01",
        end="2026-01-01",
    )

    df = data.to_df()

    print(f"Downloaded {len(df)} hourly candles")

    df.index = pd.to_datetime(df.index, utc=True)

    # Convert 1-hour candles → 4-hour candles
    df_4h = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    df_4h = df_4h.dropna(
        subset=["open", "high", "low", "close"]
    )

    print(f"Created {len(df_4h)} four-hour candles")

    # IMPORTANT:
    # Save as a separate 2025 file.
    output_file = output_dir / f"{asset}_4h_2025.parquet"

    df_4h.to_parquet(output_file)

    print(f"Saved to: {output_file}")

    print(f"First timestamp: {df_4h.index.min()}")
    print(f"Last timestamp: {df_4h.index.max()}")

print()
print("=" * 50)
print("2025 DOWNLOAD COMPLETE")
print("=" * 50)