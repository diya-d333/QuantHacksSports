import os
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Load the existing .env file from examples/backtest
load_dotenv(
    ROOT / "examples" / "backtest" / ".env"
)


# ============================================================
# SETTINGS
# ============================================================

# Same symbols used for the earlier BTC/ETH data
ASSETS = {
    "BTC": "BTC.v.0",
    "ETH": "ETH.v.0",
}

DATASET = "GLBX.MDP3"

START_DATE = "2026-01-01"

# Download through the latest available date.
# We cannot request future data.
END_DATE = "2026-10-01"


# ============================================================
# SETUP DATABENTO
# ============================================================

api_key = os.environ.get(
    "DATABENTO_API_KEY"
)

if not api_key:
    raise ValueError(
        "DATABENTO_API_KEY was not found in "
        "examples/backtest/.env"
    )

client = db.Historical(
    api_key
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

output_dir = (
    ROOT / "data" / "raw_data"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DOWNLOAD BTC AND ETH
# ============================================================

for asset, symbol in ASSETS.items():

    print()
    print("=" * 60)

    print(
        f"Downloading {asset} data for 2026"
    )

    print(
        f"Date range: {START_DATE} -> {END_DATE}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # DOWNLOAD 1-HOUR DATA
    # --------------------------------------------------------

    data = client.timeseries.get_range(
        dataset=DATASET,
        schema="ohlcv-1h",
        symbols=symbol,
        stype_in="continuous",
        start=START_DATE,
        end=END_DATE,
    )

    df = data.to_df()

    print(
        f"Downloaded {len(df)} hourly candles"
    )

    if df.empty:

        print(
            f"WARNING: No data returned for {asset}"
        )

        continue


    # --------------------------------------------------------
    # MAKE TIMESTAMPS UTC
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index,
        utc=True
    )


    # --------------------------------------------------------
    # CONVERT 1-HOUR CANDLES TO 4-HOUR CANDLES
    # --------------------------------------------------------

    df_4h = df.resample(
        "4h"
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )


    # --------------------------------------------------------
    # REMOVE EMPTY 4-HOUR PERIODS
    # --------------------------------------------------------

    df_4h = df_4h.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )


    print(
        f"Created {len(df_4h)} four-hour candles"
    )


    # --------------------------------------------------------
    # SAVE AS SEPARATE 2026 FILE
    # --------------------------------------------------------

    output_file = (
        output_dir
        / f"{asset}_4h_2026.parquet"
    )

    df_4h.to_parquet(
        output_file
    )


    # --------------------------------------------------------
    # VERIFY SAVED DATA
    # --------------------------------------------------------

    print(
        f"Saved to: {output_file}"
    )

    print(
        f"First timestamp: "
        f"{df_4h.index.min()}"
    )

    print(
        f"Last timestamp: "
        f"{df_4h.index.max()}"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("2026 DOWNLOAD COMPLETE")
print("=" * 60)
