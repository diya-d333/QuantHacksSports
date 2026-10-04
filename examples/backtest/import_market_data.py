import os
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / "data" / "raw_data"

load_dotenv(Path(__file__).resolve().parent / ".env")

files = [
    ("BTC", "BTC_4h.parquet"),
    ("ETH", "ETH_4h.parquet"),
    ("BTC", "BTC_4h_2025.parquet"),
    ("ETH", "ETH_4h_2025.parquet"),
    ("SOL", "SOL_4h.parquet"),
    ("XRP", "XRP_4h.parquet"),
    ("BTC", "BTC_4h_2026.parquet"),
    ("ETH", "ETH_4h_2026.parquet"),
]

# Check all files exist before inserting anything.
for asset, filename in files:
    if not (DATA_FOLDER / filename).exists():
        raise FileNotFoundError(DATA_FOLDER / filename)

insert_sql = """
    INSERT INTO market_data.imported_bars_4h (
        time, asset, dataset,
        open, high, low, close, volume, source_file
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (time, asset, dataset) DO NOTHING;
"""

with psycopg.connect(
    os.environ["TIMESCALE_SERVICE_URL"],
    connect_timeout=10,
) as connection:
    with connection.cursor() as cursor:
        for asset, filename in files:
            df = pd.read_parquet(DATA_FOLDER / filename)

            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{filename}: expected a datetime index")
            if df.index.tz is None:
                raise ValueError(f"{filename}: timestamps need a timezone")

            df.index = df.index.tz_convert("UTC")
            df = df.sort_index()

            columns = ["open", "high", "low", "close", "volume"]
            if df.index.hasnans or df.index.has_duplicates:
                raise ValueError(f"{filename}: invalid or duplicate timestamps")
            if df[columns].isna().any().any():
                raise ValueError(f"{filename}: missing price or volume values")

            rows = []
            for timestamp, bar in df[columns].iterrows():
                values = [Decimal(str(bar[column])) for column in columns]

                if not all(value.is_finite() for value in values):
                    raise ValueError(f"{filename}: non-finite values")
                if values[4] < 0 or values[4] != values[4].to_integral_value():
                    raise ValueError(f"{filename}: invalid volume")

                rows.append((
                    timestamp.to_pydatetime(),
                    asset,
                    "GLBX.MDP3",
                    *values,
                    filename,
                ))

            cursor.executemany(insert_sql, rows)
            print(f"{filename}: processed {len(rows)} rows")

        cursor.execute("""
            SELECT asset, COUNT(*), MIN(time), MAX(time)
            FROM market_data.imported_bars_4h
            WHERE dataset = 'GLBX.MDP3'
            GROUP BY asset
            ORDER BY asset;
        """)
        summary = cursor.fetchall()

print("\nImport committed! Stored totals:")
for asset, count, first_time, last_time in summary:
    print(f"{asset}: {count} rows | {first_time} to {last_time}")