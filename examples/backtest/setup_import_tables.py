import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

with psycopg.connect(
    os.environ["TIMESCALE_SERVICE_URL"],
    connect_timeout=10,
) as connection:
    with connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS market_data;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data.imported_bars_4h (
                time TIMESTAMPTZ NOT NULL,
                asset TEXT NOT NULL,
                dataset TEXT NOT NULL,
                open NUMERIC NOT NULL,
                high NUMERIC NOT NULL,
                low NUMERIC NOT NULL,
                close NUMERIC NOT NULL,
                volume NUMERIC(20, 0) NOT NULL,
                source_file TEXT NOT NULL,
                PRIMARY KEY (time, asset, dataset)
            );
        """)

        cursor.execute("""
            SELECT create_hypertable(
                'market_data.imported_bars_4h',
                by_range('time'),
                if_not_exists => TRUE
            );
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS imported_bars_4h_asset_time_idx
            ON market_data.imported_bars_4h (asset, time DESC);
        """)

print("Four-hour import table is ready!")