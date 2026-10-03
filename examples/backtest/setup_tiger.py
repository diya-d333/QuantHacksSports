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
            CREATE TABLE IF NOT EXISTS market_data.bars_1h (
                time TIMESTAMPTZ NOT NULL,
                dataset TEXT NOT NULL,
                publisher_id INTEGER NOT NULL,
                instrument_id BIGINT NOT NULL,
                symbol TEXT,
                open NUMERIC NOT NULL,
                high NUMERIC NOT NULL,
                low NUMERIC NOT NULL,
                close NUMERIC NOT NULL,
                volume NUMERIC(20, 0) NOT NULL,
                PRIMARY KEY (
                    time, dataset, publisher_id, instrument_id
                )
            );
        """)

        cursor.execute("""
            SELECT create_hypertable(
                'market_data.bars_1h',
                by_range('time'),
                if_not_exists => TRUE
            );
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS bars_1h_symbol_time_idx
            ON market_data.bars_1h (symbol, time DESC);
        """)

print("Hourly market-data table is ready!")