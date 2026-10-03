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
        cursor.execute("""
            CREATE OR REPLACE VIEW market_data.bars_4h AS
            SELECT
                time_bucket(INTERVAL '4 hours', time) AS time,
                dataset,
                publisher_id,
                instrument_id,
                last(symbol, time) AS symbol,
                first(open, time) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                last(close, time) AS close,
                SUM(volume) AS volume,
                COUNT(*) AS hourly_bar_count
            FROM market_data.bars_1h
            GROUP BY
                time_bucket(INTERVAL '4 hours', time),
                dataset,
                publisher_id,
                instrument_id;
        """)

print("Four-hour view is ready!")