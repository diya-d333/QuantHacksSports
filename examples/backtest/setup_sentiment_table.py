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
            CREATE TABLE IF NOT EXISTS market_data.fear_greed_daily (
                time TIMESTAMPTZ PRIMARY KEY,
                fear_greed_score INTEGER NOT NULL
                    CHECK (fear_greed_score BETWEEN 0 AND 100),
                fear_greed_classification TEXT NOT NULL,
                source TEXT NOT NULL,
                source_file TEXT NOT NULL
            );
        """)

print("Daily sentiment table is ready!")