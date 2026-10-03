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
            SELECT hypertable_schema, hypertable_name
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'market_data'
              AND hypertable_name = 'bars_1h';
        """)
        print("Hourly hypertable:", cursor.fetchone())

        cursor.execute("SELECT COUNT(*) FROM market_data.bars_1h;")
        print("Hourly rpows:", cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM market_data.bars_4h;")
        print("Four-hour rows:", cursor.fetchone()[0])

print("Database checks completed!")