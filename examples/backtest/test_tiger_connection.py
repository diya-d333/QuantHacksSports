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
        SELECT extversion
        FROM pg_extension
        WHERE extname = 'timescaledb';
    """)

        result = cursor.fetchone()

if result:
    print(f"TimescaleDB is ready! Version: {result[0]}")
else:
    print("TimescaleDB extension is not enabled.")