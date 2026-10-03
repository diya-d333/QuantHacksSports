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
            CREATE OR REPLACE VIEW market_data.market_sentiment_4h AS
            SELECT
                b.time,
                b.time + INTERVAL '4 hours' AS bar_end_time,
                b.asset,
                b.dataset,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                s.time AS sentiment_observation_time,
                s.time + INTERVAL '24 hours'
                    AS sentiment_assumed_available_at,
                s.fear_greed_score,
                s.fear_greed_classification
            FROM market_data.imported_bars_4h AS b
            LEFT JOIN LATERAL (
                SELECT
                    time,
                    fear_greed_score,
                    fear_greed_classification
                FROM market_data.fear_greed_daily
                WHERE time <= b.time - INTERVAL '24 hours'
                ORDER BY time DESC
                LIMIT 1
            ) AS s ON TRUE;
        """)

        cursor.execute("""
            SELECT time, asset, close, fear_greed_score
            FROM market_data.market_sentiment_4h
            ORDER BY time, asset
            LIMIT 5;
        """)
        sample = cursor.fetchall()

print("Combined market/sentiment view is ready!")
for row in sample:
    print(row)