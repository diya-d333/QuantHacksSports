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
            SELECT COUNT(*)
            FROM market_data.imported_bars_4h;
        """)
        price_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (
                    WHERE fear_greed_score IS NULL
                ),
                COUNT(*) FILTER (
                    WHERE sentiment_assumed_available_at > time
                )
            FROM market_data.market_sentiment_4h;
        """)
        combined_count, missing_sentiment, timing_errors = cursor.fetchone()

print("Price rows:", price_count)
print("Combined rows:", combined_count)
print("Rows missing sentiment:", missing_sentiment)
print("Sentiment timing errors:", timing_errors)

assert price_count == combined_count, "Row counts do not match."
assert missing_sentiment == 0, "Some bars have no sentiment."
assert timing_errors == 0, "Some sentiment violates the assumed delay."

print("All combined-data checks passed!")