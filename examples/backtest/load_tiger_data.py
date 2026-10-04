import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv


def load_market_data(asset, start, end):
    """Load bars within [start, end), ordered by bar start time."""
    load_dotenv(Path(__file__).resolve().parent / ".env")

    with psycopg.connect(
        os.environ["TIMESCALE_SERVICE_URL"],
        connect_timeout=10,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT *
                FROM market_data.market_sentiment_4h
                WHERE asset = %s
                  AND dataset = 'GLBX.MDP3'
                  AND time >= %s
                  AND time < %s
                ORDER BY time;
            """, (asset, start, end))

            columns = [column.name for column in cursor.description]
            df = pd.DataFrame(cursor.fetchall(), columns=columns)

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])

    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")


if __name__ == "__main__":
    data = load_market_data(
        "SOL",
        "2025-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    )
    print(f"Loaded {len(data)} BTC bars")
    print(data[["close", "fear_greed_score"]].head())