import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data/raw_data/fear_greed_daily.parquet"

load_dotenv(Path(__file__).resolve().parent / ".env")

df = pd.read_parquet(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp")

required = [
    "timestamp",
    "fear_greed_score",
    "fear_greed_classification",
]

if df[required].isna().any().any():
    raise ValueError("Sentiment data contains missing values.")

if df["timestamp"].duplicated().any():
    raise ValueError("Sentiment data contains duplicate timestamps.")

scores = pd.to_numeric(df["fear_greed_score"], errors="raise")
if not (scores.between(0, 100) & (scores % 1 == 0)).all():
    raise ValueError("Sentiment scores must be integers from 0 to 100.")

df["fear_greed_score"] = scores.astype(int)

rows = [
    (
        row.timestamp.to_pydatetime(),
        int(row.fear_greed_score),
        str(row.fear_greed_classification),
        "alternative.me",
        DATA_FILE.name,
    )
    for row in df.itertuples(index=False)
]

with psycopg.connect(
    os.environ["TIMESCALE_SERVICE_URL"],
    connect_timeout=10,
) as connection:
    with connection.cursor() as cursor:
        cursor.executemany("""
            INSERT INTO market_data.fear_greed_daily (
                time,
                fear_greed_score,
                fear_greed_classification,
                source,
                source_file
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (time) DO NOTHING;
        """, rows)

        cursor.execute("""
            SELECT COUNT(*), MIN(time), MAX(time)
            FROM market_data.fear_greed_daily;
        """)
        count, first_time, last_time = cursor.fetchone()

print("Sentiment import committed!")
print(f"Stored rows: {count}")
print(f"Date range: {first_time} to {last_time}")