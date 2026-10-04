import os
from pathlib import Path

import psycopg
import snowflake.connector
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def main():
    # Read BTC and ETH prices from Tiger Data.
    with psycopg.connect(os.environ["TIMESCALE_SERVICE_URL"]) as tiger:
        with tiger.cursor() as cursor:
            cursor.execute("""
                SELECT
                    time,
                    time + INTERVAL '4 hours',
                    asset,
                    dataset,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM market_data.imported_bars_4h
                WHERE asset IN ('BTC', 'ETH')
                  AND dataset = 'GLBX.MDP3'
                  AND time >= TIMESTAMPTZ '2021-01-01 00:00:00+00'
                  AND time < TIMESTAMPTZ '2026-01-01 00:00:00+00'
                ORDER BY asset, time
            """)
            rows = cursor.fetchall()

    if not rows:
        raise ValueError("No BTC or ETH prices found in Tiger Data.")

    keys = [(row[0], row[2], row[3]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate price rows found in Tiger Data.")

    records = [
        (
            row[0].isoformat(),
            row[1].isoformat(),
            row[2],
            row[3],
            *(float(value) for value in row[4:]),
        )
        for row in rows
    ]

    print(f"Read {len(records):,} bars from Tiger Data.")

    with snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        authenticator="OAUTH_AUTHORIZATION_CODE",
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    ) as sf_connection:
        with sf_connection.cursor() as cursor:
            cursor.execute("""
                CREATE TEMPORARY TABLE BARS_UPLOAD
                LIKE GQH_CRYPTO.ML_DATA.MARKET_BARS_4H
            """)

            insert_sql = """
                INSERT INTO BARS_UPLOAD (
                    TIME, BAR_END_TIME, ASSET, DATASET,
                    OPEN, HIGH, LOW, CLOSE, VOLUME
                )
                VALUES (
                    TO_TIMESTAMP_TZ(%s),
                    TO_TIMESTAMP_TZ(%s),
                    %s, %s, %s, %s, %s, %s, %s
                )
            """

            for start in range(0, len(records), 1000):
                cursor.executemany(
                    insert_sql,
                    records[start:start + 1000],
                )

            # Update matching bars and insert missing bars.
            # Rerunning this script does not append duplicates.
            cursor.execute("""
                MERGE INTO GQH_CRYPTO.ML_DATA.MARKET_BARS_4H AS t
                USING BARS_UPLOAD AS s
                    ON t.TIME = s.TIME
                   AND t.ASSET = s.ASSET
                   AND t.DATASET = s.DATASET
                WHEN MATCHED THEN UPDATE SET
                    t.BAR_END_TIME = s.BAR_END_TIME,
                    t.OPEN = s.OPEN,
                    t.HIGH = s.HIGH,
                    t.LOW = s.LOW,
                    t.CLOSE = s.CLOSE,
                    t.VOLUME = s.VOLUME
                WHEN NOT MATCHED THEN INSERT (
                    TIME, BAR_END_TIME, ASSET, DATASET,
                    OPEN, HIGH, LOW, CLOSE, VOLUME
                ) VALUES (
                    s.TIME, s.BAR_END_TIME, s.ASSET, s.DATASET,
                    s.OPEN, s.HIGH, s.LOW, s.CLOSE, s.VOLUME
                )
            """)

            cursor.execute("""
                SELECT ASSET, COUNT(*), MIN(TIME), MAX(TIME)
                FROM GQH_CRYPTO.ML_DATA.MARKET_BARS_4H
                GROUP BY ASSET
                ORDER BY ASSET
            """)

            for asset, count, first, last in cursor.fetchall():
                print(f"{asset}: {count:,} bars | {first} to {last}")

    print("Snowflake price import complete!")


if __name__ == "__main__":
    main()