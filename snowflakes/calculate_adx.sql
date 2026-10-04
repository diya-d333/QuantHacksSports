USE WAREHOUSE GQH_WH;
USE DATABASE GQH_CRYPTO;
USE SCHEMA ML_DATA;

CREATE OR REPLACE PROCEDURE GQH_CRYPTO.ML_DATA.CALCULATE_ADX()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'main'
EXECUTE AS CALLER
AS
$$
import math

from snowflake.snowpark.types import (
    StructType, StructField, StringType, DoubleType,
    TimestampType, TimestampTimeZone
)


def main(session):
    bars = session.sql("""
        SELECT TIME, BAR_END_TIME, ASSET, DATASET, HIGH, LOW, CLOSE
        FROM GQH_CRYPTO.ML_DATA.MARKET_BARS_4H
        WHERE ASSET IN ('BTC', 'ETH')
          AND DATASET = 'GLBX.MDP3'
        ORDER BY ASSET, DATASET, TIME
    """).collect()

    if not bars:
        raise ValueError("No market bars found.")

    groups = {}
    seen = set()

    for row in bars:
        key = (row["ASSET"], row["DATASET"], row["TIME"])
        if key in seen:
            raise ValueError("Duplicate market bars.")
        seen.add(key)

        prices = [row["HIGH"], row["LOW"], row["CLOSE"]]
        if any(v is None or not math.isfinite(v) for v in prices):
            raise ValueError("Missing or invalid prices.")

        if not 0 < row["LOW"] <= row["CLOSE"] <= row["HIGH"]:
            raise ValueError("Invalid high, low, or close.")

        groups.setdefault(
            (row["ASSET"], row["DATASET"]), []
        ).append(row)

    output = []
    period = 14

    for (asset, dataset), rows in groups.items():
        tr_sum = plus_sum = minus_sum = 0.0
        dx_seed = []
        adx = None

        for i, row in enumerate(rows):
            plus_di = minus_di = None

            if i > 0:
                previous = rows[i - 1]
                up = row["HIGH"] - previous["HIGH"]
                down = previous["LOW"] - row["LOW"]

                plus_dm = up if up > down and up > 0 else 0.0
                minus_dm = down if down > up and down > 0 else 0.0

                true_range = max(
                    row["HIGH"] - row["LOW"],
                    abs(row["HIGH"] - previous["CLOSE"]),
                    abs(row["LOW"] - previous["CLOSE"]),
                )

                if i <= period:
                    tr_sum += true_range
                    plus_sum += plus_dm
                    minus_sum += minus_dm
                else:
                    tr_sum = tr_sum - tr_sum / period + true_range
                    plus_sum = plus_sum - plus_sum / period + plus_dm
                    minus_sum = minus_sum - minus_sum / period + minus_dm

                if i >= period:
                    plus_di = 100 * plus_sum / tr_sum if tr_sum else 0.0
                    minus_di = 100 * minus_sum / tr_sum if tr_sum else 0.0

                    denominator = plus_di + minus_di
                    dx = (
                        100 * abs(plus_di - minus_di) / denominator
                        if denominator else 0.0
                    )

                    if adx is None:
                        dx_seed.append(dx)
                        if len(dx_seed) == period:
                            adx = sum(dx_seed) / period
                    else:
                        adx = (adx * (period - 1) + dx) / period

            output.append((
                row["TIME"],
                row["BAR_END_TIME"],
                asset,
                dataset,
                plus_di,
                minus_di,
                adx,
            ))

    schema = StructType([
        StructField("TIME", TimestampType(TimestampTimeZone.TZ)),
        StructField("BAR_END_TIME", TimestampType(TimestampTimeZone.TZ)),
        StructField("ASSET", StringType()),
        StructField("DATASET", StringType()),
        StructField("PLUS_DI_14", DoubleType()),
        StructField("MINUS_DI_14", DoubleType()),
        StructField("ADX_14", DoubleType()),
    ])

    session.create_dataframe(output, schema=schema).write.mode(
        "overwrite"
    ).save_as_table("GQH_CRYPTO.ML_DATA.MARKET_ADX_4H")

    ready = sum(row[-1] is not None for row in output)
    return f"Saved {len(output)} bars; {ready} have ADX."
$$;