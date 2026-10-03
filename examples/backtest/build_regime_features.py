import numpy as np
import pandas as pd

from load_tiger_data import load_market_data


FEATURE_COLUMNS = [
    "return_1",
    "return_6",
    "return_30",
    "drop_zscore",
    "volatility_30",
    "sma_distance_20",
    "sma_20_50_spread",
    "bb_width",
    "bb_position",
    "volume_zscore",
    "fear_greed_score",
]


def build_features(asset):
    # Load 2021–2025. Leave 2026 untouched.
    df = load_market_data(
        asset,
        "2021-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    ).copy()

    if df.empty:
        raise ValueError(f"No data found for {asset}")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError(
            f"{asset}: duplicate timestamps. "
            "Resolve these before calculating features."
        )

    if not (df.index == df.index.floor("4h")).all():
        raise ValueError(
            f"{asset}: timestamps do not align to UTC 4-hour boundaries."
        )

    for column in ["close", "volume", "fear_greed_score"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if (df["close"] <= 0).any():
        raise ValueError(f"{asset}: found a nonpositive closing price")

    original_times = df.index.copy()

    # Keep actual observed candles.
    # Historical rolling windows count observed candles, not calendar slots.
    df.index.name = "time"

    close = df["close"]
    volume = df["volume"]

    time_series = pd.Series(df.index, index=df.index)

    # Only treat consecutive closes as a 4-hour return
    # when their timestamps are exactly 4 hours apart.
    previous_gap = time_series.diff()
    exact_previous_4h = previous_gap.eq(pd.Timedelta(hours=4))

    df["return_1"] = (
    close / close.shift(1) - 1
    ).where(exact_previous_4h)

    # These are now 6- and 30-observed-candle returns.
    # They do not necessarily represent 24 hours or 5 calendar days.
    df["return_6"] = close / close.shift(6) - 1
    df["return_30"] = close / close.shift(30) - 1

    returns = df["return_1"]

    # For rolling volatility/drop statistics, use valid historical
    # 4-hour returns only. Gap-spanning returns remain excluded.
    valid_returns = returns.dropna()

    past_mean = (
    valid_returns.rolling(30, min_periods=30)
    .mean()
    .shift(1)
    )

    past_std = (
    valid_returns.rolling(30, min_periods=30)
    .std(ddof=1)
    .shift(1)
    .replace(0, np.nan)
    )

    df["drop_zscore"] = (
    (valid_returns - past_mean) / past_std
    ).reindex(df.index)

    df["volatility_30"] = (
    valid_returns.rolling(30, min_periods=30)
    .std(ddof=1)
    .reindex(df.index)
    )

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()

    df["sma_distance_20"] = close / sma20 - 1
    df["sma_20_50_spread"] = sma20 / sma50 - 1

    # Bollinger Bands: 20 candles, plus/minus 2 standard deviations.
    price_std20 = close.rolling(
        window=20,
        min_periods=20,
    ).std(ddof=1)

    lower_band = sma20 - 2 * price_std20
    upper_band = sma20 + 2 * price_std20
    band_width = upper_band - lower_band

    df["bb_width"] = band_width / sma20

    df["bb_position"] = (
        close - lower_band
    ) / band_width.replace(0, np.nan)
 
    # Compare current volume with the previous 30 candles.
    previous_volume = volume.shift(1).rolling(
        window=30,
        min_periods=30,
    )

    df["volume_zscore"] = (
        volume - previous_volume.mean()
    ) / previous_volume.std(ddof=1).replace(0, np.nan)

    df["asset"] = asset

    # Your source timestamp represents the candle START.
    # The closing price becomes available at the candle END.
    df["bar_end_time"] = df.index + pd.Timedelta(hours=4)

    signal_year = df["bar_end_time"].dt.year

    df["dataset_role"] = np.select(
        [
            signal_year.between(2021, 2024),
            signal_year.eq(2025),
        ],
        [
            "REGIME_TRAIN",
            "REGIME_VALIDATION",
        ],
        default="EXCLUDED",
    )

    # Evaluation target: return over the following 4 hours.
    # NEVER include this column in the trading signal or model inputs.
    
    next_gap = time_series.shift(-1) - time_series
    exact_next_4h = next_gap.eq(pd.Timedelta(hours=4))

    df["forward_return_1"] = (
        close.shift(-1) / close - 1
    ).where(exact_next_4h)

    # Do not allow targets to cross training/validation boundaries.
    same_dataset = df["dataset_role"].eq(
        df["dataset_role"].shift(-1)
    )

    df["forward_return_1"] = df["forward_return_1"].where(
        same_dataset
    )

    # Check the sentiment availability field from your combined view.
    availability_column = "sentiment_assumed_available_at"

    if availability_column not in df.columns:
        raise ValueError(
            "Missing sentiment_assumed_available_at. "
            "Check your combined market/sentiment view."
        )

    df[availability_column] = pd.to_datetime(
        df[availability_column],
        utc=True,
    )

    invalid_sentiment = df["fear_greed_score"].notna() & (
        df[availability_column].isna()
        | (df[availability_column] > df["bar_end_time"])
    )

    if invalid_sentiment.any():
        raise ValueError(
            f"{asset}: sentiment is not available at signal time."
        )

    numeric_columns = FEATURE_COLUMNS + ["forward_return_1"]

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Keep incomplete rows, but explicitly mark their readiness.
    df["features_ready"] = df[FEATURE_COLUMNS].notna().all(axis=1)
    df["target_ready"] = df["forward_return_1"].notna()

    output_columns = [
        "asset",
        "bar_end_time",
        "close",
        "volume",
        availability_column,
        *FEATURE_COLUMNS,
        "forward_return_1",
        "dataset_role",
        "features_ready",
        "target_ready",
    ]

    # Export actual source candles, not inserted missing slots.
    output = (
        df.loc[original_times, output_columns]
        .rename_axis("time")
        .reset_index()
    )

    output = output[
        output["dataset_role"] != "EXCLUDED"
    ].copy()

    print(
        f"{asset}: {len(output):,} output rows; "
        f"{int(output['features_ready'].sum()):,} feature-ready rows; "
        f"{int(output['target_ready'].sum()):,} valid 4-hour targets"
    )

    return output


def main():
    btc = build_features("BTC")
    eth = build_features("ETH")

    combined = pd.concat(
        [btc, eth],
        ignore_index=True,
    ).sort_values(["asset", "time"])

    # Uppercase column names for the Snowflake import.
    combined.columns = combined.columns.str.upper()

    filename = "regime_features_4h.csv"
    combined.to_csv(filename, index=False)

    print("\nDataset summary:")
    print(
        combined.groupby(["ASSET", "DATASET_ROLE"])[
            ["FEATURES_READY", "TARGET_READY"]
        ].agg(["size", "sum"])
    )

    print(f"\nSaved: {filename}")
    print(
        "FORWARD_RETURN_1 is an evaluation target, "
        "not a model input."
    )


if __name__ == "__main__":
    main()