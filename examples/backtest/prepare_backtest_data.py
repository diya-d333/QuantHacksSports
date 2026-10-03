from pathlib import Path

import numpy as np
import pandas as pd

from load_tiger_data import load_market_data


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "backtest_ready"


def main():
    print("Preparing 2025 Backtrader inputs...", flush=True)

    features = pd.read_csv(ROOT / "regime_features_4h.csv")
    predictions = pd.read_csv(
        ROOT / "regime_model_output" / "regime_predictions_4h.csv"
    )

    for frame in [features, predictions]:
        frame["TIME"] = pd.to_datetime(frame["TIME"], utc=True)
        frame["BAR_END_TIME"] = pd.to_datetime(
            frame["BAR_END_TIME"], utc=True
        )

    features = features.loc[
        features["DATASET_ROLE"].eq("REGIME_VALIDATION")
    ].copy()

    predictions = predictions.loc[
        predictions["DATASET_ROLE"].eq("REGIME_VALIDATION")
    ].copy()

    joined = features.merge(
        predictions[
            [
                "ASSET",
                "TIME",
                "BAR_END_TIME",
                "DATASET_ROLE",
                "REGIME",
                "RANGE_PROBABILITY",
            ]
        ],
        on=["ASSET", "TIME", "BAR_END_TIME", "DATASET_ROLE"],
        how="left",
        validate="one_to_one",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for asset in ["BTC", "ETH"]:
        print(f"Loading {asset} OHLCV...", flush=True)

        bars = load_market_data(
            asset,
            "2025-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        )

        if bars.empty:
            raise ValueError(f"No 2025 candles for {asset}")

        bars = (
            bars[["open", "high", "low", "close", "volume"]]
            .rename_axis("TIME")
            .reset_index()
        )

        bars.columns = bars.columns.str.upper()
        bars["TIME"] = pd.to_datetime(bars["TIME"], utc=True)

        if bars["TIME"].duplicated().any():
            raise ValueError(f"{asset}: duplicate OHLCV timestamps")

        asset_features = joined.loc[
            joined["ASSET"].eq(asset)
        ].copy()

        frame = bars.merge(
            asset_features[
                [
                    "TIME",
                    "BAR_END_TIME",
                    "CLOSE",
                    "FEATURES_READY",
                    "RETURN_1",
                    "DROP_ZSCORE",
                    "FEAR_GREED_SCORE",
                    "REGIME",
                    "RANGE_PROBABILITY",
                ]
            ].rename(columns={"CLOSE": "FEATURE_CLOSE"}),
            on="TIME",
            how="left",
            validate="one_to_one",
        ).sort_values("TIME")

        matched = frame["FEATURE_CLOSE"].notna()

        if not np.isclose(
            frame.loc[matched, "CLOSE"],
            frame.loc[matched, "FEATURE_CLOSE"],
            rtol=1e-10,
            atol=1e-8,
        ).all():
            raise ValueError(
                f"{asset}: OHLCV prices differ from feature prices"
            )

        numeric = frame[
            ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        ]

        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{asset}: missing or invalid OHLCV")

        if (
            (frame["LOW"] > frame[["OPEN", "CLOSE"]].min(axis=1))
            | (frame["HIGH"] < frame[["OPEN", "CLOSE"]].max(axis=1))
            | (frame["LOW"] > frame["HIGH"])
            | (frame[["OPEN", "HIGH", "LOW", "CLOSE"]] <= 0).any(axis=1)
            | (frame["VOLUME"] < 0)
        ).any():
            raise ValueError(f"{asset}: invalid candle values")

        ready = (
            frame["FEATURES_READY"].eq(True)
            & frame["REGIME"].notna()
        )

        drop = (
            ready
            & frame["RETURN_1"].lt(0)
            & frame["DROP_ZSCORE"].le(-2)
        )

        fear = frame["FEAR_GREED_SCORE"].le(25)
        range_filter = frame["REGIME"].eq("RANGE_CANDIDATE")

        frame["SIGNAL_DROP_ONLY"] = drop.astype(int)
        frame["SIGNAL_DROP_FEAR"] = (drop & fear).astype(int)
        frame["SIGNAL_DROP_RANGE"] = (
            drop & range_filter
        ).astype(int)
        frame["SIGNAL_FULL"] = (
            drop & fear & range_filter
        ).astype(int)

        # Actual bar boundaries; preserve gaps without filling prices.
        frame["BAR_START_TIME"] = frame["TIME"]
        frame["BAR_END_TIME"] = (
            frame["TIME"] + pd.Timedelta(hours=4)
        )

        # Signals are available at the candle END.
        frame["SIGNAL_TIME"] = frame["BAR_END_TIME"]

        columns = [
            "BAR_START_TIME",
            "BAR_END_TIME",
            "SIGNAL_TIME",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOLUME",
            "SIGNAL_DROP_ONLY",
            "SIGNAL_DROP_FEAR",
            "SIGNAL_DROP_RANGE",
            "SIGNAL_FULL",
        ]

        output = OUTPUT_DIR / f"{asset}_2025_backtest.csv"
        frame[columns].to_csv(output, index=False)

        print(f"{asset}: saved {len(frame):,} candles")
        print(
            frame[
                [
                    "SIGNAL_DROP_ONLY",
                    "SIGNAL_DROP_FEAR",
                    "SIGNAL_DROP_RANGE",
                    "SIGNAL_FULL",
                ]
            ].sum().to_string()
        )
        print(f"File: {output}\n")

    print("Finished preparing backtest inputs.")


if __name__ == "__main__":
    main()