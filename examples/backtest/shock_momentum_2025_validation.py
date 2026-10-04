import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# SETUP
# --------------------------------------------------

BACKTEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKTEST_DIR))

from load_tiger_data import load_market_data


# --------------------------------------------------
# FROZEN STRATEGY SETTINGS
# --------------------------------------------------

START_DATE = "2025-01-01T00:00:00Z"
END_DATE = "2026-01-01T00:00:00Z"

LOOKBACK = 20
SHOCK_ZSCORE = 2.0

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = 2 * (
    FEE_PER_SIDE + SLIPPAGE_PER_SIDE
)


# --------------------------------------------------
# IMPORTANT
# --------------------------------------------------
#
# FROZEN RULES:
#
# BTC:
#   Positive shock >= +2 sigma
#   -> LONG
#
# ETH:
#   Negative shock <= -2 sigma
#   -> SHORT
#
# Signal is calculated at the close of the shock bar.
#
# Entry:
#   Next available 4-hour bar OPEN
#
# Exit:
#   Following 4-hour bar OPEN
#
# No thresholds or parameters should be changed
# after viewing these 2025 results.
# --------------------------------------------------


def prepare_asset(asset):

    df = load_market_data(
        asset,
        START_DATE,
        END_DATE
    ).copy()

    df = df.sort_index()


    # --------------------------------------------------
    # CURRENT 4-HOUR RETURN
    # --------------------------------------------------

    df["return_1"] = (
        df["close"].pct_change()
    )


    # --------------------------------------------------
    # HISTORICAL DISTRIBUTION
    # --------------------------------------------------
    #
    # shift(1) ensures the current return does not
    # influence its own shock threshold.
    # --------------------------------------------------

    df["rolling_mean"] = (
        df["return_1"]
        .shift(1)
        .rolling(LOOKBACK)
        .mean()
    )

    df["rolling_std"] = (
        df["return_1"]
        .shift(1)
        .rolling(LOOKBACK)
        .std(ddof=1)
    )


    # --------------------------------------------------
    # SHOCK Z-SCORE
    # --------------------------------------------------

    df["shock_zscore"] = (
        (
            df["return_1"]
            - df["rolling_mean"]
        )
        / df["rolling_std"]
    )


    # --------------------------------------------------
    # FROZEN SIGNAL
    # --------------------------------------------------

    if asset == "BTC":

        # Long after unusually strong BTC move

        df["signal"] = (
            df["shock_zscore"]
            >= SHOCK_ZSCORE
        )

        df["direction"] = 1


    elif asset == "ETH":

        # Short after unusually negative ETH move

        df["signal"] = (
            df["shock_zscore"]
            <= -SHOCK_ZSCORE
        )

        df["direction"] = -1


    else:

        raise ValueError(
            f"Unsupported asset: {asset}"
        )


    # --------------------------------------------------
    # NEXT-OPEN EXECUTION
    # --------------------------------------------------

    df["entry_time"] = (
        df.index.to_series().shift(-1)
    )

    df["exit_time"] = (
        df.index.to_series().shift(-2)
    )

    df["entry_open"] = (
        df["open"].shift(-1)
    )

    df["exit_open"] = (
        df["open"].shift(-2)
    )


    # --------------------------------------------------
    # REQUIRE CONSECUTIVE 4-HOUR BARS
    # --------------------------------------------------

    df["valid_execution"] = (
        (
            df["entry_time"]
            - df.index.to_series()
            == pd.Timedelta(hours=4)
        )
        &
        (
            df["exit_time"]
            - df["entry_time"]
            == pd.Timedelta(hours=4)
        )
    )


    # --------------------------------------------------
    # RAW FUTURE PRICE RETURN
    # --------------------------------------------------

    df["future_return"] = (
        df["exit_open"]
        / df["entry_open"]
        - 1
    )


    # --------------------------------------------------
    # STRATEGY RETURN
    # --------------------------------------------------
    #
    # BTC direction = +1
    # ETH direction = -1
    # --------------------------------------------------

    df["gross_return"] = (
        df["direction"]
        * df["future_return"]
    )

    df["net_return"] = (
        df["gross_return"]
        - ROUND_TRIP_COST
    )


    # --------------------------------------------------
    # VALID TRADES
    # --------------------------------------------------

    trades = df[
        df["signal"]
        & df["valid_execution"]
        & df["future_return"].notna()
    ].copy()

    trades["asset"] = asset

    return df, trades


# --------------------------------------------------
# SUMMARY FUNCTION
# --------------------------------------------------

def print_summary(
    asset,
    df,
    trades
):

    print()
    print("=" * 70)
    print(asset)
    print("=" * 70)

    print(
        f"2025 bars: {len(df)}"
    )

    print(
        f"Frozen signals: {int(df['signal'].sum())}"
    )

    print(
        f"Valid executed events: {len(trades)}"
    )


    if trades.empty:

        print("No valid trades.")

        return


    avg_gross = (
        trades["gross_return"].mean()
    )

    avg_net = (
        trades["net_return"].mean()
    )

    median_net = (
        trades["net_return"].median()
    )

    continuation_rate = (
        (
            trades["gross_return"] > 0
        ).mean()
    )

    profitable_rate = (
        (
            trades["net_return"] > 0
        ).mean()
    )

    cumulative_net = (
        (1 + trades["net_return"])
        .prod()
        - 1
    )


    print()

    print(
        "Average gross return: "
        f"{avg_gross * 100:.3f}%"
    )

    print(
        "Average net return:   "
        f"{avg_net * 100:.3f}%"
    )

    print(
        "Median net return:    "
        f"{median_net * 100:.3f}%"
    )

    print(
        "Directional success:  "
        f"{continuation_rate * 100:.1f}%"
    )

    print(
        "Net profitable rate:  "
        f"{profitable_rate * 100:.1f}%"
    )

    print(
        "Compounded trade return: "
        f"{cumulative_net * 100:.3f}%"
    )


# --------------------------------------------------
# RUN FROZEN 2025 VALIDATION
# --------------------------------------------------

print()
print("=" * 70)
print("FROZEN SHOCK-MOMENTUM STRATEGY")
print("2025 VALIDATION")
print("=" * 70)

print()
print("Rules:")
print("BTC: LONG after >= +2 sigma shock")
print("ETH: SHORT after <= -2 sigma shock")
print("Lookback: 20 bars")
print("Holding period: 4 hours")
print("Entry: next bar open")
print("Exit: following bar open")

print(
    "Round-trip illustrative cost: "
    f"{ROUND_TRIP_COST * 100:.2f}%"
)


btc_df, btc_trades = (
    prepare_asset("BTC")
)

eth_df, eth_trades = (
    prepare_asset("ETH")
)


print_summary(
    "BTC POSITIVE SHOCK -> LONG",
    btc_df,
    btc_trades
)

print_summary(
    "ETH NEGATIVE SHOCK -> SHORT",
    eth_df,
    eth_trades
)


# --------------------------------------------------
# COMBINED EVENT-LEVEL RESULTS
# --------------------------------------------------

combined = pd.concat(
    [
        btc_trades,
        eth_trades
    ]
).sort_index()


print()
print("=" * 70)
print("COMBINED FROZEN STRATEGY")
print("=" * 70)


if combined.empty:

    print("No trades.")

else:

    print(
        f"Total events: {len(combined)}"
    )

    print(
        "Average gross return: "
        f"{combined['gross_return'].mean() * 100:.3f}%"
    )

    print(
        "Average net return:   "
        f"{combined['net_return'].mean() * 100:.3f}%"
    )

    print(
        "Median net return:    "
        f"{combined['net_return'].median() * 100:.3f}%"
    )

    print(
        "Directional success:  "
        f"{(combined['gross_return'] > 0).mean() * 100:.1f}%"
    )

    print(
        "Net profitable rate:  "
        f"{(combined['net_return'] > 0).mean() * 100:.1f}%"
    )

    cumulative = (
        (1 + combined["net_return"])
        .prod()
        - 1
    )

    print(
        "Compounded event return: "
        f"{cumulative * 100:.3f}%"
    )


    # --------------------------------------------------
    # QUARTERLY CHECK
    # --------------------------------------------------
    #
    # We are NOT using quarters to change the strategy.
    # This only tells us whether the 2025 result comes
    # from one isolated part of the year.
    # --------------------------------------------------

    combined["quarter"] = (
        combined.index.quarter
    )

    print()
    print("2025 quarter-by-quarter:")
    print("-" * 70)


    for quarter in [
        1,
        2,
        3,
        4
    ]:

        q = combined[
            combined["quarter"]
            == quarter
        ]

        if q.empty:

            print(
                f"Q{quarter}: 0 events"
            )

            continue


        print(
            f"Q{quarter}: "
            f"{len(q)} events | "
            f"avg net "
            f"{q['net_return'].mean() * 100:.3f}% | "
            f"median "
            f"{q['net_return'].median() * 100:.3f}% | "
            f"profitable "
            f"{(q['net_return'] > 0).mean() * 100:.1f}%"
        )


# --------------------------------------------------
# VALIDATION REMINDER
# --------------------------------------------------

print()
print("=" * 70)

print(
    "FROZEN 2025 VALIDATION COMPLETE."
)

print(
    "Do not change the 20-bar lookback, 2-sigma threshold, "
    "asset/direction rules, or holding period based on this result."
)

print(
    "2026 remains untouched final OOS."
)

print("=" * 70)
