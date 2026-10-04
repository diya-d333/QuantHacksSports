from pathlib import Path
import sys

import pandas as pd
import numpy as np


# --------------------------------------------------
# SETUP
# --------------------------------------------------

BACKTEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKTEST_DIR))

from load_tiger_data import load_market_data


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

START_DATE = "2021-01-01T00:00:00Z"
END_DATE = "2026-01-01T00:00:00Z"

LOOKBACK = 20
DROP_THRESHOLD = 2.0

ADX_PERIOD = 14
ADX_THRESHOLD = 25

COMMISSION = 0.001


# --------------------------------------------------
# LOAD ETH
# --------------------------------------------------

print("Loading ETH 2021-2025 data...")

eth = load_market_data(
    "ETH",
    START_DATE,
    END_DATE
)

print(f"Total bars: {len(eth)}")
print(f"Start: {eth.index.min()}")
print(f"End:   {eth.index.max()}")


# --------------------------------------------------
# ADX CALCULATION
# --------------------------------------------------

def calculate_adx(data, period=14):

    high = data["high"]
    low = data["low"]
    close = data["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=data.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=data.index,
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    plus_dm_smoothed = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    minus_dm_smoothed = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    plus_di = 100 * (
        plus_dm_smoothed / atr
    )

    minus_di = 100 * (
        minus_dm_smoothed / atr
    )

    denominator = plus_di + minus_di

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator.replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    return adx


# --------------------------------------------------
# LARGE-DROP EVENTS
# --------------------------------------------------

eth_return = eth["close"].pct_change()

eth_volatility = (
    eth_return
    .rolling(LOOKBACK)
    .std()
)

large_drop = (
    eth_return
    < -DROP_THRESHOLD * eth_volatility
)

eth_adx = calculate_adx(
    eth,
    ADX_PERIOD
)


# --------------------------------------------------
# BUILD EVENT DATASET
# --------------------------------------------------
#
# Signal = large ETH drop at close t
#
# Entry = open t+1
# Exit  = open t+2
#
# LONG return:
#   positive if ETH rebounds
#
# SHORT return:
#   positive if ETH continues falling
#
# --------------------------------------------------

events = []

i = 0

while i < len(eth) - 2:

    if bool(large_drop.iloc[i]):

        signal_time = eth.index[i]

        entry_index = i + 1
        exit_index = i + 2

        entry_price = eth["open"].iloc[
            entry_index
        ]

        exit_price = eth["open"].iloc[
            exit_index
        ]

        long_gross = (
            exit_price / entry_price
        ) - 1

        short_gross = (
            entry_price / exit_price
        ) - 1

        long_net = (
            long_gross
            - (2 * COMMISSION)
        )

        short_net = (
            short_gross
            - (2 * COMMISSION)
        )

        current_adx = eth_adx.iloc[i]

        if current_adx < ADX_THRESHOLD:
            regime = "ADX < 25"
        else:
            regime = "ADX >= 25"

        events.append(
            {
                "Year": signal_time.year,
                "Signal Time": signal_time,

                "ETH Drop":
                    eth_return.iloc[i] * 100,

                "ADX":
                    current_adx,

                "Regime":
                    regime,

                "Long Return":
                    long_net * 100,

                "Short Return":
                    short_net * 100,
            }
        )

        # Skip until this hypothetical
        # 4-hour trade is finished.
        i = exit_index

    else:

        i += 1


events = pd.DataFrame(events)


# --------------------------------------------------
# ALL LARGE DROPS
# --------------------------------------------------

print()
print(
    "========== ALL LARGE-DROP EVENTS =========="
)
print()

print(f"Events: {len(events)}")

print(
    f"Average LONG return: "
    f"{events['Long Return'].mean():.3f}%"
)

print(
    f"Average SHORT return: "
    f"{events['Short Return'].mean():.3f}%"
)

print(
    f"LONG win rate: "
    f"{(events['Long Return'] > 0).mean() * 100:.1f}%"
)

print(
    f"SHORT win rate: "
    f"{(events['Short Return'] > 0).mean() * 100:.1f}%"
)


# --------------------------------------------------
# COMPARE ADX REGIMES
# --------------------------------------------------

regime_results = (
    events
    .groupby(
        "Regime",
        observed=True
    )
    .agg(
        Events=(
            "Short Return",
            "count"
        ),

        Avg_Long=(
            "Long Return",
            "mean"
        ),

        Median_Long=(
            "Long Return",
            "median"
        ),

        Long_Win_Rate=(
            "Long Return",
            lambda x:
                (x > 0).mean() * 100
        ),

        Avg_Short=(
            "Short Return",
            "mean"
        ),

        Median_Short=(
            "Short Return",
            "median"
        ),

        Short_Win_Rate=(
            "Short Return",
            lambda x:
                (x > 0).mean() * 100
        ),
    )
    .reset_index()
)


print()
print(
    "========== ADX REGIME COMPARISON =========="
)
print()

print(
    regime_results.to_string(
        index=False,
        formatters={
            "Avg_Long":
                lambda x: f"{x:.3f}%",

            "Median_Long":
                lambda x: f"{x:.3f}%",

            "Long_Win_Rate":
                lambda x: f"{x:.1f}%",

            "Avg_Short":
                lambda x: f"{x:.3f}%",

            "Median_Short":
                lambda x: f"{x:.3f}%",

            "Short_Win_Rate":
                lambda x: f"{x:.1f}%",
        },
    )
)


# --------------------------------------------------
# ADX QUARTILES
# --------------------------------------------------
#
# Diagnostic only.
# This does NOT select an ADX threshold.
# --------------------------------------------------

events["ADX Quartile"] = pd.qcut(
    events["ADX"],
    4,
    duplicates="drop"
)


adx_quartiles = (
    events
    .groupby(
        "ADX Quartile",
        observed=True
    )
    .agg(
        Events=(
            "Short Return",
            "count"
        ),

        Avg_ADX=(
            "ADX",
            "mean"
        ),

        Avg_Long=(
            "Long Return",
            "mean"
        ),

        Avg_Short=(
            "Short Return",
            "mean"
        ),

        Short_Win_Rate=(
            "Short Return",
            lambda x:
                (x > 0).mean() * 100
        ),
    )
    .reset_index()
)


print()
print(
    "========== ADX QUARTILES =========="
)
print()

print(
    adx_quartiles.to_string(
        index=False,
        formatters={
            "Avg_ADX":
                lambda x: f"{x:.2f}",

            "Avg_Long":
                lambda x: f"{x:.3f}%",

            "Avg_Short":
                lambda x: f"{x:.3f}%",

            "Short_Win_Rate":
                lambda x: f"{x:.1f}%",
        },
    )
)


# --------------------------------------------------
# YEAR-BY-YEAR:
# ALL LARGE DROPS
# --------------------------------------------------

year_results = []

for year in sorted(
    events["Year"].unique()
):

    temp = events[
        events["Year"] == year
    ]

    year_results.append(
        {
            "Year": year,

            "Events": len(temp),

            "Avg Long":
                temp["Long Return"].mean(),

            "Avg Short":
                temp["Short Return"].mean(),

            "Short Win Rate":
                (
                    temp["Short Return"] > 0
                ).mean() * 100,
        }
    )


year_results = pd.DataFrame(
    year_results
)


print()
print(
    "===== ALL LARGE DROPS BY YEAR ====="
)
print()

print(
    year_results.to_string(
        index=False,
        formatters={
            "Avg Long":
                lambda x: f"{x:.3f}%",

            "Avg Short":
                lambda x: f"{x:.3f}%",

            "Short Win Rate":
                lambda x: f"{x:.1f}%",
        },
    )
)


# --------------------------------------------------
# YEAR-BY-YEAR:
# HIGH-ADX EVENTS ONLY
# --------------------------------------------------

high_adx = events[
    events["ADX"] >= ADX_THRESHOLD
]

high_adx_years = []

for year in sorted(
    high_adx["Year"].unique()
):

    temp = high_adx[
        high_adx["Year"] == year
    ]

    high_adx_years.append(
        {
            "Year": year,

            "Events": len(temp),

            "Avg Short":
                temp["Short Return"].mean(),

            "Median Short":
                temp["Short Return"].median(),

            "Short Win Rate":
                (
                    temp["Short Return"] > 0
                ).mean() * 100,
        }
    )


high_adx_years = pd.DataFrame(
    high_adx_years
)


print()
print(
    "===== HIGH-ADX LARGE DROPS BY YEAR ====="
)
print()

if len(high_adx_years) > 0:

    print(
        high_adx_years.to_string(
            index=False,
            formatters={
                "Avg Short":
                    lambda x: f"{x:.3f}%",

                "Median Short":
                    lambda x: f"{x:.3f}%",

                "Short Win Rate":
                    lambda x: f"{x:.1f}%",
            },
        )
    )

else:

    print(
        "No high-ADX events found."
    )


# --------------------------------------------------
# FINAL REMINDER
# --------------------------------------------------

print()
print(
    "============================================="
)
print()

print(
    "This is a diagnostic continuation analysis."
)

print(
    "No short strategy has been selected."
)

print(
    "No ADX threshold has been optimized."
)

print(
    "2026 OOS DATA HAS NOT BEEN USED."
)
