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
# LOAD ETH DATA
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
print(
    f"Sentiment observations: "
    f"{eth['fear_greed_score'].notna().sum()}"
)


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
# SAME EVENT DEFINITION AS BEFORE
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

range_bound = (
    eth_adx < ADX_THRESHOLD
)

signal = (
    large_drop
    & range_bound
)


# --------------------------------------------------
# BUILD EVENT / TRADE DATASET
# --------------------------------------------------
#
# Same execution assumption:
#
# Signal at close t
# Entry at open t+1
# Exit at open t+2
#
# We use one position at a time.
# --------------------------------------------------

trades = []

i = 0

while i < len(eth) - 2:

    if bool(signal.iloc[i]):

        signal_time = eth.index[i]

        entry_index = i + 1
        exit_index = i + 2

        entry_price = eth["open"].iloc[
            entry_index
        ]

        exit_price = eth["open"].iloc[
            exit_index
        ]

        gross_return = (
            exit_price / entry_price
        ) - 1

        net_return = (
            gross_return
            - (2 * COMMISSION)
        )

        fear_greed = (
            eth["fear_greed_score"].iloc[i]
        )

        trades.append(
            {
                "Year": signal_time.year,
                "Signal Time": signal_time,
                "Fear Greed": fear_greed,
                "ETH Drop":
                    eth_return.iloc[i] * 100,
                "ADX":
                    eth_adx.iloc[i],
                "Net Return":
                    net_return * 100,
            }
        )

        # One position at a time
        i = exit_index

    else:

        i += 1


trades = pd.DataFrame(trades)


# --------------------------------------------------
# BASIC SENTIMENT COMPARISON
# --------------------------------------------------

print()
print(
    "========== SENTIMENT DIAGNOSTIC =========="
)
print()

print(f"Total ETH trades: {len(trades)}")

winning = trades[
    trades["Net Return"] > 0
]

losing = trades[
    trades["Net Return"] <= 0
]


print()
print("AVERAGE FEAR & GREED:")

print(
    f"Winning trades: "
    f"{winning['Fear Greed'].mean():.2f}"
)

print(
    f"Losing trades:  "
    f"{losing['Fear Greed'].mean():.2f}"
)


print()
print("MEDIAN FEAR & GREED:")

print(
    f"Winning trades: "
    f"{winning['Fear Greed'].median():.2f}"
)

print(
    f"Losing trades:  "
    f"{losing['Fear Greed'].median():.2f}"
)


correlation = (
    trades[
        ["Fear Greed", "Net Return"]
    ]
    .corr()
    .iloc[0, 1]
)

print()
print(
    "Correlation between Fear & Greed "
    f"and trade return: {correlation:.3f}"
)


# --------------------------------------------------
# STANDARD FEAR & GREED CATEGORIES
# --------------------------------------------------
#
# These are descriptive categories.
# We are NOT selecting a trading threshold.
#
# 0-24   = Extreme Fear
# 25-44  = Fear
# 45-55  = Neutral
# 56-74  = Greed
# 75-100 = Extreme Greed
# --------------------------------------------------

bins = [
    -1,
    24,
    44,
    55,
    74,
    100,
]

labels = [
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed",
]

trades["Sentiment Group"] = pd.cut(
    trades["Fear Greed"],
    bins=bins,
    labels=labels,
)


sentiment_results = (
    trades
    .groupby(
        "Sentiment Group",
        observed=True
    )
    .agg(
        Trades=("Net Return", "count"),

        Avg_Return=(
            "Net Return",
            "mean"
        ),

        Median_Return=(
            "Net Return",
            "median"
        ),

        Win_Rate=(
            "Net Return",
            lambda x:
                (x > 0).mean() * 100
        ),

        Avg_Fear_Greed=(
            "Fear Greed",
            "mean"
        ),
    )
    .reset_index()
)


print()
print(
    "========== SENTIMENT GROUPS =========="
)
print()

print(
    sentiment_results.to_string(
        index=False,
        formatters={
            "Avg_Return":
                lambda x: f"{x:.3f}%",

            "Median_Return":
                lambda x: f"{x:.3f}%",

            "Win_Rate":
                lambda x: f"{x:.1f}%",

            "Avg_Fear_Greed":
                lambda x: f"{x:.1f}",
        },
    )
)


# --------------------------------------------------
# SENTIMENT QUARTILES
# --------------------------------------------------
#
# Quartiles avoid us manually choosing thresholds.
# They simply divide observed event sentiment
# into four similarly sized groups.
# --------------------------------------------------

trades["Sentiment Quartile"] = pd.qcut(
    trades["Fear Greed"],
    4,
    duplicates="drop"
)


quartile_results = (
    trades
    .groupby(
        "Sentiment Quartile",
        observed=True
    )
    .agg(
        Trades=("Net Return", "count"),

        Avg_Return=(
            "Net Return",
            "mean"
        ),

        Median_Return=(
            "Net Return",
            "median"
        ),

        Win_Rate=(
            "Net Return",
            lambda x:
                (x > 0).mean() * 100
        ),

        Avg_Fear_Greed=(
            "Fear Greed",
            "mean"
        ),
    )
    .reset_index()
)


print()
print(
    "========== SENTIMENT QUARTILES =========="
)
print()

print(
    quartile_results.to_string(
        index=False,
        formatters={
            "Avg_Return":
                lambda x: f"{x:.3f}%",

            "Median_Return":
                lambda x: f"{x:.3f}%",

            "Win_Rate":
                lambda x: f"{x:.1f}%",

            "Avg_Fear_Greed":
                lambda x: f"{x:.1f}",
        },
    )
)


# --------------------------------------------------
# YEAR-BY-YEAR RELATIONSHIP
# --------------------------------------------------

year_results = []

for year in sorted(
    trades["Year"].unique()
):

    temp = trades[
        trades["Year"] == year
    ]

    corr = (
        temp[
            ["Fear Greed", "Net Return"]
        ]
        .corr()
        .iloc[0, 1]
    )

    year_results.append(
        {
            "Year": year,

            "Trades": len(temp),

            "Correlation": corr,

            "Avg Fear Greed":
                temp["Fear Greed"].mean(),

            "Avg Trade Return":
                temp["Net Return"].mean(),
        }
    )


year_results = pd.DataFrame(
    year_results
)


print()
print(
    "========== YEAR-BY-YEAR =========="
)
print()

print(
    year_results.to_string(
        index=False,
        formatters={
            "Correlation":
                lambda x: f"{x:.3f}",

            "Avg Fear Greed":
                lambda x: f"{x:.1f}",

            "Avg Trade Return":
                lambda x: f"{x:.3f}%",
        },
    )
)


# --------------------------------------------------
# EXTREME FEAR YEAR-BY-YEAR
# --------------------------------------------------

extreme_fear = trades[
    trades["Fear Greed"] <= 24
]

extreme_year_results = []

for year in sorted(
    trades["Year"].unique()
):

    temp = extreme_fear[
        extreme_fear["Year"] == year
    ]

    if len(temp) == 0:
        continue

    extreme_year_results.append(
        {
            "Year": year,

            "Trades": len(temp),

            "Avg Return":
                temp["Net Return"].mean(),

            "Median Return":
                temp["Net Return"].median(),

            "Win Rate":
                (
                    temp["Net Return"] > 0
                ).mean() * 100,
        }
    )


extreme_year_results = pd.DataFrame(
    extreme_year_results
)


print()
print(
    "===== EXTREME FEAR EVENTS BY YEAR ====="
)
print()

if len(extreme_year_results) > 0:

    print(
        extreme_year_results.to_string(
            index=False,
            formatters={
                "Avg Return":
                    lambda x: f"{x:.3f}%",

                "Median Return":
                    lambda x: f"{x:.3f}%",

                "Win Rate":
                    lambda x: f"{x:.1f}%",
            },
        )
    )

else:

    print(
        "No Extreme Fear events found."
    )


# --------------------------------------------------
# FINAL REMINDER
# --------------------------------------------------

print()
print(
    "============================================="
)

print()
print("IMPORTANT:")
print(
    "This is a diagnostic sentiment analysis."
)
print(
    "No sentiment threshold has been added "
    "to the strategy."
)
print(
    "The ETH entry signal has not been changed."
)
print(
    "2026 OOS DATA HAS NOT BEEN USED."
)
