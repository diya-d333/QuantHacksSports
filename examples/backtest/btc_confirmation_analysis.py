from pathlib import Path
import sys

import pandas as pd
import numpy as np


# --------------------------------------------------
# Setup
# --------------------------------------------------

BACKTEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKTEST_DIR))

from load_tiger_data import load_market_data


START_DATE = "2021-01-01T00:00:00Z"
END_DATE = "2026-01-01T00:00:00Z"

LOOKBACK = 20
DROP_THRESHOLD = 2.0

ADX_PERIOD = 14
ADX_THRESHOLD = 25

COMMISSION = 0.001


# --------------------------------------------------
# Load ETH + BTC
# --------------------------------------------------

print("Loading ETH and BTC data...")

eth = load_market_data(
    "ETH",
    START_DATE,
    END_DATE
)

btc = load_market_data(
    "BTC",
    START_DATE,
    END_DATE
)


# Keep only timestamps shared by ETH and BTC

common_times = eth.index.intersection(
    btc.index
)

eth = eth.loc[common_times].copy()
btc = btc.loc[common_times].copy()

print(f"Shared ETH/BTC bars: {len(common_times)}")
print(f"Start: {common_times.min()}")
print(f"End:   {common_times.max()}")


# --------------------------------------------------
# ADX
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
# Existing ETH signal
# --------------------------------------------------

eth_return = eth["close"].pct_change()
btc_return = btc["close"].pct_change()

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
# Simulate same one-position strategy
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

        trades.append(
            {
                "Year": signal_time.year,

                "Signal Time":
                    signal_time,

                "ETH Drop":
                    eth_return.iloc[i] * 100,

                "BTC Return":
                    btc_return.iloc[i] * 100,

                "ADX":
                    eth_adx.iloc[i],

                "Net Return":
                    net_return * 100,
            }
        )

        i = exit_index

    else:

        i += 1


trades = pd.DataFrame(trades)


# --------------------------------------------------
# Relationship between BTC move and trade outcome
# --------------------------------------------------

print()
print(
    "========== BTC CONFIRMATION DIAGNOSTIC =========="
)
print()

print(f"Total ETH trades: {len(trades)}")

print()

winning = trades[
    trades["Net Return"] > 0
]

losing = trades[
    trades["Net Return"] <= 0
]


print("AVERAGE BTC RETURN AT SIGNAL:")
print(
    f"Winning ETH trades: "
    f"{winning['BTC Return'].mean():.3f}%"
)

print(
    f"Losing ETH trades:  "
    f"{losing['BTC Return'].mean():.3f}%"
)


print()
print("MEDIAN BTC RETURN AT SIGNAL:")

print(
    f"Winning ETH trades: "
    f"{winning['BTC Return'].median():.3f}%"
)

print(
    f"Losing ETH trades:  "
    f"{losing['BTC Return'].median():.3f}%"
)


# --------------------------------------------------
# Correlation
# --------------------------------------------------

correlation = trades[
    ["BTC Return", "Net Return"]
].corr().iloc[0, 1]

print()
print(
    f"Correlation between BTC return and "
    f"ETH trade return: {correlation:.3f}"
)


# --------------------------------------------------
# BTC return quartiles
#
# These are descriptive buckets, NOT strategy rules.
# --------------------------------------------------

trades["BTC Quartile"] = pd.qcut(
    trades["BTC Return"],
    4,
    duplicates="drop"
)

quartile_results = (
    trades
    .groupby(
        "BTC Quartile",
        observed=True
    )
    .agg(
        Trades=("Net Return", "count"),
        Avg_ETH_Trade=("Net Return", "mean"),
        Median_ETH_Trade=("Net Return", "median"),
        Win_Rate=(
            "Net Return",
            lambda x: (x > 0).mean() * 100
        ),
        Avg_BTC_Return=("BTC Return", "mean"),
    )
    .reset_index()
)


print()
print(
    "========== BTC RETURN QUARTILES =========="
)
print()

print(
    quartile_results.to_string(
        index=False,
        formatters={
            "Avg_ETH_Trade":
                lambda x: f"{x:.3f}%",

            "Median_ETH_Trade":
                lambda x: f"{x:.3f}%",

            "Win_Rate":
                lambda x: f"{x:.1f}%",

            "Avg_BTC_Return":
                lambda x: f"{x:.3f}%",
        },
    )
)


# --------------------------------------------------
# Year-by-year relationship
# --------------------------------------------------

year_results = []

for year in sorted(
    trades["Year"].unique()
):

    temp = trades[
        trades["Year"] == year
    ]

    corr = temp[
        ["BTC Return", "Net Return"]
    ].corr().iloc[0, 1]

    year_results.append(
        {
            "Year": year,
            "Trades": len(temp),
            "Correlation": corr,
            "Avg BTC Return":
                temp["BTC Return"].mean(),
            "Avg ETH Trade":
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

            "Avg BTC Return":
                lambda x: f"{x:.3f}%",

            "Avg ETH Trade":
                lambda x: f"{x:.3f}%",
        },
    )
)


print()
print(
    "IMPORTANT: BTC quartiles are diagnostic only."
)
print(
    "No BTC threshold has been added to the strategy."
)
print(
    "2026 OOS DATA HAS NOT BEEN USED."
)
