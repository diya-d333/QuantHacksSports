from pathlib import Path
import sys

import pandas as pd
import numpy as np


# --------------------------------------------------
# Make backtest folder importable
# --------------------------------------------------

BACKTEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKTEST_DIR))

from load_tiger_data import load_market_data


# --------------------------------------------------
# FROZEN SETTINGS
# --------------------------------------------------

START_DATE = "2021-01-01T00:00:00Z"
END_DATE = "2026-01-01T00:00:00Z"

LOOKBACK = 20
DROP_THRESHOLD = 2.0

ADX_PERIOD = 14
ADX_THRESHOLD = 25

COMMISSION = 0.001


# --------------------------------------------------
# Load ETH data only
# --------------------------------------------------

print("Loading ETH data from 2021-2025...")

eth = load_market_data(
    "ETH",
    START_DATE,
    END_DATE
)

print(f"Total 4-hour bars: {len(eth)}")
print(f"Start: {eth.index.min()}")
print(f"End:   {eth.index.max()}")


# --------------------------------------------------
# Calculate ADX
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
# Build exact same ETH signal
# --------------------------------------------------

eth_return = eth["close"].pct_change()

eth_volatility = eth_return.rolling(
    LOOKBACK
).std()

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
# Simulate ONE POSITION AT A TIME
#
# Signal: close of bar t
# Entry:  open of bar t+1
# Exit:   open of bar t+2
# --------------------------------------------------

trades = []

i = 0

while i < len(eth) - 2:

    if bool(signal.iloc[i]):

        signal_time = eth.index[i]

        entry_index = i + 1
        exit_index = i + 2

        entry_time = eth.index[entry_index]
        exit_time = eth.index[exit_index]

        entry_price = eth["open"].iloc[entry_index]
        exit_price = eth["open"].iloc[exit_index]

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
                "Signal Time": signal_time,
                "Entry Time": entry_time,
                "Exit Time": exit_time,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "ADX": eth_adx.iloc[i],
                "ETH Drop": eth_return.iloc[i] * 100,
                "Net Return": net_return,
            }
        )

        # Cannot enter another trade while
        # this position is active
        i = exit_index

    else:

        i += 1


trades = pd.DataFrame(trades)


# --------------------------------------------------
# Calculate yearly results
# --------------------------------------------------

yearly_results = []

for year in [
    2021,
    2022,
    2023,
    2024,
    2025,
]:

    year_trades = trades[
        trades["Year"] == year
    ]

    if len(year_trades) == 0:

        yearly_results.append(
            {
                "Year": year,
                "Trades": 0,
                "Avg Return": float("nan"),
                "Median Return": float("nan"),
                "Win Rate": float("nan"),
                "Cumulative Return": float("nan"),
            }
        )

        continue

    returns = year_trades["Net Return"]

    average_return = returns.mean()

    median_return = returns.median()

    win_rate = (
        returns > 0
    ).mean()

    cumulative_return = (
        (1 + returns).prod() - 1
    )

    yearly_results.append(
        {
            "Year": year,
            "Trades": len(year_trades),

            "Avg Return":
                average_return * 100,

            "Median Return":
                median_return * 100,

            "Win Rate":
                win_rate * 100,

            "Cumulative Return":
                cumulative_return * 100,
        }
    )


results = pd.DataFrame(
    yearly_results
)


# --------------------------------------------------
# Overall results
# --------------------------------------------------

overall_returns = trades[
    "Net Return"
]

overall_cumulative = (
    (1 + overall_returns).prod() - 1
)


# --------------------------------------------------
# Print
# --------------------------------------------------

print()
print(
    "========== ETH + ADX MULTI-YEAR TEST =========="
)
print()

print(
    results.to_string(
        index=False,
        formatters={
            "Avg Return":
                lambda x: f"{x:.3f}%",

            "Median Return":
                lambda x: f"{x:.3f}%",

            "Win Rate":
                lambda x: f"{x:.1f}%",

            "Cumulative Return":
                lambda x: f"{x:.3f}%",
        },
    )
)

print()
print(
    f"Total trades: {len(trades)}"
)

print(
    f"Overall cumulative trade return: "
    f"{overall_cumulative * 100:.3f}%"
)

print()
print(
    "==============================================="
)

print()
print("RULES:")
print(
    f"Large drop = more than "
    f"{DROP_THRESHOLD}x recent volatility"
)
print(
    f"ADX period = {ADX_PERIOD}"
)
print(
    f"Range-bound = ADX < {ADX_THRESHOLD}"
)
print(
    "Signal = current 4-hour close"
)
print(
    "Entry = next 4-hour open"
)
print(
    "Exit = following 4-hour open"
)
print(
    f"Commission = "
    f"{COMMISSION * 100:.1f}% per side"
)
print(
    "Only one ETH position at a time"
)

print()
print(
    "2026 OOS DATA HAS NOT BEEN USED."
)
