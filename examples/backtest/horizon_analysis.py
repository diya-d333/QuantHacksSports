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

# Each bar = 4 hours
HORIZONS = {
    "4 Hours": 1,
    "8 Hours": 2,
    "12 Hours": 3,
    "24 Hours": 6,
}


# --------------------------------------------------
# Load ETH
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
# ADX calculation
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
    ).mean()

    # Fix minus_di calculation
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
# SAME ETH ENTRY SIGNAL AS BEFORE
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
# Test one holding horizon
# --------------------------------------------------

def test_horizon(name, holding_bars):

    trade_returns = []
    trade_years = []

    i = 0

    # Signal at close t
    # Entry at open t+1
    # Exit after holding_bars 4-hour bars
    #
    # 4h:
    # entry open t+1
    # exit  open t+2
    #
    # 8h:
    # entry open t+1
    # exit  open t+3
    #
    # etc.

    while i < len(eth):

        if bool(signal.iloc[i]):

            entry_index = i + 1
            exit_index = entry_index + holding_bars

            if exit_index >= len(eth):
                break

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

            trade_returns.append(
                net_return
            )

            trade_years.append(
                eth.index[i].year
            )

            # One position at a time
            i = exit_index

        else:

            i += 1

    temp = pd.DataFrame(
        {
            "Year": trade_years,
            "Return": trade_returns,
        }
    )

    if len(temp) == 0:

        return None, None

    overall = {
        "Horizon": name,
        "Trades": len(temp),
        "Avg Return": temp["Return"].mean() * 100,
        "Median Return": temp["Return"].median() * 100,
        "Win Rate": (temp["Return"] > 0).mean() * 100,
        "Cumulative Return":
            ((1 + temp["Return"]).prod() - 1) * 100,
    }

    yearly = []

    for year in sorted(
        temp["Year"].unique()
    ):

        year_data = temp[
            temp["Year"] == year
        ]

        yearly.append(
            {
                "Horizon": name,
                "Year": year,
                "Trades": len(year_data),

                "Avg Return":
                    year_data["Return"].mean() * 100,

                "Median Return":
                    year_data["Return"].median() * 100,

                "Win Rate":
                    (
                        year_data["Return"] > 0
                    ).mean() * 100,

                "Cumulative Return":
                    (
                        (
                            1 + year_data["Return"]
                        ).prod() - 1
                    ) * 100,
            }
        )

    return overall, yearly


# --------------------------------------------------
# Run all horizons
# --------------------------------------------------

overall_results = []
yearly_results = []

for name, holding_bars in HORIZONS.items():

    overall, yearly = test_horizon(
        name,
        holding_bars
    )

    if overall is not None:

        overall_results.append(
            overall
        )

        yearly_results.extend(
            yearly
        )


overall_df = pd.DataFrame(
    overall_results
)

yearly_df = pd.DataFrame(
    yearly_results
)


# --------------------------------------------------
# Print overall comparison
# --------------------------------------------------

print()
print(
    "========== HOLDING HORIZON ANALYSIS =========="
)
print()

print(
    overall_df.to_string(
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


# --------------------------------------------------
# Print yearly comparison
# --------------------------------------------------

print()
print(
    "========== RESULTS BY YEAR =========="
)
print()

print(
    yearly_df.to_string(
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
    "============================================="
)

print()
print("ENTRY RULES WERE NOT CHANGED:")
print(
    f"Large ETH drop > {DROP_THRESHOLD}x volatility"
)
print(
    f"ADX < {ADX_THRESHOLD}"
)
print(
    f"Commission = {COMMISSION * 100:.1f}% per side"
)
print(
    "Only one ETH position at a time"
)

print()
print(
    "Horizons are diagnostic development tests."
)
print(
    "No final holding period has been selected."
)
print(
    "2026 OOS DATA HAS NOT BEEN USED."
)
