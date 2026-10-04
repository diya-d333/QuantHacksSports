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

COMMISSION = 0.001


# --------------------------------------------------
# LOAD ETH + BTC
# --------------------------------------------------

print("Loading ETH and BTC 2021-2025 data...")

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


# Use only timestamps available for BOTH assets
common_times = eth.index.intersection(
    btc.index
)

eth = eth.loc[common_times].copy()
btc = btc.loc[common_times].copy()


print(f"Shared ETH/BTC bars: {len(common_times)}")
print(f"Start: {common_times.min()}")
print(f"End:   {common_times.max()}")


# --------------------------------------------------
# RETURNS
# --------------------------------------------------

eth_return = eth["close"].pct_change()
btc_return = btc["close"].pct_change()


# --------------------------------------------------
# ORIGINAL LARGE ETH DROP EVENT
# --------------------------------------------------

eth_volatility = (
    eth_return
    .rolling(LOOKBACK)
    .std()
)

large_eth_drop = (
    eth_return
    < -DROP_THRESHOLD * eth_volatility
)


# --------------------------------------------------
# RELATIVE ETH VS BTC MOVE
# --------------------------------------------------
#
# Example:
#
# ETH = -5%
# BTC = -2%
#
# ETH relative move = -3%
#
# More negative means ETH underperformed BTC more.
# --------------------------------------------------

relative_move = (
    eth_return - btc_return
)


# --------------------------------------------------
# BUILD EVENT DATASET
# --------------------------------------------------
#
# We keep the ORIGINAL large ETH drop definition.
#
# Signal information is from close t.
#
# Entry = next open (t+1)
# Exit  = following open (t+2)
#
# We calculate:
#
# 1. ETH long return
# 2. BTC long return
# 3. ETH minus BTC forward performance
#
# If ETH really mean-reverts relative to BTC,
# larger negative dislocations may be followed
# by positive ETH-vs-BTC relative performance.
# --------------------------------------------------

events = []

i = 0

while i < len(eth) - 2:

    if bool(large_eth_drop.iloc[i]):

        signal_time = eth.index[i]

        entry_index = i + 1
        exit_index = i + 2

        # ------------------------------
        # ETH hypothetical long
        # ------------------------------

        eth_entry = eth["open"].iloc[
            entry_index
        ]

        eth_exit = eth["open"].iloc[
            exit_index
        ]

        eth_gross = (
            eth_exit / eth_entry
        ) - 1

        eth_net = (
            eth_gross
            - (2 * COMMISSION)
        )


        # ------------------------------
        # BTC hypothetical long
        # ------------------------------

        btc_entry = btc["open"].iloc[
            entry_index
        ]

        btc_exit = btc["open"].iloc[
            exit_index
        ]

        btc_gross = (
            btc_exit / btc_entry
        ) - 1

        btc_net = (
            btc_gross
            - (2 * COMMISSION)
        )


        # ------------------------------
        # Relative forward performance
        # ------------------------------
        #
        # Positive means ETH outperformed BTC
        # after the signal.
        # ------------------------------

        forward_relative = (
            eth_gross - btc_gross
        )


        events.append(
            {
                "Year": signal_time.year,

                "Signal Time":
                    signal_time,

                "ETH Return":
                    eth_return.iloc[i] * 100,

                "BTC Return":
                    btc_return.iloc[i] * 100,

                "Relative Move":
                    relative_move.iloc[i] * 100,

                "ETH Net Return":
                    eth_net * 100,

                "BTC Net Return":
                    btc_net * 100,

                "Forward Relative":
                    forward_relative * 100,
            }
        )

        # Keep one hypothetical position
        # at a time, matching earlier tests.
        i = exit_index

    else:

        i += 1


events = pd.DataFrame(events)


# --------------------------------------------------
# BASIC DIAGNOSTIC
# --------------------------------------------------

print()
print(
    "========== RELATIVE DISLOCATION DIAGNOSTIC =========="
)
print()

print(f"Large ETH-drop events: {len(events)}")

print()

print(
    "Average ETH signal-bar return: "
    f"{events['ETH Return'].mean():.3f}%"
)

print(
    "Average BTC signal-bar return: "
    f"{events['BTC Return'].mean():.3f}%"
)

print(
    "Average ETH-BTC relative move: "
    f"{events['Relative Move'].mean():.3f}%"
)

print()

print(
    "Average subsequent ETH net return: "
    f"{events['ETH Net Return'].mean():.3f}%"
)

print(
    "Average subsequent ETH-vs-BTC performance: "
    f"{events['Forward Relative'].mean():.3f}%"
)


# --------------------------------------------------
# CORRELATIONS
# --------------------------------------------------

correlation_eth = (
    events[
        ["Relative Move", "ETH Net Return"]
    ]
    .corr()
    .iloc[0, 1]
)

correlation_relative = (
    events[
        ["Relative Move", "Forward Relative"]
    ]
    .corr()
    .iloc[0, 1]
)


print()
print(
    "Correlation:"
)

print(
    "Signal ETH-BTC relative move vs "
    "future ETH return: "
    f"{correlation_eth:.3f}"
)

print(
    "Signal ETH-BTC relative move vs "
    "future ETH-BTC performance: "
    f"{correlation_relative:.3f}"
)


# --------------------------------------------------
# RELATIVE-MOVE QUARTILES
# --------------------------------------------------
#
# IMPORTANT:
#
# These are descriptive quartiles.
# We are NOT selecting a cutoff.
#
# Quartile 1 = ETH underperformed BTC the most.
# Quartile 4 = ETH underperformed BTC the least
#              (or possibly outperformed BTC).
# --------------------------------------------------

events["Relative Quartile"] = pd.qcut(
    events["Relative Move"],
    4,
    duplicates="drop"
)


quartile_results = (
    events
    .groupby(
        "Relative Quartile",
        observed=True
    )
    .agg(
        Events=(
            "Forward Relative",
            "count"
        ),

        Avg_Relative_Move=(
            "Relative Move",
            "mean"
        ),

        Avg_ETH_Return=(
            "ETH Net Return",
            "mean"
        ),

        Median_ETH_Return=(
            "ETH Net Return",
            "median"
        ),

        ETH_Win_Rate=(
            "ETH Net Return",
            lambda x:
                (x > 0).mean() * 100
        ),

        Avg_Forward_Relative=(
            "Forward Relative",
            "mean"
        ),

        Relative_Win_Rate=(
            "Forward Relative",
            lambda x:
                (x > 0).mean() * 100
        ),
    )
    .reset_index()
)


print()
print(
    "========== RELATIVE-MOVE QUARTILES =========="
)
print()

print(
    quartile_results.to_string(
        index=False,
        formatters={
            "Avg_Relative_Move":
                lambda x: f"{x:.3f}%",

            "Avg_ETH_Return":
                lambda x: f"{x:.3f}%",

            "Median_ETH_Return":
                lambda x: f"{x:.3f}%",

            "ETH_Win_Rate":
                lambda x: f"{x:.1f}%",

            "Avg_Forward_Relative":
                lambda x: f"{x:.3f}%",

            "Relative_Win_Rate":
                lambda x: f"{x:.1f}%",
        },
    )
)


# --------------------------------------------------
# MOST EXTREME RELATIVE-DISLOCATION QUARTILE
# BY YEAR
# --------------------------------------------------
#
# qcut labels depend on the actual values,
# so identify the lowest quartile using
# the 25th percentile.
#
# Again: diagnostic only.
# --------------------------------------------------

q25 = events["Relative Move"].quantile(
    0.25
)

most_extreme = events[
    events["Relative Move"] <= q25
]


extreme_year_results = []

for year in sorted(
    most_extreme["Year"].unique()
):

    temp = most_extreme[
        most_extreme["Year"] == year
    ]

    extreme_year_results.append(
        {
            "Year": year,

            "Events": len(temp),

            "Avg Signal Dislocation":
                temp["Relative Move"].mean(),

            "Avg ETH Return":
                temp["ETH Net Return"].mean(),

            "Median ETH Return":
                temp["ETH Net Return"].median(),

            "ETH Win Rate":
                (
                    temp["ETH Net Return"] > 0
                ).mean() * 100,

            "Avg Relative Rebound":
                temp["Forward Relative"].mean(),

            "Relative Win Rate":
                (
                    temp["Forward Relative"] > 0
                ).mean() * 100,
        }
    )


extreme_year_results = pd.DataFrame(
    extreme_year_results
)


print()
print(
    "===== MOST EXTREME ETH UNDERPERFORMANCE BY YEAR ====="
)
print()

print(
    extreme_year_results.to_string(
        index=False,
        formatters={
            "Avg Signal Dislocation":
                lambda x: f"{x:.3f}%",

            "Avg ETH Return":
                lambda x: f"{x:.3f}%",

            "Median ETH Return":
                lambda x: f"{x:.3f}%",

            "ETH Win Rate":
                lambda x: f"{x:.1f}%",

            "Avg Relative Rebound":
                lambda x: f"{x:.3f}%",

            "Relative Win Rate":
                lambda x: f"{x:.1f}%",
        },
    )
)


# --------------------------------------------------
# CORRELATION BY YEAR
# --------------------------------------------------

year_results = []

for year in sorted(
    events["Year"].unique()
):

    temp = events[
        events["Year"] == year
    ]

    corr = (
        temp[
            [
                "Relative Move",
                "Forward Relative"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    year_results.append(
        {
            "Year": year,

            "Events": len(temp),

            "Correlation": corr,

            "Avg Signal Dislocation":
                temp["Relative Move"].mean(),

            "Avg Forward Relative":
                temp["Forward Relative"].mean(),
        }
    )


year_results = pd.DataFrame(
    year_results
)


print()
print(
    "========== YEAR-BY-YEAR RELATIONSHIP =========="
)
print()

print(
    year_results.to_string(
        index=False,
        formatters={
            "Correlation":
                lambda x: f"{x:.3f}",

            "Avg Signal Dislocation":
                lambda x: f"{x:.3f}%",

            "Avg Forward Relative":
                lambda x: f"{x:.3f}%",
        },
    )
)


# --------------------------------------------------
# REMINDER
# --------------------------------------------------

print()
print(
    "===================================================="
)
print()

print(
    "This is a diagnostic relative-dislocation analysis."
)

print(
    "No ETH-vs-BTC threshold has been selected."
)

print(
    "No new strategy has been selected."
)

print(
    "2026 OOS DATA HAS NOT BEEN USED."
)
