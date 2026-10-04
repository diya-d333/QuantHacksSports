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
# DEVELOPMENT DATA ONLY
# --------------------------------------------------
#
# 2021-2024 = development
# 2025      = not used in this test
# 2026      = locked final OOS
# --------------------------------------------------

START_DATE = "2021-01-01T00:00:00Z"
END_DATE = "2025-01-01T00:00:00Z"


# --------------------------------------------------
# DIAGNOSTIC HORIZONS
# --------------------------------------------------
#
# Each bar = 4 hours
#
# We compare recent BTC-vs-ETH performance
# with FUTURE BTC-vs-ETH performance.
#
# No trading rule is being selected yet.
# --------------------------------------------------

HORIZONS = {
    "4 Hours": 1,
    "12 Hours": 3,
    "24 Hours": 6,
    "48 Hours": 12,
}


# --------------------------------------------------
# LOAD BTC + ETH
# --------------------------------------------------

print("Loading BTC and ETH development data...")
print("Development period: 2021-2024")
print()

btc = load_market_data(
    "BTC",
    START_DATE,
    END_DATE
)

eth = load_market_data(
    "ETH",
    START_DATE,
    END_DATE
)


# Use timestamps available for both
common_times = btc.index.intersection(
    eth.index
)

btc = btc.loc[common_times].copy()
eth = eth.loc[common_times].copy()


print(f"Shared bars: {len(common_times)}")
print(f"Start: {common_times.min()}")
print(f"End:   {common_times.max()}")


# --------------------------------------------------
# CALCULATE CLOSE-TO-CLOSE RETURNS
# --------------------------------------------------

btc_return = btc["close"].pct_change()
eth_return = eth["close"].pct_change()


# --------------------------------------------------
# TEST EACH HORIZON
# --------------------------------------------------
#
# Example with 24 hours:
#
# Look backward 6 bars:
# Which asset performed better?
#
# Then look forward 6 bars:
# Which asset performed better?
#
# If relative strength persists:
#
# positive past BTC-ETH performance
# should tend to be followed by
# positive future BTC-ETH performance.
#
# --------------------------------------------------

results = []
year_results = []

for horizon_name, bars in HORIZONS.items():

    # ----------------------------------------------
    # Past performance over the selected horizon
    # ----------------------------------------------

    btc_past = (
        btc["close"]
        / btc["close"].shift(bars)
        - 1
    )

    eth_past = (
        eth["close"]
        / eth["close"].shift(bars)
        - 1
    )

    past_relative = (
        btc_past - eth_past
    )


    # ----------------------------------------------
    # Future performance over same horizon
    # ----------------------------------------------

    btc_future = (
        btc["close"].shift(-bars)
        / btc["close"]
        - 1
    )

    eth_future = (
        eth["close"].shift(-bars)
        / eth["close"]
        - 1
    )

    future_relative = (
        btc_future - eth_future
    )


    # ----------------------------------------------
    # Build clean dataframe
    # ----------------------------------------------

    temp = pd.DataFrame(
        {
            "Past Relative":
                past_relative,

            "Future Relative":
                future_relative,
        }
    ).dropna()


    # ----------------------------------------------
    # Did the same asset remain stronger?
    # ----------------------------------------------

    temp["Same Leader"] = (
        np.sign(temp["Past Relative"])
        ==
        np.sign(temp["Future Relative"])
    )


    # ----------------------------------------------
    # Correlation
    # ----------------------------------------------

    correlation = (
        temp[
            [
                "Past Relative",
                "Future Relative"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )


    persistence_rate = (
        temp["Same Leader"].mean()
        * 100
    )


    results.append(
        {
            "Horizon": horizon_name,
            "Observations": len(temp),
            "Correlation": correlation,
            "Same-Leader Rate":
                persistence_rate,
            "Avg Past Relative":
                temp["Past Relative"].mean()
                * 100,
            "Avg Future Relative":
                temp["Future Relative"].mean()
                * 100,
        }
    )


    # ----------------------------------------------
    # YEAR-BY-YEAR
    # ----------------------------------------------

    temp["Year"] = temp.index.year

    for year in sorted(
        temp["Year"].unique()
    ):

        year_data = temp[
            temp["Year"] == year
        ]

        year_corr = (
            year_data[
                [
                    "Past Relative",
                    "Future Relative"
                ]
            ]
            .corr()
            .iloc[0, 1]
        )

        year_persistence = (
            year_data["Same Leader"].mean()
            * 100
        )

        year_results.append(
            {
                "Horizon": horizon_name,
                "Year": year,
                "Observations":
                    len(year_data),
                "Correlation":
                    year_corr,
                "Same-Leader Rate":
                    year_persistence,
            }
        )


# --------------------------------------------------
# OVERALL RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)


print()
print(
    "========== RELATIVE-STRENGTH PERSISTENCE =========="
)
print()

print(
    results_df.to_string(
        index=False,
        formatters={
            "Correlation":
                lambda x: f"{x:.3f}",

            "Same-Leader Rate":
                lambda x: f"{x:.1f}%",

            "Avg Past Relative":
                lambda x: f"{x:.3f}%",

            "Avg Future Relative":
                lambda x: f"{x:.3f}%",
        },
    )
)


# --------------------------------------------------
# YEAR-BY-YEAR RESULTS
# --------------------------------------------------

year_df = pd.DataFrame(year_results)


print()
print(
    "========== YEAR-BY-YEAR =========="
)
print()

print(
    year_df.to_string(
        index=False,
        formatters={
            "Correlation":
                lambda x: f"{x:.3f}",

            "Same-Leader Rate":
                lambda x: f"{x:.1f}%",
        },
    )
)


# --------------------------------------------------
# QUARTILE ANALYSIS
# --------------------------------------------------
#
# Now ask:
#
# Do STRONGER relative-strength signals
# show more persistence?
#
# We use the absolute size of the BTC-ETH
# relative move.
#
# Still diagnostic only.
# --------------------------------------------------

print()
print(
    "========== SIGNAL-STRENGTH QUARTILES =========="
)
print()


for horizon_name, bars in HORIZONS.items():

    btc_past = (
        btc["close"]
        / btc["close"].shift(bars)
        - 1
    )

    eth_past = (
        eth["close"]
        / eth["close"].shift(bars)
        - 1
    )

    past_relative = (
        btc_past - eth_past
    )

    btc_future = (
        btc["close"].shift(-bars)
        / btc["close"]
        - 1
    )

    eth_future = (
        eth["close"].shift(-bars)
        / eth["close"]
        - 1
    )

    future_relative = (
        btc_future - eth_future
    )

    temp = pd.DataFrame(
        {
            "Past Relative":
                past_relative,

            "Future Relative":
                future_relative,
        }
    ).dropna()


    temp["Signal Strength"] = (
        temp["Past Relative"].abs()
    )


    temp["Correct Direction"] = (
        np.sign(temp["Past Relative"])
        ==
        np.sign(temp["Future Relative"])
    )


    temp["Strength Quartile"] = pd.qcut(
        temp["Signal Strength"],
        4,
        labels=[
            "Q1 Weakest",
            "Q2",
            "Q3",
            "Q4 Strongest",
        ],
        duplicates="drop",
    )


    quartile = (
        temp
        .groupby(
            "Strength Quartile",
            observed=True
        )
        .agg(
            Observations=(
                "Correct Direction",
                "count"
            ),

            Avg_Signal_Strength=(
                "Signal Strength",
                "mean"
            ),

            Persistence_Rate=(
                "Correct Direction",
                lambda x:
                    x.mean() * 100
            ),
        )
        .reset_index()
    )


    quartile[
        "Avg_Signal_Strength"
    ] *= 100


    print()
    print(f"--- {horizon_name} ---")
    print()

    print(
        quartile.to_string(
            index=False,
            formatters={
                "Avg_Signal_Strength":
                    lambda x: f"{x:.3f}%",

                "Persistence_Rate":
                    lambda x: f"{x:.1f}%",
            },
        )
    )


# --------------------------------------------------
# FINAL REMINDER
# --------------------------------------------------

print()
print(
    "===================================================="
)
print()

print(
    "This is a development diagnostic only."
)

print(
    "No lookback or holding period has been selected."
)

print(
    "No trading strategy has been selected."
)

print(
    "2025 WAS NOT USED IN THIS TEST."
)

print(
    "2026 OOS DATA HAS NOT BEEN USED."
)
