from pathlib import Path
import sys

import numpy as np
import pandas as pd


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
END_DATE = "2025-01-01T00:00:00Z"

LOOKBACK = 20
SHOCK_ZSCORE = 2.0

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = 2 * (
    FEE_PER_SIDE + SLIPPAGE_PER_SIDE
)


# --------------------------------------------------
# ANALYZE ONE ASSET
# --------------------------------------------------

def analyze_asset(asset):

    print()
    print("=" * 70)
    print(asset)
    print("=" * 70)

    df = load_market_data(
        asset,
        START_DATE,
        END_DATE
    ).copy()

    df = df.sort_index()


    # --------------------------------------------------
    # 4-HOUR RETURN
    # --------------------------------------------------

    df["return_1"] = (
        df["close"].pct_change()
    )


    # --------------------------------------------------
    # HISTORICAL MEAN / VOLATILITY
    # --------------------------------------------------
    #
    # shift(1) prevents the current return from being
    # used to determine whether itself is unusual.
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
    # POSITIVE / NEGATIVE SHOCKS
    # --------------------------------------------------

    df["positive_shock"] = (
        df["shock_zscore"] >= SHOCK_ZSCORE
    )

    df["negative_shock"] = (
        df["shock_zscore"] <= -SHOCK_ZSCORE
    )

    df["shock"] = (
        df["positive_shock"]
        | df["negative_shock"]
    )


    # --------------------------------------------------
    # EXECUTION
    # --------------------------------------------------
    #
    # Signal:
    # current bar closes
    #
    # Entry:
    # next bar open
    #
    # Exit:
    # following bar open
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


    # Require actual consecutive 4-hour bars

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
    # FUTURE RETURN
    # --------------------------------------------------

    df["future_return"] = (
        df["exit_open"]
        / df["entry_open"]
        - 1
    )


    # --------------------------------------------------
    # MOMENTUM DIRECTION
    # --------------------------------------------------

    df["direction"] = np.where(
        df["positive_shock"],
        1,
        np.where(
            df["negative_shock"],
            -1,
            0
        )
    )


    # --------------------------------------------------
    # MOMENTUM RETURNS
    # --------------------------------------------------

    df["gross_momentum_return"] = (
        df["direction"]
        * df["future_return"]
    )

    df["net_momentum_return"] = (
        df["gross_momentum_return"]
        - ROUND_TRIP_COST
    )


    # --------------------------------------------------
    # VALID EVENTS
    # --------------------------------------------------

    events = df[
        df["shock"]
        & df["valid_execution"]
        & df["future_return"].notna()
    ].copy()

    events["year"] = (
        events.index.year
    )

    events["asset"] = asset


    print(
        f"Total shock events: {len(events)}"
    )

    print(
        f"Positive shocks: "
        f"{int(events['positive_shock'].sum())}"
    )

    print(
        f"Negative shocks: "
        f"{int(events['negative_shock'].sum())}"
    )


    # --------------------------------------------------
    # OVERALL DIRECTION RESULTS
    # --------------------------------------------------

    print()
    print("OVERALL BY DIRECTION")

    for label, mask in [

        (
            "POSITIVE SHOCK",
            events["positive_shock"]
        ),

        (
            "NEGATIVE SHOCK",
            events["negative_shock"]
        )

    ]:

        subset = events[mask]

        if subset.empty:
            continue

        print()
        print(label)

        print(
            f"Events: {len(subset)}"
        )

        print(
            "Average net return: "
            f"{subset['net_momentum_return'].mean() * 100:.3f}%"
        )

        print(
            "Median net return:  "
            f"{subset['net_momentum_return'].median() * 100:.3f}%"
        )

        print(
            "Continuation rate:  "
            f"{(subset['gross_momentum_return'] > 0).mean() * 100:.1f}%"
        )

        print(
            "Net profitable:     "
            f"{(subset['net_momentum_return'] > 0).mean() * 100:.1f}%"
        )


    # --------------------------------------------------
    # YEAR × DIRECTION
    # --------------------------------------------------

    print()
    print("YEAR-BY-YEAR BY DIRECTION")
    print("-" * 70)


    for year in sorted(
        events["year"].unique()
    ):

        print()
        print(f"{year}")

        year_data = events[
            events["year"] == year
        ]


        for label, mask_column in [

            (
                "POSITIVE",
                "positive_shock"
            ),

            (
                "NEGATIVE",
                "negative_shock"
            )

        ]:

            subset = year_data[
                year_data[mask_column]
            ]

            if subset.empty:

                print(
                    f"  {label}: 0 events"
                )

                continue


            avg_net = (
                subset["net_momentum_return"]
                .mean()
                * 100
            )

            median_net = (
                subset["net_momentum_return"]
                .median()
                * 100
            )

            continuation = (
                (
                    subset[
                        "gross_momentum_return"
                    ] > 0
                )
                .mean()
                * 100
            )

            profitable = (
                (
                    subset[
                        "net_momentum_return"
                    ] > 0
                )
                .mean()
                * 100
            )


            print(
                f"  {label}: "
                f"{len(subset)} events | "
                f"avg net {avg_net:.3f}% | "
                f"median {median_net:.3f}% | "
                f"continuation {continuation:.1f}% | "
                f"net profitable {profitable:.1f}%"
            )


    return events


# --------------------------------------------------
# RUN BTC AND ETH
# --------------------------------------------------

btc = analyze_asset("BTC")
eth = analyze_asset("ETH")


# --------------------------------------------------
# SPECIFIC SUBGROUPS THAT EMERGED FROM THE
# PREVIOUS BROAD TEST
# --------------------------------------------------

print()
print("=" * 70)
print("PROMISING SUBGROUP CHECK")
print("=" * 70)


btc_positive = btc[
    btc["positive_shock"]
].copy()

eth_negative = eth[
    eth["negative_shock"]
].copy()


def subgroup_summary(
    data,
    name
):

    print()
    print(name)
    print("-" * 70)

    print(
        f"Total events: {len(data)}"
    )

    print(
        "Overall average net: "
        f"{data['net_momentum_return'].mean() * 100:.3f}%"
    )

    print(
        "Overall median net:  "
        f"{data['net_momentum_return'].median() * 100:.3f}%"
    )

    print(
        "Overall profitable:  "
        f"{(data['net_momentum_return'] > 0).mean() * 100:.1f}%"
    )

    print()

    positive_years = 0

    for year in sorted(
        data["year"].unique()
    ):

        y = data[
            data["year"] == year
        ]

        avg_net = (
            y["net_momentum_return"]
            .mean()
        )

        if avg_net > 0:
            positive_years += 1

        print(
            f"{year}: "
            f"{len(y)} events | "
            f"avg net {avg_net * 100:.3f}% | "
            f"median "
            f"{y['net_momentum_return'].median() * 100:.3f}% | "
            f"profitable "
            f"{(y['net_momentum_return'] > 0).mean() * 100:.1f}%"
        )


    total_years = (
        data["year"]
        .nunique()
    )

    print()

    print(
        "Years with positive average net return: "
        f"{positive_years}/{total_years}"
    )


subgroup_summary(
    btc_positive,
    "BTC POSITIVE SHOCK MOMENTUM"
)

subgroup_summary(
    eth_negative,
    "ETH NEGATIVE SHOCK MOMENTUM"
)


# --------------------------------------------------
# FINAL REMINDER
# --------------------------------------------------

print()
print("=" * 70)

print(
    "This is still a 2021-2024 development analysis."
)

print(
    "The subgroups were selected because they emerged "
    "from the previous broad shock test."
)

print(
    "Therefore, their performance here is NOT independent "
    "out-of-sample evidence."
)

print(
    "Do not modify the 2-sigma threshold or 20-bar lookback "
    "based on these results."
)

print(
    "2026 remains untouched."
)

print("=" * 70)
