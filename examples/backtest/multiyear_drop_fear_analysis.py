from pathlib import Path
import sys

import pandas as pd


# --------------------------------------------------
# SETUP
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
BACKTEST_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BACKTEST_DIR))

from load_tiger_data import load_market_data


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

START_DATE = "2021-01-01T00:00:00Z"
END_DATE = "2025-01-01T00:00:00Z"

FEATURE_FILE = ROOT / "regime_features_4h.csv"

DROP_ZSCORE_THRESHOLD = -2
FEAR_THRESHOLD = 25

# Same illustrative costs as teammate:
# 0.05% fee + 0.02% slippage PER SIDE
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = 2 * (
    FEE_PER_SIDE + SLIPPAGE_PER_SIDE
)


# --------------------------------------------------
# LOAD TEAMMATE'S FEATURE DATA
# --------------------------------------------------

print("Loading teammate feature data...")
print()

features = pd.read_csv(FEATURE_FILE)

features["TIME"] = pd.to_datetime(
    features["TIME"],
    utc=True
)

# Development period only
features = features[
    (features["TIME"].dt.year >= 2021)
    & (features["TIME"].dt.year <= 2024)
].copy()


# --------------------------------------------------
# RECREATE EXACT DROP_FEAR SIGNAL
# --------------------------------------------------

ready = (
    features["FEATURES_READY"]
    .astype(str)
    .str.lower()
    .eq("true")
)

drop = (
    ready
    & features["RETURN_1"].lt(0)
    & features["DROP_ZSCORE"].le(
        DROP_ZSCORE_THRESHOLD
    )
)

fear = (
    features["FEAR_GREED_SCORE"].le(
        FEAR_THRESHOLD
    )
)

features["DROP_FEAR"] = (
    drop & fear
)


# --------------------------------------------------
# TEST BTC AND ETH
# --------------------------------------------------

all_trades = []

for asset in ["BTC", "ETH"]:

    print("=" * 60)
    print(asset)
    print("=" * 60)

    # ----------------------------------------------
    # Asset feature data
    # ----------------------------------------------

    asset_features = features[
        features["ASSET"] == asset
    ].copy()

    asset_features = (
        asset_features
        .sort_values("TIME")
        .set_index("TIME")
    )


    # ----------------------------------------------
    # Load exact TigerData OHLC
    # ----------------------------------------------

    prices = load_market_data(
        asset,
        START_DATE,
        END_DATE
    ).copy()

    prices = prices.sort_index()


    # ----------------------------------------------
    # Match feature timestamps to price timestamps
    # ----------------------------------------------

    common_times = (
        asset_features.index
        .intersection(prices.index)
    )

    f = asset_features.loc[
        common_times
    ].copy()

    p = prices.loc[
        common_times
    ].copy()


    print(
        f"Matched bars: {len(common_times)}"
    )

    print(
        f"DROP_FEAR signals: "
        f"{int(f['DROP_FEAR'].sum())}"
    )


    # ----------------------------------------------
    # EXECUTION
    # ----------------------------------------------
    #
    # Signal known after bar t closes.
    #
    # Entry = next bar open
    # Exit  = following bar open
    #
    # This creates a 4-hour holding period.
    #
    # We also require timestamps to be exactly
    # consecutive 4-hour bars so we do not
    # accidentally hold through a data gap.
    # ----------------------------------------------

    signal_positions = [
        i
        for i in range(len(f))
        if f["DROP_FEAR"].iloc[i]
    ]


    asset_trades = []

    last_exit_position = -1


    for i in signal_positions:

        # Need two future bars
        if i + 2 >= len(p):
            continue

        signal_time = p.index[i]
        entry_time = p.index[i + 1]
        exit_time = p.index[i + 2]


        # ------------------------------------------
        # Require continuous 4-hour bars
        # ------------------------------------------

        if (
            entry_time - signal_time
            != pd.Timedelta(hours=4)
        ):
            continue

        if (
            exit_time - entry_time
            != pd.Timedelta(hours=4)
        ):
            continue


        # ------------------------------------------
        # Do not overlap positions
        # ------------------------------------------

        if i + 1 <= last_exit_position:
            continue


        entry_price = float(
            p["open"].iloc[i + 1]
        )

        exit_price = float(
            p["open"].iloc[i + 2]
        )


        # ------------------------------------------
        # RETURNS
        # ------------------------------------------

        gross_return = (
            exit_price / entry_price - 1
        )

        net_return = (
            gross_return - ROUND_TRIP_COST
        )


        asset_trades.append(
            {
                "ASSET": asset,
                "SIGNAL_TIME": signal_time,
                "ENTRY_TIME": entry_time,
                "EXIT_TIME": exit_time,
                "YEAR": signal_time.year,
                "DROP_ZSCORE":
                    f["DROP_ZSCORE"].iloc[i],
                "FEAR_GREED_SCORE":
                    f["FEAR_GREED_SCORE"].iloc[i],
                "ENTRY_PRICE": entry_price,
                "EXIT_PRICE": exit_price,
                "GROSS_RETURN":
                    gross_return,
                "NET_RETURN":
                    net_return,
            }
        )


        last_exit_position = i + 2


    trades = pd.DataFrame(
        asset_trades
    )


    if trades.empty:

        print("No trades.")
        print()
        continue


    all_trades.append(trades)


    # ----------------------------------------------
    # OVERALL RESULTS
    # ----------------------------------------------

    print(
        f"Executed trades: {len(trades)}"
    )

    print(
        "Average gross trade return: "
        f"{trades['GROSS_RETURN'].mean() * 100:.3f}%"
    )

    print(
        "Average net trade return:   "
        f"{trades['NET_RETURN'].mean() * 100:.3f}%"
    )

    print(
        "Median net trade return:    "
        f"{trades['NET_RETURN'].median() * 100:.3f}%"
    )

    print(
        "Net win rate:               "
        f"{(trades['NET_RETURN'] > 0).mean() * 100:.1f}%"
    )


    cumulative = (
        (1 + trades["NET_RETURN"])
        .prod() - 1
    )

    print(
        "Cumulative net trade return:"
        f" {cumulative * 100:.3f}%"
    )


    # ----------------------------------------------
    # YEAR BY YEAR
    # ----------------------------------------------

    print()
    print("Year-by-year:")

    for year in sorted(
        trades["YEAR"].unique()
    ):

        y = trades[
            trades["YEAR"] == year
        ]

        y_cumulative = (
            (1 + y["NET_RETURN"])
            .prod() - 1
        )

        print(
            f"{year}: "
            f"{len(y)} trades | "
            f"avg net "
            f"{y['NET_RETURN'].mean() * 100:.3f}% | "
            f"median "
            f"{y['NET_RETURN'].median() * 100:.3f}% | "
            f"win rate "
            f"{(y['NET_RETURN'] > 0).mean() * 100:.1f}% | "
            f"cumulative "
            f"{y_cumulative * 100:.3f}%"
        )

    print()


# --------------------------------------------------
# COMBINED RESULTS
# --------------------------------------------------

print("=" * 60)
print("COMBINED BTC + ETH")
print("=" * 60)

if all_trades:

    combined = pd.concat(
        all_trades,
        ignore_index=True
    )

    combined = combined.sort_values(
        "SIGNAL_TIME"
    )


    print(
        f"Total trades: {len(combined)}"
    )

    print(
        "Average net trade return: "
        f"{combined['NET_RETURN'].mean() * 100:.3f}%"
    )

    print(
        "Median net trade return:  "
        f"{combined['NET_RETURN'].median() * 100:.3f}%"
    )

    print(
        "Net win rate:             "
        f"{(combined['NET_RETURN'] > 0).mean() * 100:.1f}%"
    )


    cumulative = (
        (1 + combined["NET_RETURN"])
        .prod() - 1
    )

    print(
        "Cumulative net trade return: "
        f"{cumulative * 100:.3f}%"
    )


    print()
    print("Combined year-by-year:")


    for year in sorted(
        combined["YEAR"].unique()
    ):

        y = combined[
            combined["YEAR"] == year
        ]

        y_cumulative = (
            (1 + y["NET_RETURN"])
            .prod() - 1
        )

        print(
            f"{year}: "
            f"{len(y)} trades | "
            f"avg net "
            f"{y['NET_RETURN'].mean() * 100:.3f}% | "
            f"win rate "
            f"{(y['NET_RETURN'] > 0).mean() * 100:.1f}% | "
            f"cumulative "
            f"{y_cumulative * 100:.3f}%"
        )


print()
print("=" * 60)

print(
    "Development test only: 2021-2024."
)

print(
    "2025 was not used to modify this test."
)

print(
    "2026 remains untouched final OOS."
)

print("=" * 60)
