import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETUP
# ============================================================

BACKTEST_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(BACKTEST_DIR))

from load_tiger_data import load_market_data


# ============================================================
# FINAL FROZEN OOS SETTINGS
# ============================================================

# December 2025 is loaded ONLY for indicator warm-up.
LOAD_START = "2025-12-01T00:00:00Z"

# Trades/signals must occur in 2026.
OOS_START = pd.Timestamp("2026-01-01T00:00:00Z")
OOS_END = pd.Timestamp("2027-01-01T00:00:00Z")

INITIAL_CAPITAL = 100_000.0

LOOKBACK = 20
SHOCK_ZSCORE = 2.0

ALLOCATION_PER_TRADE = 0.10

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002


# ============================================================
# FROZEN SIGNAL CREATION
# ============================================================

def prepare_asset(asset):

    df = load_market_data(
        asset,
        LOAD_START,
        OOS_END.isoformat()
    ).copy()

    df = df.sort_index()

    df["return_1"] = (
        df["close"].pct_change()
    )

    # --------------------------------------------------------
    # Historical distribution using ONLY prior observations.
    # --------------------------------------------------------

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

    df["shock_zscore"] = (
        (
            df["return_1"]
            - df["rolling_mean"]
        )
        / df["rolling_std"]
    )

    # --------------------------------------------------------
    # FROZEN ASSET-DIRECTION RULES
    # --------------------------------------------------------

    if asset == "BTC":

        # BTC positive shock -> LONG
        df["signal"] = (
            df["shock_zscore"]
            >= SHOCK_ZSCORE
        )

        direction = 1

    elif asset == "ETH":

        # ETH negative shock -> SHORT
        df["signal"] = (
            df["shock_zscore"]
            <= -SHOCK_ZSCORE
        )

        direction = -1

    else:

        raise ValueError(
            f"Unsupported asset: {asset}"
        )

    # --------------------------------------------------------
    # TRADE CANDIDATES
    # --------------------------------------------------------

    candidates = []

    index = df.index

    for i in range(len(df) - 2):

        signal_time = index[i]

        # FINAL OOS FILTER:
        # No signal before 2026 is allowed.
        if not (
            OOS_START
            <= signal_time
            < OOS_END
        ):
            continue

        if not bool(df["signal"].iloc[i]):
            continue

        entry_time = index[i + 1]
        exit_time = index[i + 2]

        # Entry and exit must also remain inside OOS.
        if entry_time >= OOS_END:
            continue

        if exit_time >= OOS_END:
            continue

        # Require exact consecutive 4-hour bars.
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

        candidates.append(
            {
                "asset": asset,
                "direction": direction,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_open": float(
                    df["open"].iloc[i + 1]
                ),
                "shock_zscore": float(
                    df["shock_zscore"].iloc[i]
                ),
            }
        )

    return df, candidates


# ============================================================
# LOAD FINAL OOS DATA
# ============================================================

print()
print("=" * 72)
print("FINAL FROZEN SHOCK STRATEGY — 2026 OUT-OF-SAMPLE")
print("=" * 72)

print()
print("NO STRATEGY PARAMETERS ARE BEING FIT OR CHANGED.")
print("December 2025 is used only as indicator warm-up.")
print("All signals and trades are restricted to 2026.")

print()
print("Loading BTC...")
btc_df, btc_candidates = prepare_asset("BTC")

print("Loading ETH...")
eth_df, eth_candidates = prepare_asset("ETH")


candidates = sorted(
    btc_candidates + eth_candidates,
    key=lambda x: (
        x["entry_time"],
        x["asset"]
    )
)


print()
print(
    f"BTC OOS candidates: {len(btc_candidates)}"
)

print(
    f"ETH OOS candidates: {len(eth_candidates)}"
)

print(
    f"Total OOS candidates: {len(candidates)}"
)


# ============================================================
# PORTFOLIO STATE
# ============================================================

cash = INITIAL_CAPITAL

open_positions = {}

closed_trades = []

equity_records = []


price_data = {
    "BTC": btc_df,
    "ETH": eth_df,
}


# Only use actual 2026 timestamps for the portfolio timeline.
timeline = sorted(
    t
    for t in (
        set(btc_df.index)
        | set(eth_df.index)
    )
    if OOS_START <= t < OOS_END
)


candidates_by_time = {}

for trade in candidates:

    candidates_by_time.setdefault(
        trade["entry_time"],
        []
    ).append(trade)


# ============================================================
# EQUITY FUNCTION
# ============================================================

def calculate_equity(time):

    equity_value = cash

    for asset, position in open_positions.items():

        df = price_data[asset]

        if time not in df.index:
            continue

        mark_price = float(
            df.loc[time, "open"]
        )

        if position["direction"] == 1:

            position_value = (
                position["units"]
                * mark_price
            )

            equity_value += position_value

        else:

            unrealized_pnl = (
                position["units"]
                * (
                    position["entry_exec_price"]
                    - mark_price
                )
            )

            position_value = (
                position["allocated_capital"]
                + unrealized_pnl
            )

            equity_value += position_value

    return equity_value


# ============================================================
# RUN FINAL OOS PORTFOLIO
# ============================================================

for current_time in timeline:

    # ========================================================
    # 1. EXIT EXISTING POSITIONS
    # ========================================================

    assets_to_close = []

    for asset, position in list(
        open_positions.items()
    ):

        if current_time < position["exit_time"]:
            continue

        df = price_data[asset]

        if current_time not in df.index:
            continue

        raw_exit = float(
            df.loc[current_time, "open"]
        )

        # ----------------------------------------------------
        # LONG EXIT
        # ----------------------------------------------------

        if position["direction"] == 1:

            exit_exec_price = (
                raw_exit
                * (1 - SLIPPAGE_PER_SIDE)
            )

            exit_notional = (
                position["units"]
                * exit_exec_price
            )

            exit_fee = (
                exit_notional
                * FEE_PER_SIDE
            )

            exit_cash = (
                exit_notional
                - exit_fee
            )

            cash += exit_cash

        # ----------------------------------------------------
        # SHORT EXIT
        # ----------------------------------------------------

        else:

            exit_exec_price = (
                raw_exit
                * (1 + SLIPPAGE_PER_SIDE)
            )

            price_pnl = (
                position["units"]
                * (
                    position["entry_exec_price"]
                    - exit_exec_price
                )
            )

            exit_notional = (
                position["units"]
                * exit_exec_price
            )

            exit_fee = (
                exit_notional
                * FEE_PER_SIDE
            )

            exit_cash = (
                position["allocated_capital"]
                + price_pnl
                - exit_fee
            )

            cash += exit_cash

        # ----------------------------------------------------
        # REALIZED NET P&L
        # ----------------------------------------------------

        net_pnl = (
            exit_cash
            - position["allocated_capital"]
            - position["entry_fee"]
        )

        net_return = (
            net_pnl
            / position["allocated_capital"]
        )

        closed_trades.append(
            {
                "asset": asset,
                "direction":
                    "LONG"
                    if position["direction"] == 1
                    else "SHORT",
                "signal_time":
                    position["signal_time"],
                "entry_time":
                    position["entry_time"],
                "exit_time":
                    current_time,
                "entry_price":
                    position["entry_exec_price"],
                "exit_price":
                    exit_exec_price,
                "allocated_capital":
                    position["allocated_capital"],
                "entry_fee":
                    position["entry_fee"],
                "exit_fee":
                    exit_fee,
                "total_fees":
                    position["entry_fee"]
                    + exit_fee,
                "net_pnl":
                    net_pnl,
                "net_return":
                    net_return,
                "shock_zscore":
                    position["shock_zscore"],
            }
        )

        assets_to_close.append(
            asset
        )

    for asset in assets_to_close:

        del open_positions[asset]


    # ========================================================
    # 2. EQUITY BEFORE ENTRIES
    # ========================================================

    current_equity = (
        calculate_equity(
            current_time
        )
    )


    # ========================================================
    # 3. ENTER NEW POSITIONS
    # ========================================================

    new_trades = candidates_by_time.get(
        current_time,
        []
    )

    for trade in new_trades:

        asset = trade["asset"]

        # One position per asset.
        if asset in open_positions:
            continue

        allocation = (
            current_equity
            * ALLOCATION_PER_TRADE
        )

        if allocation <= 0:
            continue

        raw_entry = trade["entry_open"]

        # Entry slippage.
        if trade["direction"] == 1:

            entry_exec_price = (
                raw_entry
                * (1 + SLIPPAGE_PER_SIDE)
            )

        else:

            entry_exec_price = (
                raw_entry
                * (1 - SLIPPAGE_PER_SIDE)
            )

        units = (
            allocation
            / entry_exec_price
        )

        entry_fee = (
            allocation
            * FEE_PER_SIDE
        )

        total_cash_required = (
            allocation
            + entry_fee
        )

        if total_cash_required > cash:
            continue

        cash -= total_cash_required

        open_positions[asset] = {
            **trade,
            "entry_exec_price":
                entry_exec_price,
            "allocated_capital":
                allocation,
            "entry_fee":
                entry_fee,
            "units":
                units,
        }


    # ========================================================
    # 4. RECORD EQUITY
    # ========================================================

    equity_records.append(
        {
            "time": current_time,
            "equity":
                calculate_equity(
                    current_time
                ),
        }
    )


# ============================================================
# RESULTS
# ============================================================

trades = pd.DataFrame(
    closed_trades
)

equity = pd.DataFrame(
    equity_records
).set_index(
    "time"
).sort_index()


print()
print("=" * 72)
print("FINAL 2026 OOS RESULTS")
print("=" * 72)


if trades.empty:

    print("No OOS trades executed.")
    raise SystemExit


final_equity = float(
    equity["equity"].iloc[-1]
)

portfolio_pnl = (
    final_equity
    - INITIAL_CAPITAL
)

portfolio_return = (
    final_equity
    / INITIAL_CAPITAL
    - 1
)

realized_trade_pnl = (
    trades["net_pnl"].sum()
)

reconciliation_difference = (
    portfolio_pnl
    - realized_trade_pnl
)


print(
    f"Starting capital:      "
    f"${INITIAL_CAPITAL:,.2f}"
)

print(
    f"Final equity:          "
    f"${final_equity:,.2f}"
)

print(
    f"OOS portfolio P&L:     "
    f"${portfolio_pnl:,.2f}"
)

print(
    f"OOS portfolio return:  "
    f"{portfolio_return * 100:.3f}%"
)

print(
    f"Closed OOS trades:     "
    f"{len(trades)}"
)


# ============================================================
# ACCOUNTING CHECK
# ============================================================

print()
print("=" * 72)
print("ACCOUNTING CHECK")
print("=" * 72)

print(
    f"Portfolio P&L:         "
    f"${portfolio_pnl:,.2f}"
)

print(
    f"Sum of trade P&L:      "
    f"${realized_trade_pnl:,.2f}"
)

print(
    f"Difference:            "
    f"${reconciliation_difference:,.6f}"
)

if abs(reconciliation_difference) < 0.01:

    print(
        "Accounting reconciliation: PASS"
    )

else:

    print(
        "Accounting reconciliation: FAIL"
    )


# ============================================================
# TRADE STATISTICS
# ============================================================

print()
print("=" * 72)
print("OOS TRADE STATISTICS")
print("=" * 72)

print(
    "Average trade return:  "
    f"{trades['net_return'].mean() * 100:.3f}%"
)

print(
    "Median trade return:   "
    f"{trades['net_return'].median() * 100:.3f}%"
)

print(
    "Trade win rate:        "
    f"{(trades['net_pnl'] > 0).mean() * 100:.1f}%"
)

print(
    "Total realized P&L:    "
    f"${realized_trade_pnl:,.2f}"
)

print(
    "Total fees paid:       "
    f"${trades['total_fees'].sum():,.2f}"
)


# ============================================================
# DAILY PERFORMANCE
# ============================================================

daily_equity = (
    equity["equity"]
    .resample("1D")
    .last()
    .dropna()
)

daily_returns = (
    daily_equity
    .pct_change()
    .dropna()
)


# ============================================================
# ANNUALIZED RETURN
# ============================================================

days = (
    daily_equity.index[-1]
    - daily_equity.index[0]
).days

years = (
    days / 365.25
)


if years > 0:

    annualized_return = (
        (
            final_equity
            / INITIAL_CAPITAL
        )
        ** (1 / years)
        - 1
    )

else:

    annualized_return = np.nan


# ============================================================
# ANNUALIZED VOLATILITY
# ============================================================

if len(daily_returns) > 1:

    annualized_volatility = (
        daily_returns.std(ddof=1)
        * np.sqrt(365)
    )

else:

    annualized_volatility = np.nan


# ============================================================
# SHARPE
# ============================================================

if (
    len(daily_returns) > 1
    and daily_returns.std(ddof=1) > 0
):

    sharpe = (
        daily_returns.mean()
        / daily_returns.std(ddof=1)
        * np.sqrt(365)
    )

else:

    sharpe = np.nan


# ============================================================
# MAX DRAWDOWN
# ============================================================

running_peak = (
    equity["equity"].cummax()
)

drawdown = (
    equity["equity"]
    / running_peak
    - 1
)

max_drawdown = (
    drawdown.min()
)


# ============================================================
# TURNOVER
# ============================================================

total_traded_notional = (
    trades["allocated_capital"].sum()
    * 2
)

average_equity = (
    equity["equity"].mean()
)

cumulative_turnover = (
    total_traded_notional
    / average_equity
)

annualized_turnover = (
    cumulative_turnover / years
    if years > 0
    else np.nan
)


# ============================================================
# REQUIRED FINAL OOS METRICS
# ============================================================

print()
print("=" * 72)
print("FINAL OOS PERFORMANCE METRICS")
print("=" * 72)

print(
    f"Annualized return:       "
    f"{annualized_return * 100:.3f}%"
)

print(
    f"Annualized volatility:   "
    f"{annualized_volatility * 100:.3f}%"
)

print(
    f"Sharpe ratio:            "
    f"{sharpe:.3f}"
)

print(
    f"Maximum drawdown:        "
    f"{max_drawdown * 100:.3f}%"
)

print(
    f"Cumulative turnover:     "
    f"{cumulative_turnover:.3f}x"
)

print(
    f"Annualized turnover:     "
    f"{annualized_turnover:.3f}x/year"
)


# ============================================================
# ASSET BREAKDOWN
# ============================================================

print()
print("=" * 72)
print("FINAL OOS ASSET BREAKDOWN")
print("=" * 72)

for asset in [
    "BTC",
    "ETH"
]:

    a = trades[
        trades["asset"] == asset
    ]

    if a.empty:
        continue

    print()
    print(asset)

    print(
        f"Trades: {len(a)}"
    )

    print(
        "Average trade return: "
        f"{a['net_return'].mean() * 100:.3f}%"
    )

    print(
        "Median trade return:  "
        f"{a['net_return'].median() * 100:.3f}%"
    )

    print(
        "Win rate:             "
        f"{(a['net_pnl'] > 0).mean() * 100:.1f}%"
    )

    print(
        "Net P&L:              "
        f"${a['net_pnl'].sum():,.2f}"
    )


# ============================================================
# SAVE FINAL OOS RESULTS
# ============================================================

OUTPUT_DIR = (
    ROOT / "backtest_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

trade_output = (
    OUTPUT_DIR
    / "final_shock_2026_oos_trades.csv"
)

equity_output = (
    OUTPUT_DIR
    / "final_shock_2026_oos_equity.csv"
)

trades.to_csv(
    trade_output,
    index=False
)

equity.to_csv(
    equity_output
)


print()
print("=" * 72)
print("FINAL OOS FILES SAVED")
print("=" * 72)

print(trade_output)
print(equity_output)


# ============================================================
# FINAL DECLARATION
# ============================================================

print()
print("=" * 72)

print(
    "FINAL 2026 OUT-OF-SAMPLE TEST COMPLETE."
)

print(
    "No strategy parameters were changed using 2026 data."
)

print(
    "These results must be reported whether positive or negative."
)

print("=" * 72)
