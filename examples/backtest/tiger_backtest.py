from pathlib import Path
import sys

import backtrader as bt
import pandas as pd


# --------------------------------------------------
# Make project folders importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BACKTEST_DIR.parent

sys.path.insert(0, str(BACKTEST_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))


# --------------------------------------------------
# Import our TigerData loader and strategy
# --------------------------------------------------

from load_tiger_data import load_market_data
from behavioral_mean_reversion import BehavioralMeanReversionStrategy


# --------------------------------------------------
# Custom Backtrader feed
# Adds Fear & Greed to normal OHLCV data
# --------------------------------------------------

class TigerPandasData(bt.feeds.PandasData):

    lines = (
        "fear_greed_score",
        "range_probability",
        "snowflake_range",
        "return_4h",
        "large_drop",
    )

    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("fear_greed_score", "fear_greed_score"),
        ("range_probability", "range_probability"),
        ("snowflake_range", "snowflake_range"),
        ("return_4h", "return_4h"),
        ("large_drop", "large_drop"),
    )


# --------------------------------------------------
# Load development data
# IMPORTANT: We are NOT using 2026 OOS data
# --------------------------------------------------

START_DATE = "2025-01-01T00:00:00Z"
END_DATE = "2026-01-01T00:00:00Z"

print("Loading 2025 development data...")

btc = load_market_data("BTC", START_DATE, END_DATE)
eth = load_market_data("ETH", START_DATE, END_DATE)
xrp = load_market_data("XRP", START_DATE, END_DATE)
sol = load_market_data("SOL", START_DATE, END_DATE)

def prepare_market_features(frame):
    frame = frame.sort_index().copy()

    consecutive = (
        frame.index.to_series()
        .diff()
        .eq(pd.Timedelta(hours=4))
    )

    # Missing intervals are not treated as four-hour returns.
    frame["return_4h"] = (
        frame["close"]
        .pct_change(fill_method=None)
        .where(consecutive)
    )

    # Preserve the current strategy's 20-return volatility rule.
    volatility = (
        frame["return_4h"]
        .rolling(window=20, min_periods=20)
        .std(ddof=0)
    )

    frame["large_drop"] = (
        frame["return_4h"] < -2.0 * volatility
    ).astype(float)

    return frame


btc = prepare_market_features(btc)
eth = prepare_market_features(eth)
xrp = prepare_market_features(xrp)
sol = prepare_market_features(sol)

# Load regime predictions generated inside Snowflake.
prediction_file = (
    PROJECT_ROOT
    / "regime_model_output"
    / "regime_predictions_adx_4h.csv"
)

predictions = pd.read_csv(prediction_file)
predictions.columns = predictions.columns.str.strip().str.upper()

predictions["TIME"] = pd.to_datetime(predictions["TIME"], utc=True)
predictions = predictions.loc[
    predictions["DATASET_ROLE"].eq("REGIME_VALIDATION")
].copy()

if predictions.duplicated(["ASSET", "TIME"]).any():
    raise ValueError("Duplicate Snowflake asset/timestamp predictions")


def attach_regime(frame, asset):
    asset_predictions = (
        predictions.loc[predictions["ASSET"].eq(asset)]
        .set_index("TIME")
    )

    result = frame.copy()
    result["range_probability"] = asset_predictions[
        "RANGE_PROBABILITY"
    ].reindex(result.index)

    # Use the model's saved state label for the range filter.
    result["snowflake_range"] = (
        asset_predictions["REGIME"]
        .reindex(result.index)
        .eq("RANGE_CANDIDATE")
        .astype(float)
    )

    print(
        f"{asset}: "
        f"{result['range_probability'].notna().sum()} "
        f"bars with Snowflake predictions"
    )
    return result


btc = attach_regime(btc, "BTC")
eth = attach_regime(eth, "ETH")

# Confirmation feeds do not have BTC/ETH regime predictions.
for confirmation_frame in [xrp, sol]:
    confirmation_frame["range_probability"] = float("nan")
    confirmation_frame["snowflake_range"] = 0.0


# --------------------------------------------------
# Keep only timestamps shared by all four assets
# --------------------------------------------------

common_times = (
    btc.index
    .intersection(eth.index)
    .intersection(xrp.index)
    .intersection(sol.index)
)

btc = btc.loc[common_times].copy()
eth = eth.loc[common_times].copy()
xrp = xrp.loc[common_times].copy()
sol = sol.loc[common_times].copy()

# Require consecutive candles for a valid 4-hour entry signal.
valid_4h = (
    pd.Series(common_times, index=common_times)
    .diff()
    .eq(pd.Timedelta(hours=4))
)

shared_clock = pd.Series(common_times, index=common_times)

next_open_is_signal_end = (
    shared_clock.shift(-1) - shared_clock
).eq(pd.Timedelta(hours=4))

entry_timing_valid = valid_4h & next_open_is_signal_end

for frame in [btc, eth]:
    frame.loc[~entry_timing_valid, "snowflake_range"] = 0.0

print(f"Shared 4-hour bars: {len(common_times)}")

if len(common_times) == 0:
    raise ValueError("No shared timestamps were found between the four assets.")

shared_times = pd.Series(common_times).sort_values()
gaps = shared_times.diff().dropna()

print("Shared timeline gaps:")
print(gaps.value_counts().sort_index().to_string())
print(f"First shared bar: {shared_times.iloc[0]}")
print(f"Last shared bar:  {shared_times.iloc[-1]}")


# --------------------------------------------------
# Create Backtrader
# --------------------------------------------------

cerebro = bt.Cerebro()

STARTING_CASH = 100000

cerebro.broker.setcash(STARTING_CASH)


# --------------------------------------------------
# Add BTC, ETH, XRP, SOL in EXACTLY this order
# --------------------------------------------------

btc_feed = TigerPandasData(
    dataname=btc,
    timeframe=bt.TimeFrame.Minutes,
    compression=240,
)

eth_feed = TigerPandasData(
    dataname=eth,
    timeframe=bt.TimeFrame.Minutes,
    compression=240,
)

xrp_feed = TigerPandasData(
    dataname=xrp,
    timeframe=bt.TimeFrame.Minutes,
    compression=240,
)

sol_feed = TigerPandasData(
    dataname=sol,
    timeframe=bt.TimeFrame.Minutes,
    compression=240,
)

cerebro.adddata(btc_feed, name="BTC")
cerebro.adddata(eth_feed, name="ETH")
cerebro.adddata(xrp_feed, name="XRP")
cerebro.adddata(sol_feed, name="SOL")


# --------------------------------------------------
# Add our strategy
# --------------------------------------------------

cerebro.addstrategy(BehavioralMeanReversionStrategy)


# --------------------------------------------------
# Basic position sizing
#
# Temporary development setting:
# use a small fixed size while we verify that the
# strategy executes correctly.
# --------------------------------------------------

class ResearchCashSizer(bt.Sizer):
    params = (("cash_fraction", 0.10),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            return self.broker.getposition(data).size

        price = float(data.close[0])
        if price <= 0:
            return 0.0

        budget = cash * self.p.cash_fraction
        return budget / (
            price * (1 + FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
        )


cerebro.addsizer(ResearchCashSizer, cash_fraction=0.10)


# --------------------------------------------------
# Add basic transaction cost
#
# This is a DEVELOPMENT assumption, not a final
# estimate of real trading costs.
# --------------------------------------------------

# Illustrative assumptions, charged on entry and exit.
FEE_PER_SIDE = 0.0005       # 0.05%
SLIPPAGE_PER_SIDE = 0.0002  # 0.02%

cerebro.broker.setcommission(
    commission=FEE_PER_SIDE,
    stocklike=True,
    percabs=True,
)

cerebro.broker.set_slippage_perc(
    perc=SLIPPAGE_PER_SIDE,
    slip_open=True,
)


# --------------------------------------------------
# Add analyzers
# --------------------------------------------------

cerebro.addanalyzer(
    bt.analyzers.TradeAnalyzer,
    _name="trades",
)

cerebro.addanalyzer(
    bt.analyzers.DrawDown,
    _name="drawdown",
)


# --------------------------------------------------
# Run backtest
# --------------------------------------------------

print(f"Starting portfolio value: ${cerebro.broker.getvalue():,.2f}")

results = cerebro.run()

strategy = results[0]

final_value = cerebro.broker.getvalue()

print(f"Final portfolio value:   ${final_value:,.2f}")
print(f"Net P&L:                 ${final_value - STARTING_CASH:,.2f}")


# --------------------------------------------------
# Basic results
# --------------------------------------------------

trade_analysis = strategy.analyzers.trades.get_analysis()
drawdown_analysis = strategy.analyzers.drawdown.get_analysis()

total_trades = trade_analysis.get("total", {}).get("closed", 0)
max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0)

print(f"Closed trades:           {total_trades}")
print(f"Maximum drawdown:        {max_drawdown:.2f}%")