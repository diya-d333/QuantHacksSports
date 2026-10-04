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

    lines = ("fear_greed_score",)

    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("fear_greed_score", "fear_greed_score"),
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

print(f"Shared 4-hour bars: {len(common_times)}")

if len(common_times) == 0:
    raise ValueError("No shared timestamps were found between the four assets.")


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

cerebro.addsizer(bt.sizers.FixedSize, stake=0.01)


# --------------------------------------------------
# Add basic transaction cost
#
# This is a DEVELOPMENT assumption, not a final
# estimate of real trading costs.
# --------------------------------------------------

cerebro.broker.setcommission(commission=0.001)


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