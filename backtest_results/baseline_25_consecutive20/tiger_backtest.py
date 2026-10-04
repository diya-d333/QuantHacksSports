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
    # Calculate volatility from the last 20 valid four-hour returns.
    # Returns spanning gaps remain excluded.
    volatility = (
        frame["return_4h"]
        .rolling(window=20, min_periods=20)
        .std(ddof=0)
    )

    valid_returns = frame["return_4h"].notna()
    volatility_ready = volatility.notna()

    print(
        f"  Total candles: {len(frame)} | "
        f"Valid 4-hour returns: {valid_returns.sum()} | "
        f"Volatility ready: {volatility_ready.sum()} | "
        f"Valid returns blocked by volatility warm-up: "
        f"{(valid_returns & ~volatility_ready).sum()}"
    )

    frame["large_drop"] = (
        frame["return_4h"] < -2.0 * volatility
    ).astype(float)

    return frame


print("BTC feature coverage:")
btc = prepare_market_features(btc)

print("ETH feature coverage:")
eth = prepare_market_features(eth)

print("XRP feature coverage:")
xrp = prepare_market_features(xrp)

print("SOL feature coverage:")
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

# Trading timeline: timestamps available for both BTC and ETH.
common_times = btc.index.intersection(eth.index).sort_values()

if common_times.empty:
    raise ValueError("No shared BTC/ETH timestamps.")

btc = btc.loc[common_times].copy()
eth = eth.loc[common_times].copy()

# Align confirmation observations to the trading timeline.
# Missing observations remain NaN; do not forward-fill returns.
xrp = xrp.reindex(common_times).copy()
sol = sol.reindex(common_times).copy()

confirmation_available = (
    xrp["return_4h"].notna()
    & sol["return_4h"].notna()
)

print(f"BTC/ETH trading bars: {len(common_times)}")
print(
    "Bars with both XRP/SOL returns: "
    f"{confirmation_available.sum()}"
)

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

cerebro.addstrategy(
    BehavioralMeanReversionStrategy,
    require_sentiment=True,
    require_market_confirmation=False,
    sentiment_threshold=25,
)


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
FEE_PER_SIDE = 0.0010       # 0.10% per side
SLIPPAGE_PER_SIDE = 0.0005  # 0.05% per side

cerebro.broker.setcommission(
    commission=FEE_PER_SIDE,
    stocklike=True,
    percabs=True,
)

cerebro.broker.set_slippage_perc(
    perc=SLIPPAGE_PER_SIDE,
    slip_open=True,
)

class EquityRecorder(bt.Analyzer):
    def start(self):
        self.rows = []

    def next(self):
        # Input timestamps mark candle starts.
        bar_end = (
            pd.Timestamp(self.strategy.datas[0].datetime.datetime(0))
            .tz_localize("UTC")
            + pd.Timedelta(hours=4)
        )

        self.rows.append({
            "time_utc": bar_end.isoformat(),
            "portfolio_value": self.strategy.broker.getvalue(),
            "cash": self.strategy.broker.getcash(),
        })

    def get_analysis(self):
        return self.rows


cerebro.addanalyzer(EquityRecorder, _name="equity")

class TradeRecorder(bt.Analyzer):
    def start(self):
        self.entries = {}
        self.rows = []

    def notify_order(self, order):
        if order.status != order.Completed:
            return

        asset = order.data._name
        fill = order.executed
        fill_time = pd.Timestamp(
            bt.num2date(fill.dt)
        ).tz_localize("UTC")

        if order.isbuy():
            self.entries[asset] = {
                "time": fill_time,
                "price": fill.price,
                "units": fill.size,
                "commission": fill.comm,
            }

        elif order.issell():
            entry = self.entries.pop(asset, None)
            if entry is None:
                raise RuntimeError(f"{asset}: exit without recorded entry")

            gross_pnl = (
                fill.price - entry["price"]
            ) * entry["units"]

            commissions = entry["commission"] + fill.comm

            self.rows.append({
                "asset": asset,
                "entry_time_utc": entry["time"].isoformat(),
                "exit_time_utc": fill_time.isoformat(),
                "entry_price": entry["price"],
                "exit_price": fill.price,
                "units": entry["units"],
                "hours_held": (
                    fill_time - entry["time"]
                ).total_seconds() / 3600,
                "gross_pnl": gross_pnl,
                "commissions": commissions,
                "net_pnl": gross_pnl - commissions,
            })

    def get_analysis(self):
        return self.rows


cerebro.addanalyzer(TradeRecorder, _name="trade_log")

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

output_dir = PROJECT_ROOT / "backtest_results"
output_dir.mkdir(parents=True, exist_ok=True)

equity = pd.DataFrame(strategy.analyzers.equity.get_analysis())
equity.to_csv(
    output_dir / "mean_reversion_2025_equity.csv",
    index=False,
)

print(f"Saved {len(equity)} equity observations.")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

equity["time_utc"] = pd.to_datetime(equity["time_utc"], utc=True)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    equity["time_utc"],
    equity["portfolio_value"],
    label="Mean reversion strategy",
    color="navy",
)

ax.set_title(
    "2025 Development Equity Curve — January–December\n"
    "Fear & Greed required; XRP/SOL informational"
)
ax.set_xlabel("Date (UTC)")
ax.set_ylabel("Portfolio value (USD)")
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
ax.set_ylim(99000, 101000)
ax.grid(alpha=0.3)
ax.legend()

fig.autofmt_xdate()
fig.tight_layout()

chart_file = output_dir / "mean_reversion_2025_equity.png"
fig.savefig(chart_file, dpi=200)
plt.close(fig)

print(f"Saved equity curve: {chart_file}")

trade_log = pd.DataFrame(
    strategy.analyzers.trade_log.get_analysis()
)

trade_file = output_dir / "mean_reversion_2025_trades.csv"
trade_log.to_csv(trade_file, index=False)

print(f"Saved trade log: {trade_file}")