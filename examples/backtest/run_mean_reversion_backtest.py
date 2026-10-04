from datetime import timedelta
from pathlib import Path

import backtrader as bt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "backtest_ready"
OUTPUT_DIR = ROOT / "backtest_results"

STARTING_CASH = 100_000.0

VARIANTS = {
    "DROP_ONLY": "SIGNAL_DROP_ONLY",
    "DROP_FEAR": "SIGNAL_DROP_FEAR",
    "DROP_RANGE": "SIGNAL_DROP_RANGE",
    "FULL": "SIGNAL_FULL",
}

# Illustrative assumptions, not verified broker fees.
COST_CASES = {
    "ZERO_COST": (0.0, 0.0),
    "ILLUSTRATIVE_COST": (0.0005, 0.0002),
}

class EquityRecorder(bt.Analyzer):
    def start(self):
        self.rows = []

    def next(self):
        # Source timestamps mark the bar start.
        bar_end = (
            self.strategy.data.datetime.datetime(0)
            + timedelta(hours=4)
        )

        self.rows.append({
            "TIME_UTC": bar_end,
            "EQUITY": self.strategy.broker.getvalue(),
            "CASH": self.strategy.broker.getcash(),
        })

    def get_analysis(self):
        return self.rows

class SignalFeed(bt.feeds.PandasData):
    lines = ("entry_signal",)

    params = (
        ("datetime", None),
        ("open", "OPEN"),
        ("high", "HIGH"),
        ("low", "LOW"),
        ("close", "CLOSE"),
        ("volume", "VOLUME"),
        ("openinterest", -1),
        ("entry_signal", "ENTRY_SIGNAL"),
    )


class MeanReversionStrategy(bt.Strategy):
    params = (
        ("allocation", 0.10),
    )

    def __init__(self):
        self.pending_order = None
        self.cached_signal = False
        self.cached_bar_end = None

        self.entry_time = None
        self.entry_price = None
        self.entry_size = None
        self.exit_time = None
        self.exit_price = None

        self.closed_trades = []
        self.rejected_orders = 0
        self.skipped_gap_signals = 0
        self.submitted_entries = 0

    def next(self):
        # This callback represents the completed candle.
        # Source index is the candle START.
        start = self.data.datetime.datetime(0)

        self.cached_bar_end = start + timedelta(hours=4)
        self.cached_signal = bool(self.data.entry_signal[0])

    def next_open(self):
        # Only the current OPEN/time and cached prior signal
        # are used here. Never read current close or signal.
        now = self.data.datetime.datetime(0)

        if self.pending_order is not None:
            return

        if self.position:
            due = self.entry_time + timedelta(hours=4)

            if now >= due:
                self.pending_order = self.close()

            # Do not exit and re-enter at the same open.
            return

        if not self.cached_signal:
            return

        if now != self.cached_bar_end:
            # A stale signal must not enter after a weekend/gap.
            self.skipped_gap_signals += 1
            self.cached_signal = False
            return

        opening_price = float(self.data.open[0])
        budget = self.broker.getcash() * self.p.allocation

        # Fractional research units, not futures contracts.
        size = budget / opening_price

        self.pending_order = self.buy(size=size)
        self.submitted_entries += 1
        self.cached_signal = False

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            executed_time = bt.num2date(
                order.executed.dt
            ).replace(tzinfo=None)

            if order.isbuy():
                self.entry_time = executed_time
                self.entry_price = order.executed.price
                self.entry_size = order.executed.size
            else:
                self.exit_time = executed_time
                self.exit_price = order.executed.price

        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ]:
            self.rejected_orders += 1

        self.pending_order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        hours_held = (
            self.exit_time - self.entry_time
        ).total_seconds() / 3600.0

        entry_notional = self.entry_price * self.entry_size

        self.closed_trades.append({
            "ENTRY_TIME_UTC": self.entry_time,
            "EXIT_TIME_UTC": self.exit_time,
            "ENTRY_PRICE": self.entry_price,
            "EXIT_PRICE": self.exit_price,
            "RESEARCH_UNITS": self.entry_size,
            "HOURS_HELD": hours_held,
            "DELAYED_EXIT": hours_held > 4.000001,
            "GROSS_PNL": trade.pnl,
            "NET_PNL": trade.pnlcomm,
            "NET_TRADE_RETURN_PCT": (
                100 * trade.pnlcomm / entry_notional
            ),
        })


def load_frame(asset, signal_column):
    path = INPUT_DIR / f"{asset}_2025_backtest.csv"
    frame = pd.read_csv(path)

    frame["BAR_START_TIME"] = pd.to_datetime(
        frame["BAR_START_TIME"], utc=True
    )

    frame = frame.sort_values("BAR_START_TIME")

    if frame["BAR_START_TIME"].duplicated().any():
        raise ValueError(f"{asset}: duplicate timestamps")

    frame["ENTRY_SIGNAL"] = frame[signal_column].astype(int)

    frame = frame.set_index("BAR_START_TIME")

    # Backtrader timestamps remain UTC, represented as naive datetimes.
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    return frame

def calculate_daily_metrics(equity):
    frame = equity.copy()
    frame["TIME_UTC"] = pd.to_datetime(
        frame["TIME_UTC"], utc=True
    )

    values = frame.set_index("TIME_UTC")["EQUITY"].sort_index()

    # Starting portfolio value before the 2025 evaluation period.
    baseline = pd.Series(
        [STARTING_CASH],
        index=pd.DatetimeIndex(["2025-01-01"], tz="UTC"),
    )

    values = pd.concat([baseline, values])

    # Midnight belongs to the day that just ended.
    # Carry the last recorded valuation through days without bars.
    daily_equity = values.resample(
        "1D", closed="right", label="right"
    ).last().ffill()

    daily_returns = daily_equity.pct_change(
        fill_method=None
    ).dropna()

    daily_std = daily_returns.std(ddof=1)

    metrics = {
        "ANNUALIZED_RETURN_PCT": 100 * (
            (daily_equity.iloc[-1] / STARTING_CASH)
            ** (365 / len(daily_returns)) - 1
        ),
        "ANNUALIZED_VOLATILITY_PCT": (
            100 * daily_std * 365 ** 0.5
        ),
        # Assumes zero risk-free rate and no interest on cash.
        "SHARPE_RATIO_RF_ZERO": (
            daily_returns.mean() / daily_std * 365 ** 0.5
            if daily_std > 0 else None
        ),
    }

    daily = daily_equity.rename("EQUITY").to_frame()
    daily["DAILY_RETURN"] = daily_equity.pct_change(
        fill_method=None
    )
    daily.index.name = "DAY_END_UTC"

    return daily, metrics

def run_case(asset, variant, signal_column, cost_case, fee, slip):
    frame = load_frame(asset, signal_column)

    cerebro = bt.Cerebro(
        cheat_on_open=True,
        stdstats=False,
    )

    cerebro.adddata(
        SignalFeed(
            dataname=frame,
            timeframe=bt.TimeFrame.Minutes,
            compression=240,
        ),
        name=asset,
    )

    cerebro.addstrategy(MeanReversionStrategy)

    cerebro.broker.setcash(STARTING_CASH)

    # Explicitly use unleveraged price-unit accounting.
    cerebro.broker.setcommission(
        commission=fee,
        commtype=bt.CommInfoBase.COMM_PERC,
        stocklike=True,
        percabs=True,
    )

    # Apply the full assumed slippage even outside candle extremes,
    # rather than silently cap the configured cost.
    cerebro.broker.set_slippage_perc(
        slip,
        slip_open=True,
        slip_match=True,
        slip_out=True,
    )

    cerebro.addanalyzer(
        bt.analyzers.DrawDown,
        _name="drawdown",
    )

    cerebro.addanalyzer(
        EquityRecorder,
        _name="equity",
    )

    strategy = cerebro.run(runonce=False)[0]
    trades = pd.DataFrame(strategy.closed_trades)

    prefix = f"{asset}_{variant}_{cost_case}"

    equity = pd.DataFrame(
        strategy.analyzers.equity.get_analysis()
    )

    equity.to_csv(
        OUTPUT_DIR / f"{prefix}_equity.csv",
        index=False,
    )

    if not trades.empty:
        trades.to_csv(
            OUTPUT_DIR / f"{prefix}_trades.csv",
            index=False,
        )

    final_value = cerebro.broker.getvalue()
    drawdown = strategy.analyzers.drawdown.get_analysis()

    result = {
        "ASSET": asset,
        "VARIANT": variant,
        "COST_CASE": cost_case,
        "INPUT_SIGNALS": int(frame["ENTRY_SIGNAL"].sum()),
        "SUBMITTED_ENTRIES": strategy.submitted_entries,
        "CLOSED_TRADES": len(trades),
        "OPEN_POSITION_AT_END": bool(strategy.position),
        "REJECTED_ORDERS": strategy.rejected_orders,
        "SKIPPED_GAP_SIGNALS": strategy.skipped_gap_signals,
        "DELAYED_EXITS": (
            int(trades["DELAYED_EXIT"].sum())
            if not trades.empty else 0
        ),
        "FINAL_EQUITY": final_value,
        "PORTFOLIO_RETURN_PCT": (
            100 * (final_value / STARTING_CASH - 1)
        ),
        "MAX_DRAWDOWN_PCT": drawdown["max"]["drawdown"],
        "CLOSED_NET_PNL": (
            trades["NET_PNL"].sum()
            if not trades.empty else 0.0
        ),
        "AVG_NET_TRADE_RETURN_PCT": (
            trades["NET_TRADE_RETURN_PCT"].mean()
            if not trades.empty else None
        ),
        "NET_WIN_RATE_PCT": (
            100 * trades["NET_PNL"].gt(0).mean()
            if not trades.empty else None
        ),
    }

    print(
        f"{asset} | {variant} | {cost_case}: "
        f"{len(trades)} closed trades, "
        f"portfolio return={result['PORTFOLIO_RETURN_PCT']:.4f}%, "
        f"delayed exits={result['DELAYED_EXITS']}, "
        f"open at end={result['OPEN_POSITION_AT_END']}",
        flush=True,
    )

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Running 2025 research backtests; "
        "fractional units, not futures-contract accounting.",
        flush=True,
    )

    results = []

    for asset in ["BTC", "ETH"]:
        for variant, signal_column in VARIANTS.items():
            for cost_case, (fee, slip) in COST_CASES.items():
                results.append(
                    run_case(
                        asset,
                        variant,
                        signal_column,
                        cost_case,
                        fee,
                        slip,
                    )
                )

    summary = pd.DataFrame(results)
    path = OUTPUT_DIR / "mean_reversion_2025_summary.csv"
    summary.to_csv(path, index=False)

    print("\nSummary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()