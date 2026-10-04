from pathlib import Path
import sys

import backtrader as bt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavioral_mean_reversion import BehavioralMeanReversionStrategy


class TestFeed(bt.feeds.PandasData):
    lines = (
        "fear_greed_score",
        "snowflake_range",
        "return_4h",
        "large_drop",
    )

    params = (
        ("fear_greed_score", "fear_greed_score"),
        ("snowflake_range", "snowflake_range"),
        ("return_4h", "return_4h"),
        ("large_drop", "large_drop"),
    )


class CheckedStrategy(BehavioralMeanReversionStrategy):
    def __init__(self):
        super().__init__()
        self.fills = []

    def notify_order(self, order):
        super().notify_order(order)

        if order.status == order.Completed:
            self.fills.append((
                order.data._name,
                "BUY" if order.isbuy() else "SELL",
                bt.num2date(order.executed.dt),
            ))


def main():
    # Artificial candles: no historical or 2026 data is loaded.
    times = pd.to_datetime([
        "2000-01-01 00:00:00",
        "2000-01-01 04:00:00",
        "2000-01-01 08:00:00",
        "2000-01-03 08:00:00",
        "2000-01-03 12:00:00",
        "2000-01-03 16:00:00",
    ])
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    for asset in ["BTC", "ETH", "XRP", "SOL"]:
        frame = pd.DataFrame({
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
            "fear_greed_score": 20.0,
            "snowflake_range": 1.0,
            "return_4h": 0.0,
            "large_drop": 0.0,
        }, index=times)

        # Exactly one BTC entry candidate.
        if asset == "BTC":
            frame.loc[times[1], "large_drop"] = 1.0

        cerebro.adddata(TestFeed(dataname=frame), name=asset)

    cerebro.addstrategy(CheckedStrategy)
    strategy = cerebro.run()[0]

    actual = [
        (asset, side, pd.Timestamp(timestamp))
        for asset, side, timestamp in strategy.fills
    ]

    expected = [
        ("BTC", "BUY", times[2]),
        ("BTC", "SELL", times[3]),
    ]

    print("Executed orders:", actual)
    assert actual == expected, (
        f"Unexpected execution timing: {actual}"
    )
    assert strategy.getposition(strategy.btc).size == 0
    print("PASS: exit fills at the first available open after the gap.")


if __name__ == "__main__":
    main()