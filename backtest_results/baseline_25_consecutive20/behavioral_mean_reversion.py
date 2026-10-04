import backtrader as bt
import math


class BehavioralMeanReversionStrategy(bt.Strategy):

    params = (
        ("lookback", 20),
        ("drop_threshold", 2.0),

        # ADX regime settings
        ("adx_period", 14),
        ("adx_threshold", 25),

        # Broad-market confirmation
        ("market_crash_threshold", -0.02),

        # Confirmation filters; both required by default.
        ("sentiment_threshold", 25),
        ("require_sentiment", True),
        ("require_market_confirmation", True),
    )

    def __init__(self):

        self.drop_range_counts = {"BTC": 0, "ETH": 0}
        self.drop_range_fear_counts = {"BTC": 0, "ETH": 0}

        # ------------------------------------------
        # Data feeds
        # ------------------------------------------

        self.btc = self.datas[0]
        self.eth = self.datas[1]
        self.xrp = self.datas[2]
        self.sol = self.datas[3]

        # ------------------------------------------
        # Order tracking
        # ------------------------------------------

        self.btc_order = None
        self.eth_order = None

        # Entry bar is recorded only AFTER
        # Backtrader confirms that the buy filled.
        self.btc_entry_bar = None
        self.eth_entry_bar = None

        # ------------------------------------------
        # Diagnostic counters
        # ------------------------------------------

        self.total_bars_checked = 0

        self.btc_large_drop_count = 0
        self.eth_large_drop_count = 0

        self.btc_range_bound_count = 0
        self.eth_range_bound_count = 0

        self.no_broad_crash_count = 0
        self.fearful_sentiment_count = 0

        self.btc_signal_count = 0
        self.eth_signal_count = 0

        # ------------------------------------------
        # 4-hour returns
        # ------------------------------------------
        # Features calculated on each asset's full timeline.

        self.btc_return = self.btc.return_4h
        self.eth_return = self.eth.return_4h
        self.xrp_return = self.xrp.return_4h
        self.sol_return = self.sol.return_4h

        self.btc_large_drop = self.btc.large_drop
        self.eth_large_drop = self.eth.large_drop

        # ------------------------------------------
        # Large-drop conditions
        # ------------------------------------------


        # ------------------------------------------
        # ADX regime detection
        # ------------------------------------------



        self.btc_range_bound = self.btc.snowflake_range
        self.eth_range_bound = self.eth.snowflake_range

        # ------------------------------------------
        # XRP + SOL broad-market confirmation
        # ------------------------------------------

        self.market_return = (
            self.xrp_return + self.sol_return
        ) / 2

        self.broad_market_crash = (
            self.market_return
            < self.p.market_crash_threshold
        )

        # ------------------------------------------
        # Fear & Greed sentiment
        # ------------------------------------------

        self.sentiment = self.btc.fear_greed_score

    def notify_order(self, order):

        # ------------------------------------------
        # Ignore orders that are not finished yet
        # ------------------------------------------

        if order.status in [
            order.Submitted,
            order.Accepted,
        ]:
            return

        # ------------------------------------------
        # Completed order
        # ------------------------------------------

        if order.status == order.Completed:

            # BTC order
            if order.data is self.btc:

                if order.isbuy():
                    self.btc_entry_bar = order.executed.dt

                elif order.issell():
                    self.btc_entry_bar = None

                self.btc_order = None

            # ETH order
            elif order.data is self.eth:

                if order.isbuy():
                    self.eth_entry_bar = order.executed.dt

                elif order.issell():
                    self.eth_entry_bar = None

                self.eth_order = None

        # ------------------------------------------
        # Failed / cancelled order
        # ------------------------------------------

        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:

            if order.data is self.btc:
                self.btc_order = None

            elif order.data is self.eth:
                self.eth_order = None

    def next(self):

        self.total_bars_checked += 1

        # ------------------------------------------
        # Current conditions
        # ------------------------------------------

        confirmation_ready = (
            math.isfinite(self.xrp_return[0])
            and math.isfinite(self.sol_return[0])
        )

        no_broad_crash = (
            confirmation_ready
            and self.market_return[0] >= self.p.market_crash_threshold
        )

        fearful_sentiment = (
            self.sentiment[0]
            <= self.p.sentiment_threshold
        )

        # ------------------------------------------
        # Diagnostic counts
        # ------------------------------------------

        if self.btc_large_drop[0]:
            self.btc_large_drop_count += 1

        if self.eth_large_drop[0]:
            self.eth_large_drop_count += 1

        if self.btc_range_bound[0]:
            self.btc_range_bound_count += 1

        if self.eth_range_bound[0]:
            self.eth_range_bound_count += 1

        if no_broad_crash:
            self.no_broad_crash_count += 1

        if fearful_sentiment:
            self.fearful_sentiment_count += 1

        for asset, drop, range_flag in [
            ("BTC", self.btc_large_drop[0], self.btc_range_bound[0]),
            ("ETH", self.eth_large_drop[0], self.eth_range_bound[0]),
        ]:
            if drop and range_flag:
                self.drop_range_counts[asset] += 1
                if fearful_sentiment:
                    self.drop_range_fear_counts[asset] += 1

                    print(
                        f"{asset} candidate | "
                        f"bar_start={self.btc.datetime.datetime(0)} | "
                        f"fear={self.sentiment[0]:.1f} | "
                        f"XRP_return={self.xrp_return[0]:.4%} | "
                        f"SOL_return={self.sol_return[0]:.4%} | "
                        f"confirmation_ready={confirmation_ready} | "
                        f"no_broad_crash={no_broad_crash}"
                    )

        # ------------------------------------------
        # Signals
        # ------------------------------------------

        sentiment_pass = (
            not self.p.require_sentiment
            or fearful_sentiment
        )

        market_pass = (
            not self.p.require_market_confirmation
            or no_broad_crash
        )

        btc_signal = (
            self.btc_large_drop[0]
            and self.btc_range_bound[0]
            and sentiment_pass
            and market_pass
        )

        eth_signal = (
            self.eth_large_drop[0]
            and self.eth_range_bound[0]
            and sentiment_pass
            and market_pass
        )

        if btc_signal:
            self.btc_signal_count += 1

        if eth_signal:
            self.eth_signal_count += 1

        # ------------------------------------------
        # EXIT BTC
        #
        # Exit after one complete 4-hour bar
        # following the actual filled entry.
        # ------------------------------------------

        if (
            self.getposition(self.btc).size > 0
            and self.btc_entry_bar is not None
            and (
                self.btc.datetime[0] + 4.0 / 24.0
                >= self.btc_entry_bar + 4.0 / 24.0
            )
            and self.btc_order is None
        ):
            self.btc_order = self.close(
                data=self.btc
            )

        # ------------------------------------------
        # EXIT ETH
        # ------------------------------------------

        if (
            self.getposition(self.eth).size > 0
            and self.eth_entry_bar is not None
            and (
                self.eth.datetime[0] + 4.0 / 24.0
                >= self.eth_entry_bar + 4.0 / 24.0
            )
            and self.eth_order is None
        ):
            self.eth_order = self.close(
                data=self.eth
            )

        # ------------------------------------------
        # ENTER BTC
        # ------------------------------------------

        if (
            btc_signal
            and self.getposition(self.btc).size == 0
            and self.btc_order is None
        ):
            self.btc_order = self.buy(
                data=self.btc
            )

        # ------------------------------------------
        # ENTER ETH
        # ------------------------------------------

        if (
            eth_signal
            and self.getposition(self.eth).size == 0
            and self.eth_order is None
        ):
            self.eth_order = self.buy(
                data=self.eth
            )

    def stop(self):

        print("Drop + Snowflake range:", self.drop_range_counts)
        print("Drop + range + fear:", self.drop_range_fear_counts)

        print()
        print("========== SIGNAL DIAGNOSTICS ==========")
        print(f"Bars checked:             {self.total_bars_checked}")

        print()
        print("BTC:")
        print(f"Large drops:              {self.btc_large_drop_count}")
        print(f"ADX range-bound bars:     {self.btc_range_bound_count}")
        print(f"Final BTC signals:        {self.btc_signal_count}")

        print()
        print("ETH:")
        print(f"Large drops:              {self.eth_large_drop_count}")
        print(f"ADX range-bound bars:     {self.eth_range_bound_count}")
        print(f"Final ETH signals:        {self.eth_signal_count}")

        print()
        print("MARKET / SENTIMENT:")
        print(f"No broad-market crash:    {self.no_broad_crash_count}")
        print(f"Fearful sentiment:        {self.fearful_sentiment_count}")

        print()
        print("Entry requires a large drop and a Snowflake range label.")
        print(
            f"Fear & Greed filter: {self.p.require_sentiment} "
            f"(score <= {self.p.sentiment_threshold})"
        )
        print(
            f"XRP/SOL filter: {self.p.require_market_confirmation} "
            f"(average return >= {self.p.market_crash_threshold:.1%})"
        )

        print("========================================")
        print()


STRATEGY_CLASS = BehavioralMeanReversionStrategy
