import backtrader as bt


class BehavioralMeanReversionStrategy(bt.Strategy):

    params = (
        ("lookback", 20),
        ("drop_threshold", 2.0),

        # ADX regime settings
        ("adx_period", 14),
        ("adx_threshold", 25),

        # Broad-market confirmation
        ("market_crash_threshold", -0.02),

        # Sentiment is tracked, but is not required for entry
        ("sentiment_threshold", 40),
    )

    def __init__(self):

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

        self.btc_return = bt.indicators.PercentChange(
            self.btc.close,
            period=1,
        )

        self.eth_return = bt.indicators.PercentChange(
            self.eth.close,
            period=1,
        )

        self.xrp_return = bt.indicators.PercentChange(
            self.xrp.close,
            period=1,
        )

        self.sol_return = bt.indicators.PercentChange(
            self.sol.close,
            period=1,
        )

        # ------------------------------------------
        # Recent volatility
        # ------------------------------------------

        self.btc_volatility = bt.indicators.StandardDeviation(
            self.btc_return,
            period=self.p.lookback,
        )

        self.eth_volatility = bt.indicators.StandardDeviation(
            self.eth_return,
            period=self.p.lookback,
        )

        # ------------------------------------------
        # Large-drop conditions
        # ------------------------------------------

        self.btc_large_drop = (
            self.btc_return
            < -self.p.drop_threshold * self.btc_volatility
        )

        self.eth_large_drop = (
            self.eth_return
            < -self.p.drop_threshold * self.eth_volatility
        )

        # ------------------------------------------
        # ADX regime detection
        # ------------------------------------------

        self.btc_adx = bt.indicators.AverageDirectionalMovementIndex(
            self.btc,
            period=self.p.adx_period,
        )

        self.eth_adx = bt.indicators.AverageDirectionalMovementIndex(
            self.eth,
            period=self.p.adx_period,
        )

        self.btc_range_bound = (
            self.btc_adx < self.p.adx_threshold
        )

        self.eth_range_bound = (
            self.eth_adx < self.p.adx_threshold
        )

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
                    self.btc_entry_bar = len(self)

                elif order.issell():
                    self.btc_entry_bar = None

                self.btc_order = None

            # ETH order
            elif order.data is self.eth:

                if order.isbuy():
                    self.eth_entry_bar = len(self)

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

        no_broad_crash = (
            not self.broad_market_crash[0]
        )

        fearful_sentiment = (
            self.sentiment[0]
            < self.p.sentiment_threshold
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

        # ------------------------------------------
        # Signals
        # ------------------------------------------

        btc_signal = (
            self.btc_large_drop[0]
            and self.btc_range_bound[0]
            and no_broad_crash
        )

        eth_signal = (
            self.eth_large_drop[0]
            and self.eth_range_bound[0]
            and no_broad_crash
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
            and len(self) > self.btc_entry_bar
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
            and len(self) > self.eth_entry_bar
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
        print(
            "NOTE: Sentiment is measured but is not "
            "currently required for entry."
        )

        print("========================================")
        print()


STRATEGY_CLASS = BehavioralMeanReversionStrategy
