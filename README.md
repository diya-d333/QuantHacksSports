# QuantHacks_PRISM

PRISM is a regime-aware BTC/ETH trading framework that combines Databento market data, Tiger Data storage, and frozen regime models trained inside Snowflake.

The reported PRISM baseline uses market-regime classification and sentiment filters to identify potential short-term rebounds following large price declines.

## Run the results report

Download or clone the complete repository. With Python 3.11 or newer installed, run this command from the project root:

```bash
python run.py
```

No additional Python packages, API keys, or database credentials are required for this report.

The command recalculates metrics from saved PRISM baseline equity and trade logs, prints them, and opens a report with equity curves.

Generated files:

- `judge_results/report.html`
- `judge_results/metrics.csv`
- `judge_results/metrics.json`

If the browser does not open automatically, open `judge_results/report.html` manually.

## Reported PRISM baseline results

| Metric | 2025 development | 2026 OOS |
| --- | --- | --- |
| Starting capital | $100,000 | $100,000 |
| Net P&L | +$223.91 | −$215.19 |
| Total return | +0.224% | −0.215% |
| Annualized return | +0.225% | −0.288% |
| Annualized daily volatility | 0.161% | 0.498% |
| Annualized daily Sharpe | 1.396 | −0.578 |
| Maximum drawdown | 0.040% | 0.502% |
| Closed trades | 3 | 8 |
| Distinct entry events | 2 | 4 |
| Two-way period turnover | 60.178% | 159.934% |

2025 is development data. The 2026 out-of-sample evaluation covers January–September and uses frozen models without retraining on 2026 data.

These metrics describe the specific PRISM baseline and configuration documented below.

## PRISM signal logic and cost assumptions

The reported baseline requires all three entry conditions:

- A large negative four-hour return.
- A range-candidate regime label from the frozen Snowflake model.
- A Fear & Greed Index value of 25 or lower.

This configuration implements a mean-reversion component within PRISM. XRP/SOL confirmation is disabled in the reported baseline.

The backtest uses 10% of available cash per entry, next-open execution, and an intended four-hour holding period. Trading gaps can delay exits; two 2026 trades held longer than four hours.

Commission is 0.10% per side and assumed slippage is 0.05% per side. Both are already reflected in the saved results.

## Metric definitions

Sharpe uses UTC calendar-day equity returns, a zero risk-free rate, sample standard deviation, and √365 annualization. Missing calendar days carry forward the last recorded equity.

Annualized return uses the elapsed observed period.

Turnover is completed-trade entry plus exit dollar notional divided by average daily equity; it is not annualized.

Maximum drawdown uses recorded four-hour equity observations, so it does not measure intrabar losses.

## Reproduction scope

`run.py` reproduces the reported PRISM baseline analysis from saved backtest outputs. It does not rerun model training, data downloads, signal generation, or the full execution backtest.

Its input files are:

- `backtest_results/baseline_25_consecutive20/equity.csv`
- `backtest_results/baseline_25_consecutive20/trades.csv`
- `backtest_results/mean_reversion_2026_oos_equity.csv`
- `backtest_results/mean_reversion_2026_oos_trades.csv`

The 2026 input filenames retain the original strategy-component naming and contain the saved outputs used for this PRISM baseline report.

The full research pipeline requires its separate dependencies, authorized data access, and database setup.

## Interpretation and limitations

The PRISM baseline made money during development but lost money in the 2026 out-of-sample test after costs. These results do not demonstrate a reliable profitable edge.

The sample contains few trading events, with BTC and ETH often trading together. Execution checks were limited. Fractional, stock-like position accounting on continuous futures prices is a research simplification, not a complete futures execution model.

The reported evaluation applies to this baseline configuration; it does not establish the performance of additional PRISM components or alternative configurations.