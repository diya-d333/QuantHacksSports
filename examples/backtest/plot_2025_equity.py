from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "backtest_results"

equity = pd.read_csv(
    RESULTS / "mean_reversion_2025_equity.csv"
)
equity["time_utc"] = pd.to_datetime(equity["time_utc"], utc=True)
equity = equity.sort_values("time_utc")

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    equity["time_utc"],
    equity["portfolio_value"],
    color="navy",
    linewidth=2,
    label="Strategy equity",
)
ax.axhline(
    100000,
    color="gray",
    linestyle="--",
    linewidth=1,
    label="Starting capital: $100,000",
)

ax.set_title(
    "2025 Development Equity Curve — January–December\n"
    "Fear & Greed required; XRP/SOL informational"
)
ax.set_xlabel("Date (UTC)")
ax.set_ylabel("Portfolio value (USD)")
ax.yaxis.set_major_formatter(
    ticker.StrMethodFormatter("${x:,.0f}")
)
ax.ticklabel_format(axis="x", style="plain") if False else None
ax.grid(alpha=0.25)
ax.legend(loc="upper left")

fig.text(
    0.5, 0.02,
    "Net P&L: +$223.91 | Return: +0.224% | "
    "3 closed trades | Maximum drawdown: 0.04%\n"
    "Assumed costs per side: 0.10% commission + 0.05% slippage. "
    "Three trades span only two market events.",
    ha="center",
    fontsize=9,
)

fig.autofmt_xdate()
fig.tight_layout(rect=(0, 0.10, 1, 1))

output = RESULTS / "mean_reversion_2025_final_equity.png"
fig.savefig(output, dpi=250)
plt.close(fig)

print(f"Saved: {output}")

# Daily closing equity, including days without trades.
daily_equity = (
    equity.set_index("time_utc")["portfolio_value"]
    .resample("1D")
    .last()
    .ffill()
)

daily_returns = daily_equity.pct_change(fill_method=None).dropna()

# Assumption: zero risk-free rate; 365 calendar days per year.
daily_std = daily_returns.std(ddof=1)

if len(daily_returns) < 2 or daily_std == 0 or pd.isna(daily_std):
    print("Annualized daily Sharpe: N/A")
else:
    sharpe = (
        daily_returns.mean() / daily_std
    ) * (365 ** 0.5)

    print(f"Annualized daily Sharpe: {sharpe:.3f}")

print(
    "Sharpe uses daily equity returns and a zero risk-free rate. "
    "Missing calendar days carry forward the last recorded equity."
)