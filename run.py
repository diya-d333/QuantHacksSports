"""Offline analysis of saved mean-reversion backtests. Run: python run.py.
Uses Python's standard library only; does not rerun signals or model training.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import csv
import html
import json
import math
import statistics
import sys
import webbrowser

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'backtest_results'
STARTING_CAPITAL = 100000.0
CASES = [
    ('2025 development baseline', 'baseline_25_consecutive20/equity.csv',
     'baseline_25_consecutive20/trades.csv'),
    ('2026 out-of-sample', 'mean_reversion_2026_oos_equity.csv',
     'mean_reversion_2026_oos_trades.csv'),
]


def read_csv(name):
    path = RESULTS / name
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Equity timestamps must include a UTC offset')
    return result.astimezone(timezone.utc)


def analyze(label, equity_name, trades_name):
    raw = read_csv(equity_name)
    points = sorted((timestamp(r['time_utc']), float(r['portfolio_value'])) for r in raw)
    if len(points) < 2 or len({t for t, _ in points}) != len(points):
        raise ValueError(f'{label}: insufficient or duplicate equity observations')
    if any(not math.isfinite(v) or v <= 0 for _, v in points):
        raise ValueError(f'{label}: invalid equity')
    trades = read_csv(trades_name)
    # UTC calendar-day closing equity; carry forward days without observations.
    closes = {t.date(): v for t, v in points}
    day, last_day = points[0][0].date(), points[-1][0].date()
    daily = []
    previous = points[0][1]
    while day <= last_day:
        previous = closes.get(day, previous)
        daily.append(previous)
        day += timedelta(days=1)
    returns = [b / a - 1 for a, b in zip(daily, daily[1:])]
    daily_std = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sharpe = statistics.mean(returns) / daily_std * math.sqrt(365) if daily_std else None
    peak, max_dd = STARTING_CAPITAL, 0.0
    for _, value in points:
        peak = max(peak, value)
        max_dd = max(max_dd, 1 - value / peak)
    elapsed_days = (points[-1][0] - points[0][0]).total_seconds() / 86400
    final = points[-1][1]
    net = sum(float(t['net_pnl']) for t in trades)
    # Both entry and exit dollar notionals, divided by average daily equity.
    traded_notional = sum(abs(float(t['units'])) *
        (float(t['entry_price']) + float(t['exit_price'])) for t in trades)
    metrics = {
        'period': label, 'first_observation_utc': points[0][0].isoformat(),
        'last_observation_utc': points[-1][0].isoformat(),
        'starting_capital': STARTING_CAPITAL, 'final_equity': final,
        'net_pnl': final - STARTING_CAPITAL,
        'return_pct': 100 * (final / STARTING_CAPITAL - 1),
        'annualized_return_pct': 100 * ((final / STARTING_CAPITAL) ** (365 / elapsed_days) - 1),
        'annualized_volatility_pct': 100 * daily_std * math.sqrt(365),
        'annualized_daily_sharpe': sharpe,
        'max_drawdown_pct': 100 * max_dd, 'closed_trades': len(trades),
        'entry_events': len({t['entry_time_utc'] for t in trades}),
        'two_way_period_turnover_pct': 100 * traded_notional / statistics.mean(daily),
        'closed_trade_net_pnl': net,
        'equity_pnl_minus_closed_trade_pnl': final - STARTING_CAPITAL - net,
        'delayed_exit_trades': sum(float(t['hours_held']) > 4.001 for t in trades),
        'equity_source': equity_name, 'trade_source': trades_name,
    }
    return metrics, points


def svg_chart(points):
    width, height = 960, 300
    left, top, plot_w, plot_h = 90, 20, 830, 230
    lo, hi = min(v for _, v in points), max(v for _, v in points)
    padding = max((hi - lo) * 0.12, 10)
    lo, hi = lo - padding, hi + padding
    start, end = points[0][0].timestamp(), points[-1][0].timestamp()
    coords = ' '.join(f'{left+(t.timestamp()-start)/(end-start)*plot_w:.2f},{top+(hi-v)/(hi-lo)*plot_h:.2f}' for t,v in points)
    labels = []
    for i in range(5):
        y = top + i * plot_h / 4
        value = hi - i * (hi-lo) / 4
        labels.append(f'<line x1="{left}" x2="920" y1="{y}" y2="{y}" stroke="#ddd"/><text x="80" y="{y+5}" text-anchor="end">${value:,.0f}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Portfolio equity over time">'+''.join(labels)+f'<polyline points="{coords}" fill="none" stroke="#163d82" stroke-width="2"/><text x="90" y="285">{points[0][0].date()}</text><text x="920" y="285" text-anchor="end">{points[-1][0].date()} (UTC)</text></svg>'


def main():
    analyses = [analyze(*case) for case in CASES]
    output = ROOT / 'judge_results'
    output.mkdir(exist_ok=True)
    metrics = [m for m, _ in analyses]
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    sections = []
    fields = [('Net P&L', 'net_pnl', '$'), ('Return', 'return_pct', '%'),
        ('Annualized return', 'annualized_return_pct', '%'),
        ('Annualized daily volatility', 'annualized_volatility_pct', '%'),
        ('Annualized daily Sharpe', 'annualized_daily_sharpe', ''),
        ('Maximum drawdown', 'max_drawdown_pct', '%'),
        ('Closed trades', 'closed_trades', 'count'),
        ('Distinct entry events', 'entry_events', 'count'),
        ('Two-way period turnover', 'two_way_period_turnover_pct', '%'),
        ('Trades held beyond four hours', 'delayed_exit_trades', 'count')]
    for m, points in analyses:
        print('\n' + m['period'])
        rows = []
        for label, key, unit in fields:
            val = m[key]
            text = 'N/A' if val is None else (str(val) if unit == 'count' else
                (f'${val:,.2f}' if unit == '$' else f'{val:.3f}{unit}'))
            print(f'  {label}: {text}')
            rows.append(f'<tr><th>{label}</th><td>{text}</td></tr>')
        difference = m['equity_pnl_minus_closed_trade_pnl']
        if abs(difference) > 0.01:
            print(f'  NOTE: equity P&L differs from closed-trade P&L by ${difference:.2f}. Check open positions/source pairing.')
        sections.append(f'<section><h2>{html.escape(m["period"])}</h2><p>Observed coverage: {points[0][0].date()} to {points[-1][0].date()} UTC.</p><table>'+''.join(rows)+'</table>'+svg_chart(points)+'</section>')
    notes = ('Metrics recomputed from saved backtest equity and completed trade logs; '
        'this command does not rerun the signal-generation or execution pipeline. '
        '2025 is development data; 2026 is the frozen-model OOS test. '
        'Fear & Greed <= 25 required; XRP/SOL confirmation disabled. '
        'Commission 0.10% and assumed slippage 0.05% per side are already reflected in equity. '
        'Sharpe: zero risk-free rate, sample daily standard deviation, sqrt(365) annualization; '
        'missing calendar days carry forward equity. Annualized return uses actual observed elapsed time. '
        'Turnover: entry plus exit notional for completed trades / average daily equity, not annualized. '
        'Drawdown uses recorded four-hour equity, not intrabar lows. '
        'Few correlated trading events and limited execution checks constrain conclusions. '
        'A timestamp on January 1, 2026 can represent the close of the final 2025 candle.')
    report = '<!doctype html><meta charset="utf-8"><title>Mean Reversion Results</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#182234}section{margin:35px 0}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}svg{width:100%;margin-top:20px}p{line-height:1.6}</style><h1>Mean Reversion: Development and OOS Results</h1><p>'+html.escape(notes)+'</p>'+''.join(sections)
    target = output / 'report.html'
    target.write_text(report, encoding='utf-8')
    print(f'\nReport: {target}\nMetrics: {output / "metrics.csv"}')
    if '--no-open' not in sys.argv:
        webbrowser.open(target.as_uri())


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f'Cannot produce report: {exc}')
