# BIST100 Signal Prototype
A minimal research prototype for generating, backtesting and reporting daily trading signals on BIST100 tickers. The baseline strategy holds each selected position for 3 trading days, uses intraday confirmations, and includes portfolio sizing, fees/slippage and CPI (TUFE) deflation for “real” performance.

## Highlights

- Deterministic runs (stable sorting + fixed seeds)

- Position sizing by equity fraction (e.g., 5–10%, tunable)

- Intraday confirmations (30m & 4h) with safe fallbacks

- CPI-deflated “real” equity curves

- Rich reporting (equity curves, contributors, buckets, waterfall)

- Signals persisted to SQLite with run_id + run_datetime

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pre-commit install
```

## Run
```bash
bist100 run

bist100 report --limit 20
```

## Outputs
```
equity_curve_*.png – nominal equity curves (strategy + benchmarks)

equity_curve_real_*.png – CPI-deflated / “real” equity curves

portfolio_monthly_*.png – ledger equity vs. research equity index

portfolio_waterfall_*.png – monthly step PnL with running equity

contrib_*.csv/png – contribution table + charts (top/bottom)

trade_buckets_*.csv/png – trade quality bins and counts

signals_scatter_*.png – signals by month

executions_*.csv – ledger of position changes (buys/sells)

trades_3d_*.csv – trade list (per-trade returns, durations, etc.)

portfolio_*.csv – daily equity curve used for plots and metrics

Signals are also persisted to SQLite.
```
