import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
from pandas.tseries.offsets import BDay

from .config import DB_PATH


def build_equity_curve_from_active(active: pd.DataFrame, daily_prices: dict) -> pd.Series:
    """
    Create a clean daily equity curve from the portfolio 'active' table
    by marking-to-market open positions through the last business day.

    active: must contain columns ['dt_tr', 'equity_mtm'] where equity_mtm is total
            portfolio equity after each execution (cash + MTM holdings).
    daily_prices: dict[ticker] -> DataFrame with a DatetimeIndex (business days).
                  Use it only to learn the 'true' last date in the database.
    """
    if active.empty:
        return pd.Series(dtype="float64")

    # 1) collapse executions to one equity point per day (last known equity that day)
    df = active.loc[:, ["dt_tr", "equity_mtm"]].copy()
    df["date"] = pd.to_datetime(df["dt_tr"]).dt.normalize()
    eq = df.groupby("date", as_index=True)["equity_mtm"].last().sort_index()

    # 2) build a complete business-day calendar to the latest known date in prices
    start = eq.index[0]
    # last date either from equity itself or from any of the price panels (whichever is later)
    price_last = None
    if daily_prices:
        try:
            price_last = max(v.index.max() for v in daily_prices.values() if len(v.index))
        except Exception:
            price_last = None
    end = max(eq.index.max(), price_last) if price_last is not None else eq.index.max()

    cal = pd.date_range(start, end, freq=BDay())  # business days
    # 3) forward-fill so open positions stay MTM'd
    eq = eq.reindex(cal).ffill()
    eq.name = "equity"
    eq.index.name = "date"
    return eq


def write_rules_daily(signals: pd.DataFrame, db_path: str = DB_PATH):
    rules = signals[
        [
            "ticker",
            "dt_tr",
            "date_tr",
            "close",
            "volume",
            "near_52w_high",
            "ma200_uptrend",
            "vol_surge",
            "orb_confirm",
            "vwap_support",
            "prox_to_52w",
            "vol_ratio",
            "rank_prox_vol",
            "signal",
            "selected",
        ]
    ].copy()
    rules["dt_tr"] = pd.to_datetime(rules["dt_tr"])
    rules["date_tr"] = pd.to_datetime(rules["date_tr"])
    con = sqlite3.connect(db_path)
    rules.to_sql("rules_daily", con, if_exists="replace", index=False)
    con.commit()
    con.close()
    return True


def log_run_to_db(
    db_path, strategy_name, cfg_dict, metrics_dict, n_trades, est_turnover=None, note=None
):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS runs_summary (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        strategy_name TEXT NOT NULL,
        params_json TEXT NOT NULL,
        hold_days INTEGER,
        cagr REAL,
        sharpe REAL,
        maxdd REAL,
        trades INTEGER,
        turnover REAL,
        note TEXT
    );
    """
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    params_json = json.dumps(cfg_dict, ensure_ascii=False)
    cur.execute(
        """
        INSERT INTO runs_summary
        (ts_utc, strategy_name, params_json, hold_days, cagr, sharpe, maxdd, trades, turnover, note)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """,
        (
            ts,
            strategy_name,
            params_json,
            cfg_dict.get("hold_days"),
            float(metrics_dict.get("CAGR", 0) or 0),
            float(metrics_dict.get("Sharpe", 0) or 0),
            float(metrics_dict.get("MaxDD", 0) or 0),
            int(n_trades or 0),
            None if est_turnover is None else float(est_turnover),
            note,
        ),
    )
    con.commit()
    con.close()


def compare_runs(db_path, limit=50):
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT run_id, ts_utc, strategy_name, params_json, hold_days,
               cagr, sharpe, maxdd, trades, turnover
        FROM runs_summary
        ORDER BY run_id DESC
        LIMIT ?
    """,
        con,
        params=(limit,),
    )
    con.close()

    def getp(s, k):
        try:
            return json.loads(s).get(k)
        except Exception:
            return None

    for k in ["prox_pct", "vol_mult", "orb_window_bars", "vwap_strict", "exposure_cap", "atr_k"]:
        df[k] = df["params_json"].apply(lambda s: getp(s, k))
    df = df.drop(columns=["params_json"])
    return df


def plot_signals_scatter(trades: pd.DataFrame, outdir: str, ts_tag: str):
    """
    One dot per signal. X = month (YYYY-MM), Y = within-month index (just to spread dots),
    color = interval (e.g., '1d', '240m', '60m', etc.).
    """
    import matplotlib.pyplot as plt

    df = trades.copy()
    # Expect 'dt' (or 'entry_date') and 'interval' in trades_*.csv
    dtcol = "dt" if "dt" in df.columns else ("entry_date" if "entry_date" in df.columns else None)
    if dtcol is None:
        return

    df[dtcol] = pd.to_datetime(df[dtcol])
    df["month"] = df[dtcol].dt.to_period("M").dt.to_timestamp()
    if "interval" not in df.columns:
        df["interval"] = "1d"  # fallback

    # rank-within-month for vertical stacking
    df["y"] = df.groupby("month").cumcount() + 1

    # color by interval
    intervals = sorted(df["interval"].unique().tolist())

    plt.figure(figsize=(18, 7))
    for iv in intervals:
        sub = df[df["interval"] == iv]
        plt.scatter(sub["month"], sub["y"], s=15, label=iv)

    plt.title("Signals by Month (colored by interval)")
    plt.xlabel("Month")
    plt.ylabel("Count within month (dots)")
    plt.legend(title="Interval")
    plt.tight_layout()
    fn = os.path.join(outdir, f"signals_scatter_{ts_tag}.png")
    plt.savefig(fn, dpi=140)
    plt.close()
