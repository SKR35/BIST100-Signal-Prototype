import hashlib
import json

import os
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import (
    apply_costs_to_daily_curve,
    apply_roundtrip_costs,
    equity_curve_from_trades,
    make_trades_table_full,
)
from .config import CFG, DB_PATH, ORB_MINUTES
from .data_io import load_daily, load_intraday_best
from .indicators import compute_daily_indicators
from .intraday import compute_intraday_confirmations
from .metrics import contribution_report, portfolio_metrics, return_buckets, trade_stats
from .portfolio import PortfolioConfig, build_virtual_portfolio
from .reporting import log_run_to_db, plot_signals_scatter, write_rules_daily
from .signals import build_signal_table, select_top_signals_per_day
from .stops import apply_atr_stop

# --- determinism guardrails ---
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)


# --- TUFE inflation helpers ---------------------------------------------------
def _tufe_daily_index(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Build a daily CPI index from hard-coded monthly TUFE % changes.
    Returns dataframe: dt_tr, cpi_index (normalized to 1.0 at the first month)
    """
    _tufe = [
        (2022, 9, 3.08),
        (2022, 10, 3.54),
        (2022, 11, 2.88),
        (2022, 12, 1.18),
        (2023, 1, 6.65),
        (2023, 2, 3.15),
        (2023, 3, 2.29),
        (2023, 4, 2.39),
        (2023, 5, 0.04),
        (2023, 6, 3.92),
        (2023, 7, 9.49),
        (2023, 8, 9.09),
        (2023, 9, 4.75),
        (2023, 10, 3.43),
        (2023, 11, 3.28),
        (2023, 12, 2.93),
        (2024, 1, 6.7),
        (2024, 2, 4.53),
        (2024, 3, 3.16),
        (2024, 4, 3.18),
        (2024, 5, 3.37),
        (2024, 6, 1.64),
        (2024, 7, 3.23),
        (2024, 8, 2.47),
        (2024, 9, 2.97),
        (2024, 10, 2.88),
        (2024, 11, 2.24),
        (2024, 12, 1.03),
        (2025, 1, 5.03),
        (2025, 2, 2.27),
        (2025, 3, 2.46),
        (2025, 4, 3),
        (2025, 5, 1.53),
        (2025, 6, 1.37),
        (2025, 7, 2.06),
        (2025, 8, 2.04),
    ]
    tufe = pd.DataFrame(_tufe, columns=["year", "month", "pct"])
    tufe["ym"] = pd.to_datetime(tufe["year"].astype(str) + "-" + tufe["month"].astype(str) + "-01")
    tufe["factor"] = 1.0 + tufe["pct"] / 100.0
    tufe["cpi_index"] = tufe["factor"].cumprod()
    # normalize to 1.0 at the first available month
    base = tufe["cpi_index"].iloc[0]
    tufe["cpi_index"] = tufe["cpi_index"] / base

    # Daily frame, mapped by month, forward-filled
    daily = pd.DataFrame(
        {"dt_tr": pd.date_range(start_dt.normalize(), end_dt.normalize(), freq="D")}
    )
    daily["ym"] = daily["dt_tr"].values.astype("datetime64[M]")
    daily = daily.merge(tufe[["ym", "cpi_index"]], on="ym", how="left").sort_values("dt_tr")
    daily["cpi_index"] = daily["cpi_index"].ffill()
    return daily[["dt_tr", "cpi_index"]]


# -----------------------------------------------------------------------------


def compute_benchmarks(daily_df: pd.DataFrame, cap_abs: float | None = 0.05):
    """
    Build synthetic equal-weight benchmarks from daily close-to-close returns.
    Returns a dict with DataFrames: {'mean': df, 'median': df, 'capped': df or None}
    df columns: ['dt_tr', 'ret', 'eq']
    """
    # 1) canonical dates & stable sort
    d = daily_df.copy()
    d["dt_tr"] = pd.to_datetime(d["dt_tr"]).dt.normalize()
    d = d.sort_values(["ticker", "dt_tr", "close"], kind="mergesort")

    # 2) ensure a single close per (ticker, date)
    #    (if multiple rows exist for the same day, take the last close)
    d = d.groupby(["ticker", "dt_tr"], as_index=False).agg(close=("close", "last"))

    # 3) close-to-close returns
    d["ret_cc"] = d.groupby("ticker", sort=False)["close"].pct_change()

    # 4) robust pivot (safe if a duplicate ever slips through)
    R = d.pivot_table(index="dt_tr", columns="ticker", values="ret_cc", aggfunc="mean")
    R = R.sort_index()

    # Equal-weight MEAN
    ret_mean = R.mean(axis=1, skipna=True).fillna(0.0)
    mean_df = pd.DataFrame({"dt_tr": ret_mean.index, "ret": ret_mean.values})
    mean_df["eq"] = (1.0 + mean_df["ret"]).cumprod()

    # MEDIAN (robust to outliers)
    ret_median = R.median(axis=1, skipna=True).fillna(0.0)
    median_df = pd.DataFrame({"dt_tr": ret_median.index, "ret": ret_median.values})
    median_df["eq"] = (1.0 + median_df["ret"]).cumprod()

    # CAPPED MEAN (clip extreme single-name moves, e.g., ±5%)
    capped_df = None
    if cap_abs is not None:
        Rc = R.clip(lower=-cap_abs, upper=cap_abs)
        ret_capped = Rc.mean(axis=1, skipna=True).fillna(0.0)
        capped_df = pd.DataFrame({"dt_tr": ret_capped.index, "ret": ret_capped.values})
        capped_df["eq"] = (1.0 + capped_df["ret"]).cumprod()

    return {"mean": mean_df, "median": median_df, "capped": capped_df}


def run_pipeline():
    daily = load_daily(DB_PATH)
    m_intra = load_intraday_best(DB_PATH)

    # canonical sort & normalized dates (prevents groupby/merge order drift)
    daily["dt_tr"] = pd.to_datetime(daily["dt_tr"]).dt.normalize()
    m_intra["dt_tr"] = pd.to_datetime(m_intra["dt_tr"]).dt.normalize()

    daily = daily.sort_values(["ticker", "dt_tr"], kind="mergesort").reset_index(drop=True)
    m_intra = m_intra.sort_values(["ticker", "dt_tr"], kind="mergesort").reset_index(drop=True)

    daily_ind = compute_daily_indicators(daily)
    # Liquidity/price guards (optional, default thresholds above)
    liq = daily_ind[
        (daily_ind["adv20"] >= CFG["min_adv_try"]) & (daily_ind["close"] >= CFG["min_price"])
    ].copy()
    daily_ind = liq

    # --- intraday confirmations (older vs recent) ---
    cutoff = daily["dt_tr"].max() - pd.Timedelta(days=CFG["recent_intraday_days"])

    older = m_intra.loc[m_intra["dt_tr"] < cutoff].copy()
    recent = m_intra.loc[m_intra["dt_tr"] >= cutoff].copy()

    intra_older = compute_intraday_confirmations(
        older,
        orb_minutes=ORB_MINUTES,
        window_bars=CFG.get("orb_window_bars"),
        vwap_required=CFG.get("vwap_strict", True),
        force_min_bar=None,  # allow small bars / even daily for far past
    )

    intra_recent = compute_intraday_confirmations(
        recent,
        orb_minutes=ORB_MINUTES,
        window_bars=CFG.get("orb_window_bars"),
        vwap_required=True,  # be strict for execution realism
        force_min_bar=CFG.get("min_intraday_bar", 240),  # >=4h in the recent window
    )

    intra_flags = pd.concat([intra_older, intra_recent], ignore_index=True)

    # --- normalize intraday join keys so both signal merge and trade merge work ---
    # needed: date_tr (date) for signals, and dt_tr_day (date) for trades->interval merge
    if "dt_tr" in intra_flags.columns and "date_tr" not in intra_flags.columns:
        intra_flags["date_tr"] = pd.to_datetime(intra_flags["dt_tr"]).dt.date

    if "dt_tr_day" not in intra_flags.columns:
        # provide a consistent name for the trade-merge key
        if "dt_tr" in intra_flags.columns:
            intra_flags["dt_tr_day"] = pd.to_datetime(intra_flags["dt_tr"]).dt.normalize()
        else:
            intra_flags["dt_tr_day"] = pd.to_datetime(intra_flags["date_tr"]).astype(
                "datetime64[ns]"
            )

    # interval: always present, string where available, NaN otherwise
    if "interval" not in intra_flags.columns:
        intra_flags["interval"] = pd.Series(index=intra_flags.index, dtype="string")
    else:
        intra_flags["interval"] = intra_flags["interval"].astype("string")

    signals_all = build_signal_table(daily_ind, intra_flags)
    signals = select_top_signals_per_day(signals_all)

    trades = make_trades_table_full(signals, daily_ind, hold_days=CFG["hold_days"])
    trades = apply_atr_stop(trades, daily_ind, atr_k=CFG["atr_k"])
    trades = apply_roundtrip_costs(
        trades,
        commission_rate=CFG["commission_rate"],
        bsmv_rate=CFG["bsmv_rate"],
        exchange_fee_rate=CFG["exchange_fee_rate"],
    )

    # decorate trades with interval used (or 1d if daily-only / no intraday that day)
    # robust interval join: ensure keys exist and are normalized
    if all(c in intra_flags.columns for c in ["ticker", "dt_tr_day"]):
        _iv = (
            intra_flags[["ticker", "dt_tr_day", "interval"]]
            .sort_values(["ticker", "dt_tr_day", "interval"], kind="mergesort")
            .drop_duplicates(["ticker", "dt_tr_day"], keep="first")
        )
        trades = trades.merge(
            _iv, left_on=["ticker", "entry_date"], right_on=["ticker", "dt_tr_day"], how="left"
        )

    # always expose an interval column and keep types stable
    if "interval" not in trades.columns:
        trades["interval"] = pd.Series(index=trades.index, dtype="string")
    else:
        trades["interval"] = trades["interval"].astype("string")

    # fill any remaining gaps with '1d'
    trades["interval"] = trades["interval"].fillna("1d")

    curve, active_counts = equity_curve_from_trades(daily, trades)

    curve = apply_costs_to_daily_curve(
        curve,
        trades,
        active_counts,
        commission_rate=CFG["commission_rate"],
        bsmv_rate=CFG["bsmv_rate"],
        exchange_fee_rate=CFG["exchange_fee_rate"],
    )

    trades_for_stats = trades.copy()
    if "ret_after_cost" in trades_for_stats.columns:
        trades_for_stats["ret"] = trades_for_stats["ret_after_cost"]

    ts = trade_stats(trades_for_stats)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dt = pd.Timestamp.now(tz=None)

    _run_fingerprint = {
        "ts_tag": ts_tag,
        "cfg": {k: CFG[k] for k in sorted(CFG.keys())},  # keep order stable
    }
    run_id = hashlib.sha1(json.dumps(_run_fingerprint, sort_keys=True).encode()).hexdigest()[:10]

    # Save signal tables for auditing/repro
    _date_col = (
        "date" if "date" in signals.columns else ("dt_tr" if "dt_tr" in signals.columns else None)
    )
    if _date_col is None:
        raise KeyError("Neither 'date' nor 'dt_tr' exists in signals dataframe.")

    (
        signals.assign(file_tag=ts_tag)
        .sort_values([_date_col, "ticker"], kind="mergesort")
        .to_csv(OUTDIR / f"signals_{ts_tag}.csv", index=False)
    )

    # ==== Save signals to SQLite with run_id & run_dt ====
    _date_col = (
        "date" if "date" in signals.columns else ("dt_tr" if "dt_tr" in signals.columns else None)
    )
    if _date_col is None:
        raise KeyError("Neither 'date' nor 'dt_tr' exists in signals dataframe.")

    # Normalize the datetime column name for consistency in the DB
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        # keep original column but also provide a normalized iso string for SQLite
        df = df.copy()
        if _date_col in df.columns:
            df["dt_tr_norm"] = pd.to_datetime(df[_date_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df

    sig_main = (
        _norm(signals)
        .assign(file_tag=ts_tag, run_id=run_id, run_dt=run_dt)
        .sort_values([_date_col, "ticker"], kind="mergesort")
    )

    ##db_path = OUTDIR / "signals.sqlite"
    ##db_path = sqlite3.connect(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        # Optional: improve durability/throughput
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        # Create tables if they don't exist (to ensure indices can be created)
        # pandas.to_sql will auto-create, but we add useful indexes afterward.
        sig_main.to_sql("signals", conn, if_exists="append", index=False)

        # Helpful indexes for fast filtering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_runid     ON signals(run_id);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_dt_ticker ON signals(dt_tr_norm, ticker);"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signalsall_runid  ON signals_all(run_id);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signalsall_dt_tic  ON signals_all(dt_tr_norm, ticker);"
        )
    # ==== /SQLite save ====

    # --- Benchmarks: mean / median / capped-mean ---
    benches = compute_benchmarks(daily, cap_abs=0.05)  # cap at ±5% per day (tune if you like)
    bench_mean = benches["mean"].rename(columns={"ret": "bench_ret_mean", "eq": "bench_eq_mean"})
    bench_median = benches["median"].rename(
        columns={"ret": "bench_ret_median", "eq": "bench_eq_median"}
    )
    bench_capped = benches["capped"]
    if bench_capped is not None:
        bench_capped = bench_capped.rename(
            columns={"ret": "bench_ret_capped", "eq": "bench_eq_capped"}
        )

    trades_path = OUTDIR / f"trades_3d_{ts_tag}.csv"
    curve_path = OUTDIR / f"curve_3d_{ts_tag}.csv"

    trades.to_csv(trades_path, index=False)
    curve.to_csv(curve_path, index=False)

    buckets_df = return_buckets(trades)
    buckets_path = OUTDIR / f"trade_buckets_{ts_tag}.csv"
    buckets_df.to_csv(buckets_path, index=False)

    # --- Buckets plot: count of trades per return bucket -------------------------
    if not buckets_df.empty:
        fig = plt.figure(figsize=(9, 4.5))
        ax = plt.gca()
        ax.bar(buckets_df["bucket"].astype(str), buckets_df["n"], alpha=0.9)
        ax.set_title("Trade Count by Return Bucket")
        ax.set_xlabel("Return bucket")
        ax.set_ylabel("Number of trades")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        buckets_plot_path = OUTDIR / f"trade_buckets_{ts_tag}.png"
        plt.savefig(buckets_plot_path, dpi=150)
        plt.close()
        print(f"[bist100] Buckets plot saved -> {buckets_plot_path}")

    contrib_df = contribution_report(trades, min_trades=1)
    contrib_path = OUTDIR / f"contrib_{ts_tag}.csv"
    contrib_df.to_csv(contrib_path, index=False)

    # Print a small Pareto preview
    top_n = min(10, len(contrib_df))
    print("\n[bist100] Top contributors (equal-notional proxy):")
    print(contrib_df.head(top_n).to_string(index=False))
    print(f"[bist100] Contribution table saved -> {contrib_path}")

    top = contrib_df.head(10).copy()
    top["label"] = top.apply(
        lambda r: f"{r['ticker']}  |  n={int(r['n_trades'])}  |  WR={r['win_rate']*100:.0f}%",
        axis=1,
    )

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.barh(top["label"], top["total_ret"])
    ax.invert_yaxis()

    ax.set_title("Top Contributors (equal-notional proxy)")
    ax.set_xlabel("Total Return Contribution (sum of trade returns)")
    ax.set_ylabel("Ticker | trades | win rate")

    # annotate bar values (share %)
    for i, r in top.reset_index().iterrows():
        ax.text(
            r["total_ret"] + (0.005 if r["total_ret"] >= 0 else -0.005),
            i,
            f"{r['share']*100:.1f}%",
            va="center",
            ha="left" if r["total_ret"] >= 0 else "right",
        )

    plt.tight_layout()
    contrib_plot_path = OUTDIR / f"contrib_top10_{ts_tag}.png"
    plt.savefig(contrib_plot_path, dpi=150)
    plt.close()
    print(f"[bist100] Contributors chart saved -> {contrib_plot_path}")

    # --- Bottom 10 contributors chart ---
    bottom = contrib_df.sort_values("total_ret", ascending=True).head(10).copy()
    bottom["label"] = bottom.apply(
        lambda r: f"{r['ticker']}  |  n={int(r['n_trades'])}  |  WR={r['win_rate']*100:.0f}%",
        axis=1,
    )

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.barh(bottom["label"], bottom["total_ret"])
    ax.invert_yaxis()

    ax.set_title("Bottom Contributors (equal-notional proxy)")
    ax.set_xlabel("Total Return Contribution (sum of trade returns)")
    ax.set_ylabel("Ticker | trades | win rate")

    for i, r in bottom.reset_index().iterrows():
        ax.text(
            r["total_ret"] + (0.005 if r["total_ret"] >= 0 else -0.005),
            i,
            f"{r['share']*100:.1f}%",
            va="center",
            ha="left" if r["total_ret"] >= 0 else "right",
        )

    plt.tight_layout()
    contrib_bottom_plot_path = OUTDIR / f"contrib_bottom10_{ts_tag}.png"
    plt.savefig(contrib_bottom_plot_path, dpi=150)
    plt.close()
    print(f"[bist100] Contributors (bottom-10) chart saved -> {contrib_bottom_plot_path}")

    # print quick headline stats
    overall_win_rate = (trades["ret"] > 0).mean()
    print(f"[bist100] Overall win rate: {overall_win_rate:.2%}")
    print(f"[bist100] Buckets saved -> {buckets_path}")

    # Merge all for plotting
    merged_plot = curve.merge(bench_mean[["dt_tr", "bench_eq_mean"]], on="dt_tr", how="outer")
    merged_plot = merged_plot.merge(
        bench_median[["dt_tr", "bench_eq_median"]], on="dt_tr", how="outer"
    )
    if bench_capped is not None:
        merged_plot = merged_plot.merge(
            bench_capped[["dt_tr", "bench_eq_capped"]], on="dt_tr", how="outer"
        )

    merged_plot = merged_plot.sort_values("dt_tr").ffill()

    _plot_cols = ["equity", "bench_eq_mean", "bench_eq_median"]
    if "bench_eq_capped" in merged_plot.columns:
        _plot_cols.append("bench_eq_capped")

    # Find the last date where every plotted series is non-NaN
    _last_full_row = merged_plot.dropna(subset=_plot_cols).tail(1)
    if not _last_full_row.empty:
        _last_good_dt = _last_full_row["dt_tr"].iloc[0]
        merged_plot = merged_plot[merged_plot["dt_tr"] <= _last_good_dt]

    # --- Use the same cutoff for the strategy curve and metrics ---
    plot_cutoff = (
        _last_good_dt
        if "_last_good_dt" in locals() and pd.notna(_last_good_dt)
        else merged_plot["dt_tr"].max()
    )

    # Clip the strategy curve to the plot cutoff (canonical series for both chart + metrics)
    curve_plot = curve[curve["dt_tr"] <= plot_cutoff].copy()

    # === Inflation adjustment (TUFE) ==============================================
    # Build CPI index over the final plotting span (use the merged plot window)
    _span_start = pd.to_datetime(merged_plot["dt_tr"].min()).normalize()
    _span_end = pd.to_datetime(merged_plot["dt_tr"].max()).normalize()
    cpi_daily = _tufe_daily_index(_span_start, _span_end)

    # Normalize date dtypes (avoid tz/clock mismatches)
    for _df in (curve_plot, merged_plot, cpi_daily):
        _df["dt_tr"] = pd.to_datetime(_df["dt_tr"]).dt.normalize()

    # Merge CPI into both nominal series (strategy & benchmarks)
    curve_plot = curve_plot.merge(cpi_daily, on="dt_tr", how="left").sort_values("dt_tr")
    merged_plot = merged_plot.merge(cpi_daily, on="dt_tr", how="left").sort_values("dt_tr")

    # Safety: forward/back fill any gaps that might appear after the merge
    curve_plot["cpi_index"] = curve_plot["cpi_index"].ffill().bfill()
    merged_plot["cpi_index"] = merged_plot["cpi_index"].ffill().bfill()

    # Real (deflated) series: divide nominal equity by CPI index
    curve_plot["equity_real"] = curve_plot["equity"] / curve_plot["cpi_index"]

    if "bench_eq_mean" in merged_plot:
        merged_plot["bench_eq_mean_real"] = merged_plot["bench_eq_mean"] / merged_plot["cpi_index"]
    if "bench_eq_median" in merged_plot:
        merged_plot["bench_eq_median_real"] = (
            merged_plot["bench_eq_median"] / merged_plot["cpi_index"]
        )
    if "bench_eq_capped" in merged_plot:
        merged_plot["bench_eq_capped_real"] = (
            merged_plot["bench_eq_capped"] / merged_plot["cpi_index"]
        )

    # Defensive: if everything somehow NaN, don’t crash later
    if curve_plot["equity_real"].notna().sum() == 0:
        print(
            "[warn] equity_real is empty after CPI merge — check TUFE dates; plotting nominal instead."
        )
        curve_plot["equity_real"] = curve_plot["equity"]
    # ==============================================================================

    # Recompute portfolio metrics
    m = portfolio_metrics(curve_plot)

    # --- Real (CPI-deflated) metrics --------------------------------------------
    # Build a minimal curve with real equity + real daily returns
    _curve_real = pd.DataFrame({"dt_tr": curve_plot["dt_tr"].values})
    _curve_real["equity"] = curve_plot["equity_real"].values
    _curve_real["daily_port_ret"] = _curve_real["equity"].pct_change().fillna(0.0).values
    _curve_real["rolling_max"] = _curve_real["equity"].cummax()
    _curve_real["drawdown"] = (_curve_real["equity"] / _curve_real["rolling_max"]) - 1.0
    m_real = portfolio_metrics(_curve_real)

    print("== Portfolio metrics ==")
    print({"CAGR": m["CAGR"], "Sharpe": m["Sharpe"], "MaxDD": m["MaxDD"]})
    print("== Real (CPI-deflated) metrics ==")
    print(
        {
            "CAGR_real": m_real["CAGR"],
            "Sharpe_real": m_real["Sharpe"],
            "MaxDD_real": m_real["MaxDD"],
        }
    )

    # Cut before first trade to remove flatline
    first_trade_date = trades["entry_date"].min()
    if pd.notna(first_trade_date):
        merged_plot = merged_plot[merged_plot["dt_tr"] >= pd.to_datetime(first_trade_date)]

    plt.figure()
    plt.plot(curve_plot["dt_tr"], curve_plot["equity"], label="Strategy 3d")
    plt.plot(merged_plot["dt_tr"], merged_plot["bench_eq_mean"], label="Benchmark EW (mean)")
    plt.plot(merged_plot["dt_tr"], merged_plot["bench_eq_median"], label="Benchmark EW (median)")
    if "bench_eq_capped" in merged_plot.columns:
        plt.plot(
            merged_plot["dt_tr"], merged_plot["bench_eq_capped"], label="Benchmark EW (capped ±5%)"
        )

    plt.legend()
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity (Start=1.0)")
    plt.tight_layout()

    plot_path = OUTDIR / f"equity_curve_{ts_tag}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[bist100] Plot saved to {plot_path}")

    # --- Real-only (inflation-adjusted) equity chart -----------------------------
    _has_real = curve_plot["equity_real"].notna().any()
    if not _has_real:
        print(
            "[warn] No real equity points to plot — falling back to nominal curves for the real-only chart."
        )

    plt.figure()

    # Strategy (real)
    plt.plot(curve_plot["dt_tr"], curve_plot["equity_real"], label="Strategy 3d (real)")

    # Benchmarks (real)
    plt.plot(
        merged_plot["dt_tr"], merged_plot["bench_eq_mean_real"], label="Benchmark EW (mean, real)"
    )
    plt.plot(
        merged_plot["dt_tr"],
        merged_plot["bench_eq_median_real"],
        label="Benchmark EW (median, real)",
    )
    if "bench_eq_capped_real" in merged_plot.columns:
        plt.plot(
            merged_plot["dt_tr"],
            merged_plot["bench_eq_capped_real"],
            label="Benchmark EW (capped ±5%, real)",
        )

    plt.legend()
    plt.title("Equity Curves (Real, CPI-deflated)")
    plt.xlabel("Date")
    plt.ylabel("Equity (Start=1.0, real)")
    plt.tight_layout()

    plot_path_real = OUTDIR / f"equity_curve_real_{ts_tag}.png"
    plt.savefig(plot_path_real, dpi=150)
    plt.close()
    print(f"[bist100] Real (CPI-deflated) plot saved to {plot_path_real}")
    # ---------------------------------------------------------------------------

    sig = trades.dropna(subset=["entry_date"]).copy()
    sig["month"] = pd.to_datetime(sig["entry_date"]).dt.to_period("M").dt.to_timestamp()
    sig = sig.sort_values(
        ["month", "interval", "entry_date", "ticker"], kind="mergesort"
    ).reset_index(drop=True)
    sig["rank_in_month"] = sig.groupby(["month", "interval"]).cumcount()

    plt.figure(figsize=(11, 5))

    for iv, sub in sig.sort_values("month").groupby("interval"):
        plt.scatter(sub["month"], sub["rank_in_month"], s=20, alpha=0.8, label=iv)

    plt.title("Signals by Month (colored by interval)")
    plt.xlabel("Month")
    plt.ylabel("count within month (dots)")
    plt.legend(title="Interval", ncol=4)
    plt.tight_layout()

    sig_plot_path = OUTDIR / f"signals_scatter_{ts_tag}.png"
    plt.savefig(sig_plot_path, dpi=150)
    plt.close()
    print(f"[bist100] Signals-by-month chart saved -> {sig_plot_path}")

    # Save benchmark CSVs too
    bench_mean.to_csv(OUTDIR / f"benchmark_mean_{ts_tag}.csv", index=False)
    bench_median.to_csv(OUTDIR / f"benchmark_median_{ts_tag}.csv", index=False)
    if bench_capped is not None:
        bench_capped.to_csv(OUTDIR / f"benchmark_capped_{ts_tag}.csv", index=False)

    write_rules_daily(signals, db_path=DB_PATH)
    log_run_to_db(
        DB_PATH,
        "baseline_v2",
        CFG,
        m,
        n_trades=len(trades),
        est_turnover=trades.shape[0] / max(1, curve.shape[0]),
        note="prox=0.2%, vol=1.2x, ORB=90m, VWAP strict, cap=30, ATR=2x",
    )

    p_cfg = PortfolioConfig(
        initial_cash=100_000.0,
        max_names=CFG["exposure_cap"],
        slip_frac=0.0,
        commission_rate=CFG["commission_rate"],
        bsmv_rate=CFG["bsmv_rate"],
        exchange_fee_rate=CFG["exchange_fee_rate"],
    )

    # enforce stable, deterministic order
    trades = trades.sort_values(
        ["entry_date", "ticker"], kind="mergesort"  # stable sort
    ).reset_index(drop=True)

    execs_df, ledg_df = build_virtual_portfolio(trades, daily, p_cfg)

    # save
    execs_df.to_csv(OUTDIR / f"executions_{ts_tag}.csv", index=False)
    ledg_df.to_csv(OUTDIR / f"portfolio_{ts_tag}.csv", index=False)
    print(f"[bist100] Portfolio saved -> executions_{ts_tag}.csv, portfolio_{ts_tag}.csv")

    con = sqlite3.connect(DB_PATH)
    execs_df.to_sql("executions", con, if_exists="append", index=False)
    ledg_df.to_sql("portfolio_ledger", con, if_exists="append", index=False)
    con.close()

    # === Nominal vs Real ledger equity (overlay plots) ============================
    if ledg_df is not None and not ledg_df.empty:
        # Build CPI over the ledger span
        _ls, _le = ledg_df["dt_tr"].min().normalize(), ledg_df["dt_tr"].max().normalize()
        cpi_leg = _tufe_daily_index(_ls, _le)
        tmp = ledg_df.copy()
        tmp["dt_tr"] = pd.to_datetime(tmp["dt_tr"]).dt.normalize()
        tmp = tmp.merge(cpi_leg, on="dt_tr", how="left").sort_values("dt_tr")
        tmp["cpi_index"] = tmp["cpi_index"].ffill().bfill()

        # Real (deflated) ledger equity
        tmp["equity_real"] = tmp["equity"] / tmp["cpi_index"]

        # Plot nominal ledger vs. scaled nominal index (₺ terms) and real ledger
        # Scale the nominal research curve by 100k so axes match TRY
        curve_scaled = curve_plot.copy()[["dt_tr", "equity"]].sort_values("dt_tr")
        curve_scaled["equity_try"] = 100_000.0 * curve_scaled["equity"]

        # Extend research curve to cover the full ledger span
        # (research curve was trimmed at plot_cutoff; ledger can be longer)
        curve_scaled = (
            curve_scaled.set_index("dt_tr")
            .reindex(tmp["dt_tr"])  # align to ledger dates
            .ffill()  # hold last research value forward
            .reset_index()
            .rename(columns={"index": "dt_tr"})
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tmp["dt_tr"], tmp["equity"], label="Ledger equity (nominal)")
        ax.plot(tmp["dt_tr"], tmp["equity_real"], label="Ledger equity (real, CPI-deflated)")
        ax.plot(
            curve_scaled["dt_tr"],
            curve_scaled["equity_try"],
            alpha=0.5,
            linestyle="--",
            label="Research index × ₺100k (nominal)",
        )

        # after plotting
        ymax = (
            max(tmp["equity"].max(), tmp["equity_real"].max(), curve_scaled["equity_try"].max())
            * 1.1
        )
        ax.set_ylim(0, ymax)  # focuses on ledger

        ax.set_title("Ledger vs Research Index (nominal & real)")
        ax.set_xlabel("Date")
        ax.set_ylabel("TRY")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=24))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        plt.legend()
        plt.tight_layout()
        out_path = OUTDIR / f"ledger_vs_index_{ts_tag}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[bist100] Ledger/index comparison saved -> {out_path}")
    # ============================================================================

    # --- Monthly portfolio breakdown (line: equity; bars: P&L/Costs) ---
    try:
        execs_df = pd.read_csv(OUTDIR / f"executions_{ts_tag}.csv", parse_dates=["dt_tr"])
        ledg_df = pd.read_csv(OUTDIR / f"portfolio_{ts_tag}.csv", parse_dates=["dt_tr"])
    except Exception:
        execs_df = None
        ledg_df = None

    if execs_df is not None and ledg_df is not None and not ledg_df.empty:
        # fees per execution (if not stored)
        if "fee" not in execs_df.columns:
            fee_rate = CFG["commission_rate"] * (1.0 + CFG["bsmv_rate"]) + CFG["exchange_fee_rate"]
            execs_df["fee"] = execs_df["price"] * execs_df["qty"] * fee_rate

        ledg_df["month"] = ledg_df["dt_tr"].dt.to_period("M").dt.to_timestamp()
        execs_df["month"] = execs_df["dt_tr"].dt.to_period("M").dt.to_timestamp()

        # end-of-month equity, monthly fees
        eq_m = ledg_df.sort_values("dt_tr").groupby("month")["equity"].last()
        fee_m = execs_df.groupby("month")["fee"].sum().reindex(eq_m.index).fillna(0.0)

        if len(eq_m) > 0:
            # equity change (delta) and gross pnl
            initial_equity = 100_000.0
            delta = eq_m.diff()
            if pd.isna(delta.iloc[0]):
                delta.iloc[0] = eq_m.iloc[0] - initial_equity
            gross_m = (delta + fee_m).rename("gross_pnl")

            dfm = pd.concat([eq_m.rename("equity"), gross_m, fee_m.rename("fees")], axis=1)

            x = dfm.index.to_pydatetime()

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(x, dfm["equity"], label="Equity (EOM)")
            ax.bar(x, dfm["gross_pnl"], width=20, label="Gross P&L")
            ax.bar(x, -dfm["fees"], width=20, label="Costs (fees+BSMV+exch)")

            ax.set_title("Monthly Portfolio: Equity line + P&L/Costs bars")
            ax.set_xlabel("Month")
            ax.set_ylabel("TRY")

            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=24))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            plt.tight_layout()

            wf_path = OUTDIR / f"portfolio_monthly_{ts_tag}.png"
            plt.savefig(wf_path, dpi=150)
            plt.close()
            print(f"[bist100] Portfolio monthly chart saved -> {wf_path}")

    # --- True Waterfall: one bar per month with cumulative steps ---
    if execs_df is not None and ledg_df is not None and not ledg_df.empty:
        eq_m = (
            ledg_df.sort_values("dt_tr")
            .groupby(ledg_df["dt_tr"].dt.to_period("M").dt.to_timestamp())["equity"]
            .last()
        )
        fee_m = (
            execs_df.groupby(execs_df["dt_tr"].dt.to_period("M").dt.to_timestamp())["fee"]
            .sum()
            .reindex(eq_m.index)
            .fillna(0.0)
        )

        if len(eq_m) > 0:
            initial_equity = 100_000.0
            net_pnl_m = eq_m.diff()
            if pd.isna(net_pnl_m.iloc[0]):
                net_pnl_m.iloc[0] = eq_m.iloc[0] - initial_equity
            cum = initial_equity + net_pnl_m.cumsum()

            x = eq_m.index.to_pydatetime()
            colors = np.where(net_pnl_m.values >= 0, "#3BAA3B", "#D9534F")

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(
                x, net_pnl_m.values, width=20, color=colors, alpha=0.9, label="Net P&L (monthly)"
            )
            ax.step(x, cum.values, where="mid", linewidth=2, label="Equity (step)")

            # overlay fees
            if not fee_m.empty:
                ax.bar(x, -fee_m.values, width=8, color="#A94442", alpha=0.6, label="Costs")

            ax.set_title("Portfolio Waterfall (monthly steps)")
            ax.set_xlabel("Month")
            ax.set_ylabel("TRY")

            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=24))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            plt.tight_layout()

            waterfall_path = OUTDIR / f"portfolio_waterfall_{ts_tag}.png"
            plt.savefig(waterfall_path, dpi=150)
            plt.close()
            print(f"[bist100] Portfolio waterfall chart saved -> {waterfall_path}")

    plot_signals_scatter(trades, OUTDIR, ts_tag)

    return {
        "metrics": m,  # dict from portfolio_metrics(...)
        "trade_stats": ts,  # dict from trade_stats(...)
        "files": [str(trades_path), str(curve_path), str(plot_path)],
    }
