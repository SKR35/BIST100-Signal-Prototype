import numpy as np
import pandas as pd


def make_trades_table_full(
    signals: pd.DataFrame, daily_df: pd.DataFrame, hold_days: int
) -> pd.DataFrame:
    d = signals.sort_values(["ticker", "dt_tr"]).copy()
    d["next_open"] = d.groupby("ticker")["open"].shift(-1)
    d["exit_close"] = d.groupby("ticker")["close"].shift(-hold_days)
    d["exit_date"] = d.groupby("ticker")["dt_tr"].shift(-hold_days)

    trades = d[(d["signal"] == 1) & (d["selected"] == 1)][
        ["ticker", "dt_tr", "next_open", "exit_close", "exit_date"]
    ].copy()
    trades = trades.rename(columns={"dt_tr": "entry_date"}).reset_index(drop=True)

    daily = daily_df.copy()
    daily["dt_tr"] = pd.to_datetime(daily["dt_tr"])

    entry_info = daily.rename(
        columns={
            "open": "entry_open",
            "high": "entry_high",
            "low": "entry_low",
            "close": "entry_close",
            "volume": "entry_volume",
        }
    )
    trades = trades.merge(
        entry_info[
            [
                "ticker",
                "dt_tr",
                "entry_open",
                "entry_high",
                "entry_low",
                "entry_close",
                "entry_volume",
            ]
        ],
        left_on=["ticker", "entry_date"],
        right_on=["ticker", "dt_tr"],
        how="left",
    ).drop(columns=["dt_tr"])

    exit_info = daily.rename(
        columns={
            "open": "exit_open",
            "high": "exit_high",
            "low": "exit_low",
            "close": "exit_close_full",
            "volume": "exit_volume",
        }
    )
    trades = trades.merge(
        exit_info[
            [
                "ticker",
                "dt_tr",
                "exit_open",
                "exit_high",
                "exit_low",
                "exit_close_full",
                "exit_volume",
            ]
        ],
        left_on=["ticker", "exit_date"],
        right_on=["ticker", "dt_tr"],
        how="left",
    ).drop(columns=["dt_tr"])

    trades["status"] = np.where(trades["exit_close"].isna(), "OPEN", "CLOSED")
    trades["ret"] = np.where(
        trades["status"] == "CLOSED", trades["exit_close"] / trades["next_open"] - 1, np.nan
    )
    # trades = trades.fillna(999999999)
    return trades


def apply_roundtrip_costs(
    trades: pd.DataFrame, commission_rate: float, bsmv_rate: float, exchange_fee_rate: float
) -> pd.DataFrame:
    """
    Subtract costs from trade returns (equal-notional proxy).
    Total cost = 2 * [commission_rate * (1 + bsmv_rate) + exchange_fee_rate]
    Applied only to CLOSED trades; OPEN keep NaN.
    """
    t = trades.copy()
    roundtrip_cost = 2.0 * (commission_rate * (1.0 + bsmv_rate) + exchange_fee_rate)
    mask = t["ret"].notna()
    t.loc[mask, "ret_after_cost"] = t.loc[mask, "ret"] - roundtrip_cost
    return t


def equity_curve_from_trades(daily: pd.DataFrame, trades: pd.DataFrame):
    """
    Build a strategy equity curve by:
      1) computing daily close-to-close returns per ticker from `daily`
      2) aggregating ONLY on the days each trade is active
         (entry_date < day <= exit_date) using equal-notional weights
    Returns:
      curve_df: DataFrame(dt_tr, equity)
      active_counts: Series(index=dt_tr) = number of open positions per day
    """

    # --- 1) daily returns matrix ------------------------------------------------
    # --- make daily prices unique per (dt_tr, ticker) -----------------------------
    # keep only what we need
    d = daily[["dt_tr", "ticker", "close"]].copy()

    # normalize and stable-sort so groupby is deterministic and repeatable
    d["dt_tr"] = pd.to_datetime(d["dt_tr"]).dt.normalize()

    # if the DB provides multiple rows per (dt_tr, ticker), collapse to one:
    # take the LAST close of the day
    d = d.groupby(["dt_tr", "ticker"], as_index=False, sort=False).last()

    # quick guardrail: verify we truly have one row per (dt_tr,ticker)
    _dups = d.duplicated(["dt_tr", "ticker"]).sum()
    if _dups:
        print(
            f"[warn] equity_curve_from_trades: {int(_dups)} duplicate daily rows remained after dedup."
        )

    # compute close-to-close returns per ticker on the deduped, stably-sorted data
    d = d.sort_values(["ticker", "dt_tr"], kind="mergesort").reset_index(drop=True)
    d["ret_cc"] = d.groupby("ticker", sort=False)["close"].pct_change()

    # robust pivot (tolerates any accidental duplicates)
    R = d.pivot_table(index="dt_tr", columns="ticker", values="ret_cc", aggfunc="mean").sort_index()

    # guardrail (usually unnecessary now, but harmless)
    if R.index.duplicated().any():
        R = R[~R.index.duplicated(keep="first")]
    # ------------------------------------------------------------------------------

    # --- 2) loop trades to accumulate portfolio returns when positions are open -
    # prepare accumulators aligned to calendar trading days
    port_ret = pd.Series(0.0, index=R.index, dtype=float)
    active_counts = pd.Series(0.0, index=R.index, dtype=float)

    # canonicalize dates in trades
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
    t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.normalize()

    for _, row in t.iterrows():
        tkr = row["ticker"]
        if tkr not in R.columns:
            continue  # safety: skip names not in daily
        # active on (entry_date, exit_date]  i.e., after entry close, until exit close
        m = (R.index > row["entry_date"]) & (R.index <= row["exit_date"])
        if not m.any():
            continue
        # accumulate strategy return & open-count
        port_ret.loc[m] += R.loc[m, tkr].fillna(0.0).values
        active_counts.loc[m] += 1.0

    # equal-notional: average of active legs; flat (0) when no open position
    with np.errstate(divide="ignore", invalid="ignore"):
        strat_ret = np.where(
            active_counts.values > 0, port_ret.values / np.maximum(active_counts.values, 1.0), 0.0
        )
    strat_ret = pd.Series(strat_ret, index=R.index)

    # build DataFrame with daily portfolio return AND equity
    curve_df = pd.DataFrame({"dt_tr": R.index, "daily_port_ret": strat_ret.values})
    curve_df["equity"] = (1.0 + curve_df["daily_port_ret"]).cumprod()

    return curve_df, active_counts


def holdings_diary(daily_df, trades):
    diary = []
    for _, tr in trades.iterrows():
        entry, exit_ = pd.to_datetime(tr["entry_date"]), tr["exit_date"]
        tkr = tr["ticker"]
        status = tr["status"]
        path = daily_df[(daily_df["ticker"] == tkr) & (daily_df["dt_tr"] >= entry)]
        if status == "CLOSED":
            path = path[path["dt_tr"] <= exit_]
        for _, r in path.iterrows():
            diary.append({"date": r["dt_tr"], "ticker": tkr, "position": 1, "price": r["close"]})
    df = pd.DataFrame(diary)
    return df.groupby("date").size().reset_index(name="n_positions")


def apply_costs_to_daily_curve(
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    active_counts: pd.Series,
    commission_rate: float,
    bsmv_rate: float,
    exchange_fee_rate: float,
) -> pd.DataFrame:
    """
    Subtract round-trip trading cost on the first active day of each trade.
    We deduct cost / active_count_on_that_day from the portfolio daily return,
    consistent with equal-weight average of active positions.
    """
    c = curve.copy()

    # Ensure daily returns exist; if not, derive them from equity
    if "daily_port_ret" not in c.columns:
        c = c.sort_values("dt_tr").reset_index(drop=True)
        c["daily_port_ret"] = c["equity"].pct_change().fillna(0.0)

    c_idx = c["dt_tr"].reset_index(drop=True)

    # ---- align active_counts to the curve calendar & force integer
    if isinstance(active_counts, pd.Series):
        ac = active_counts.copy()
    else:
        # if someone passed a DataFrame, take the first column
        ac = pd.Series(active_counts.iloc[:, 0].values, index=active_counts.index)

    # make sure the index is the same calendar as the curve
    ac.index = pd.to_datetime(ac.index)
    # reindex to curve calendar; fill missing with 0
    ac = ac.reindex(pd.to_datetime(c["dt_tr"].values)).fillna(0)
    # store back as int Series (and keep a fast .iloc access path)
    active_counts = ac.astype(int)
    # precompute round-trip fraction
    roundtrip_cost = 2.0 * (commission_rate * (1.0 + bsmv_rate) + exchange_fee_rate)

    # Start accruing the trade from the day AFTER entry_date (same as equity_curve_from_trades)
    for _, r in trades.iterrows():
        if pd.isna(r.get("entry_date")):
            continue
        # entry_effect_day = next calendar day
        entry_effect = pd.to_datetime(r["entry_date"]) + pd.Timedelta(days=1)
        pos = c_idx.searchsorted(entry_effect)
        if 0 <= pos < len(c):
            denom = float(active_counts.iloc[pos]) if active_counts.iloc[pos] > 0 else 1.0
            c.loc[pos, "daily_port_ret"] = c.loc[pos, "daily_port_ret"] - (roundtrip_cost / denom)

    # Rebuild equity/drawdown after costs
    c["equity"] = (1.0 + c["daily_port_ret"]).cumprod()
    c["rolling_max"] = c["equity"].cummax()
    c["drawdown"] = (c["equity"] / c["rolling_max"]) - 1.0
    return c
