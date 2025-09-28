import math

import numpy as np
import pandas as pd


def portfolio_metrics(curve):
    daily_ret = curve["daily_port_ret"]
    ann_factor = 252.0
    cagr = (
        (curve["equity"].iloc[-1]) ** (ann_factor / len(curve)) - 1.0 if len(curve) > 0 else np.nan
    )
    sharpe = (
        (daily_ret.mean() / (daily_ret.std() + 1e-12)) * math.sqrt(252.0)
        if daily_ret.std() > 0
        else np.nan
    )
    max_dd = curve["drawdown"].min() if len(curve) > 0 else np.nan
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}


def trade_stats(trades):
    if len(trades) == 0:
        return {
            "n": 0,
            "hit_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "avg_ret": np.nan,
        }
    n = len(trades)
    wins = (trades["ret"] > 0).sum()
    hit = wins / n
    avg_ret = trades["ret"].mean()
    avg_win = trades.loc[trades["ret"] > 0, "ret"].mean() if wins > 0 else np.nan
    avg_loss = trades.loc[trades["ret"] <= 0, "ret"].mean() if wins < n else np.nan
    return {"n": n, "hit_rate": hit, "avg_win": avg_win, "avg_loss": avg_loss, "avg_ret": avg_ret}


def return_buckets(
    trades: pd.DataFrame, bins=(-1.0, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 1.0)
):
    """
    Bucket trade returns and summarize distribution.
    Returns a DataFrame with columns:
    ['bucket','n','share','avg_ret','ret_sum','cum_share','wins','win_rate_bucket']
    """
    t = trades.copy()
    t = t[np.isfinite(t["ret"])]  # drop NaNs if any
    if t.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "n",
                "share",
                "avg_ret",
                "ret_sum",
                "cum_share",
                "wins",
                "win_rate_bucket",
            ]
        )

    labels = [f"{int(bins[i]*100)}% to {int(bins[i+1]*100)}%" for i in range(len(bins) - 1)]
    t["bucket"] = pd.cut(t["ret"].values, bins=bins, labels=labels, include_lowest=True)
    g = t.groupby("bucket", dropna=False)

    out = g["ret"].agg(n="count", avg_ret="mean", ret_sum="sum").reset_index()
    out["wins"] = g.apply(lambda x: (x["ret"] > 0).sum()).to_list()
    out["win_rate_bucket"] = out["wins"] / out["n"].replace(0, np.nan)
    total_n = out["n"].sum()
    out["share"] = out["n"] / total_n
    out["cum_share"] = out["share"].cumsum()
    # nice ordering from worst to best
    return out.reset_index(drop=True)


def contribution_report(trades: pd.DataFrame, min_trades: int = 1) -> pd.DataFrame:
    """
    Per-ticker P&L contribution table (equal-notional proxy).
    Returns columns:
    ['ticker','n_trades','wins','win_rate','avg_ret','total_ret','share','cum_share']
    """
    t = trades.copy()
    # only CLOSED trades have realized returns
    t = t[np.isfinite(t["ret"])].copy()
    if t.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "n_trades",
                "wins",
                "win_rate",
                "avg_ret",
                "total_ret",
                "share",
                "cum_share",
            ]
        )

    g = t.groupby("ticker")
    df = pd.DataFrame(
        {
            "n_trades": g["ret"].size(),
            "wins": g.apply(lambda x: (x["ret"] > 0).sum()),
            "avg_ret": g["ret"].mean(),
            "total_ret": g["ret"].sum(),
        }
    ).reset_index()

    df = df[df["n_trades"] >= int(min_trades)].copy()
    df["win_rate"] = df["wins"] / df["n_trades"]
    df = df.sort_values("total_ret", ascending=False).reset_index(drop=True)

    tot = df["total_ret"].sum()
    df["share"] = df["total_ret"] / (tot if tot != 0 else np.nan)
    df["cum_share"] = df["share"].cumsum()
    return df
