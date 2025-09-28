import pandas as pd

from .config import CFG


def compute_daily_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["ticker", "dt_tr"]).copy()
    g = d.groupby("ticker")

    d["vol_ma20"] = g["volume"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    d["ma200"] = g["close"].rolling(200, min_periods=1).mean().reset_index(level=0, drop=True)
    d["high_252"] = g["high"].rolling(252, min_periods=1).max().reset_index(level=0, drop=True)

    d["prox_to_52w"] = d["close"] / d["high_252"] - 1.0
    d["vol_ratio"] = d["volume"] / d["vol_ma20"]

    d["turnover_try"] = d["close"] * d["volume"]
    d["adv20"] = g["turnover_try"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)

    d["prev_close"] = g["close"].shift(1)
    tr = pd.concat(
        [
            (d["high"] - d["low"]).abs(),
            (d["high"] - d["prev_close"]).abs(),
            (d["low"] - d["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr14"] = g.apply(lambda x: tr.loc[x.index].rolling(14, min_periods=1).mean()).reset_index(
        level=0, drop=True
    )

    d["near_52w_high"] = (d["close"] >= (1 - CFG["prox_pct"]) * d["high_252"]).astype(int)
    d["ma200_uptrend"] = (d["close"] >= d["ma200"]).astype(int)
    d["vol_surge"] = (d["vol_ratio"] >= CFG["vol_mult"]).astype(int)
    return d
