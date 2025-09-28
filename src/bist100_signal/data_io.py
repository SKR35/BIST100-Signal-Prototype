import os
import sqlite3

import pandas as pd

from .config import DB_PATH


def load_daily(db_path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {os.path.abspath(db_path)}")
    con = sqlite3.connect(db_path)
    q = """
    SELECT ticker, datetime_tr AS dt_tr, datetime_utc AS dt_utc,
           open, high, low, close, volume
    FROM prices
    WHERE interval='1d'
    """
    df = pd.read_sql(q, con, parse_dates=["dt_tr", "dt_utc"])
    df = df.sort_values(["ticker", "dt_tr"]).reset_index(drop=True)
    con.close()

    df = df.sort_values(["ticker", "dt_tr"], kind="mergesort").reset_index(drop=True)
    return df


def load_intraday_30m(db_path: str = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    q = """
    SELECT ticker, datetime_tr AS dt_tr, datetime_utc AS dt_utc,
           open, high, low, close, volume
    FROM prices
    WHERE interval='30m'
    """
    df = pd.read_sql(q, con, parse_dates=["dt_tr", "dt_utc"])
    df = df.sort_values(["ticker", "dt_tr"]).reset_index(drop=True)
    con.close()
    df["dt_tr"] = pd.to_datetime(df["dt_tr"]).dt.normalize()
    df = df.sort_values(["ticker", "dt_tr"], kind="mergesort").reset_index(drop=True)
    return df


def load_intraday_best(db_path: str = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    q = """
    SELECT ticker, interval, datetime_tr AS dt_tr, datetime_utc AS dt_utc,
           open, high, low, close, volume
    FROM prices
    WHERE interval IN ('1m','5m','30m','60m','4h')
    """
    df = pd.read_sql(q, con, parse_dates=["dt_tr", "dt_utc"])
    con.close()
    if df.empty:
        raise RuntimeError("No intraday rows found for intervals 1m/5m/30m/60m/4h.")
    # Prefer the smallest bar (highest resolution) on any overlapping timestamps
    pref = {"1m": 0, "5m": 1, "30m": 2, "60m": 3, "4h": 4}
    df["_rank"] = df["interval"].map(pref).fillna(9)
    df = (
        df.sort_values(["ticker", "dt_tr", "_rank"])
        .drop_duplicates(["ticker", "dt_tr"], keep="first")
        .drop(columns=["_rank"])
        .sort_values(["ticker", "dt_tr"])
        .reset_index(drop=True)
    )
    df["dt_tr"] = pd.to_datetime(df["dt_tr"]).dt.normalize()
    df = df.sort_values(["ticker", "dt_tr"], kind="mergesort").reset_index(drop=True)
    return df
