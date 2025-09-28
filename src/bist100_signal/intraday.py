from math import ceil

import numpy as np
import pandas as pd


def _infer_bar_minutes(df: pd.DataFrame) -> int:
    # infer median bar size in minutes per ticker/day, then take global median
    g = df.sort_values(["ticker", "dt_tr"]).groupby("ticker")
    diffs = (g["dt_tr"].diff().dt.total_seconds() / 60.0).dropna()
    if diffs.empty:
        return 30
    minutes = int(np.median(diffs.values))
    return max(1, minutes)


def compute_intraday_confirmations(
    mbars: pd.DataFrame,
    orb_minutes: int | None = None,
    window_bars: int | None = None,
    vwap_required: bool = True,
    force_min_bar: int | None = None,
) -> pd.DataFrame:

    m = mbars.sort_values(["ticker", "dt_tr"]).reset_index(drop=True).copy()
    m["date_tr"] = m["dt_tr"].dt.date

    bar_minutes = _infer_bar_minutes(m)

    # enforce a minimum bar size if requested (e.g., 240=4h for the recent window)
    if force_min_bar is not None:
        bar_minutes = max(int(bar_minutes), int(force_min_bar))

    # If our raw bars are thinner than the enforced minimum, aggregate up.
    if force_min_bar is not None and _infer_bar_minutes(m) < int(force_min_bar):
        agg_min = int(force_min_bar)
        # 1) ensure having a timestamp column for grouping (tolerant)
        _possible_ts = ["dt", "ts", "timestamp", "datetime", "dt_utc", "ts_utc", "ts_local"]
        dtcol = next((c for c in _possible_ts if c in m.columns), None)

        if dtcol is None:
            # accept a DatetimeIndex too
            if isinstance(m.index, pd.DatetimeIndex):
                m = m.copy()
                m["dt"] = m.index.tz_convert(None) if getattr(m.index, "tz", None) else m.index
                dtcol = "dt"
            else:
                raise ValueError(
                    f"intraday frame must have a timestamp col; tried {_possible_ts} "
                    f"or a DatetimeIndex; got columns={list(m.columns)}"
                )

        m = m.copy()
        m[dtcol] = pd.to_datetime(m[dtcol])

        # 2) resample per ticker into agg_minute bins
        #    open=first, high=max, low=min, close=last, volume=sum
        freq = f"{agg_min}T"

        # ... already chosen dtcol, set bar_minutes, etc. ...

        def _resample(df):
            # get this group's ticker (works whether 'ticker' is a column or via df.name)
            tkr = df["ticker"].iloc[0] if "ticker" in df.columns else getattr(df, "name", None)

            df = df.set_index(dtcol)
            out = (
                df.resample(freq, label="right", closed="right")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna(subset=["open", "close"])
                .reset_index()
            )

            if tkr is not None:
                out["ticker"] = tkr
            return out

        # resample per ticker and keep ticker in the result
        m = m.groupby("ticker", group_keys=False).apply(_resample).reset_index(drop=True)

        # re-derive time helpers needed later
        m["dt_tr"] = pd.to_datetime(m[dtcol])
        m["date_tr"] = m["dt_tr"].dt.date

        bar_minutes = agg_min

    if orb_minutes is not None:
        win_bars = ceil(float(orb_minutes) / float(bar_minutes))
    elif window_bars is not None:
        win_bars = int(window_bars)
    else:
        win_bars = ceil(90.0 / float(bar_minutes))  # default

    m["rn"] = m.groupby(["ticker", "date_tr"])["dt_tr"].rank(method="first")

    or_mask = m["rn"] <= float(win_bars)
    or_high = m[or_mask].groupby(["ticker", "date_tr"])["high"].max().rename("or_high")
    or_low = m[or_mask].groupby(["ticker", "date_tr"])["low"].min().rename("or_low")

    last_idx = m.groupby(["ticker", "date_tr"])["dt_tr"].idxmax()
    lastbars = (
        m.loc[last_idx, ["ticker", "date_tr", "close"]]
        .rename(columns={"close": "close_last"})
        .set_index(["ticker", "date_tr"])
    )

    m["tp"] = (m["high"] + m["low"] + m["close"]) / 3.0

    def _vwap(df):
        px = df["close"].astype(float)
        w = df["volume"].astype(float)
        denom = np.nansum(w)
        if not np.isfinite(denom) or denom <= 0:
            return np.nan
        return np.nansum(px * w) / denom

    vw = m.groupby(["ticker", "date_tr"], sort=False).apply(_vwap)
    # if pandas returns a 1-col DF, coerce to Series
    if isinstance(vw, pd.DataFrame):
        vw = vw.iloc[:, 0]
    vw.name = "vwap"  # name the Series
    # IMPORTANT: do NOT reset_index() here
    # vwap = vw  # (we'll use 'vw' directly below)

    or_high = or_high.rename("or_high")
    or_low = or_low.rename("or_low")
    cl = lastbars["close_last"].rename("close_last")

    out = pd.concat([or_high, or_low, cl, vw], axis=1).reset_index()
    out["orb_confirm"] = (out["close_last"] > out["or_high"]).astype(int)
    out["vwap_support"] = (out["close_last"] >= out["vwap"]).astype(int)
    if not vwap_required:
        out["vwap_support"] = 1
    out["dt_tr_day"] = pd.to_datetime(out["date_tr"])

    # tag which bar size was inferred and a friendly interval label
    out["bar_minutes"] = bar_minutes
    lbl = {1: "1m", 5: "5m", 30: "30m", 60: "60m", 120: "2h", 240: "4h"}
    out["interval"] = lbl.get(bar_minutes, f"{bar_minutes}m")

    print(f"[bist100] Intraday bar size inferred: {bar_minutes} minutes")
    slow_bar = bar_minutes >= 240  # treat 4h+ bars as 'slow intraday'

    # --- make intraday confirmations robust when bar size is slow (≥240m)
    # 'flags' must already have: ['ticker','date_tr','close_last','orb_confirm','vwap_support']
    # and 'intra' must have 'vwap' at bar level (groupable to end-of-day VWAP)

    # --- make intraday confirmations robust when bar size is slow (≥240m)
    # Use 'out' (daily confirmations) and 'm' (intraday bars) built above.

    # Standard fill (works for 30/60/90m etc.)
    out["orb_confirm"] = out["orb_confirm"].fillna(0).astype(int)
    out["vwap_support"] = out["vwap_support"].fillna(0).astype(int)

    if slow_bar:
        # 4h bars don't really have a meaningful "opening-range breakout"
        out["orb_confirm"] = 1

        # End-of-day VWAP fallback computed from intraday bars 'm'
        def _vwap_grp(df):
            px = df["close"].astype(float)
            w = df["volume"].astype(float)
            denom = np.nansum(w)
            if not np.isfinite(denom) or denom <= 0:
                return np.nan
            return np.nansum(px * w) / denom

        day_vwap = (
            m.groupby(["ticker", "date_tr"]).apply(_vwap_grp).rename("day_vwap").reset_index()
        )

        out = out.merge(day_vwap, on=["ticker", "date_tr"], how="left")

        # If vwap_support already 1 keep it; otherwise use close_last >= day_vwap.
        out["vwap_support"] = (
            (out["vwap_support"] == 1)
            | (out["day_vwap"].notna() & (out["close_last"] >= out["day_vwap"]))
        ).astype(int)

        out.drop(columns=["day_vwap"], inplace=True)

    # return the 4 standard columns in a stable order
    out = out[["ticker", "date_tr", "orb_confirm", "vwap_support"]].copy()
    return out.sort_values(["ticker", "date_tr"], kind="mergesort").reset_index(drop=True)
