import pandas as pd

from .config import CFG


def build_signal_table(daily_ind: pd.DataFrame, intraday_flags: pd.DataFrame) -> pd.DataFrame:
    d = daily_ind.copy()
    d["date_tr"] = d["dt_tr"].dt.date

    f = intraday_flags.copy()
    # guarantee the date key exists
    if "date_tr" not in f.columns:
        if "dt_tr" in f.columns:
            f["date_tr"] = pd.to_datetime(f["dt_tr"]).dt.date
        elif "dt_tr_day" in f.columns:
            f["date_tr"] = pd.to_datetime(f["dt_tr_day"]).dt.date
    needed = ["ticker", "date_tr", "orb_confirm", "vwap_support"]
    f = f[[c for c in needed if c in f.columns]].copy()

    merged = pd.merge(d, f, how="left", on=["ticker", "date_tr"])
    # mark which rows had an intraday record at all (row exists in `f`)
    merged["_had_intra"] = merged["orb_confirm"].notna() | merged["vwap_support"].notna()

    # Strict when intraday exists; fall back to daily when the intra row is missing
    if CFG["require_intraday"]:
        # rows that actually have an intraday record → keep strict filling
        has_intra = merged["_had_intra"]
        merged.loc[has_intra, "orb_confirm"] = merged.loc[has_intra, "orb_confirm"].fillna(0)
        merged.loc[has_intra, "vwap_support"] = merged.loc[has_intra, "vwap_support"].fillna(0)

        # rows with NO intraday row at all → allow daily fallback (treat as pass)
        no_intra = ~has_intra
        merged.loc[no_intra, ["orb_confirm", "vwap_support"]] = merged.loc[
            no_intra, ["orb_confirm", "vwap_support"]
        ].fillna(1)
    else:
        merged["orb_confirm"] = merged["orb_confirm"].fillna(1)
        merged["vwap_support"] = merged["vwap_support"].fillna(1)

    merged[["orb_confirm", "vwap_support"]] = merged[["orb_confirm", "vwap_support"]].astype(int)

    if not CFG["vwap_strict"]:
        merged["vwap_support"] = 1

    merged["signal"] = (
        (merged["near_52w_high"] == 1)
        & (merged["ma200_uptrend"] == 1)
        & (merged["vol_surge"] == 1)
        & (merged["orb_confirm"] == 1)
        & (merged["vwap_support"] == 1)
    ).astype(int)

    merged["rank_prox_vol"] = (-merged["prox_to_52w"]).rank(
        method="first", ascending=True
    ) * 0.7 + (-merged["vol_ratio"]).rank(method="first", ascending=False) * 0.3
    return merged.sort_values(["dt_tr", "ticker"], kind="mergesort").reset_index(drop=True)


def select_top_signals_per_day(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Select top-N per day with a stable, deterministic tie-breaker:
    1) CFG['rank_key'] ascending
    2) prox_to_52w ascending (if present)
    3) ticker alphabetical
    """
    d = signals.copy()
    key = CFG["rank_key"]
    nmax = int(CFG["exposure_cap"])

    sort_cols = [c for c in [key, "prox_to_52w", "ticker"] if c in d.columns]
    d = d.sort_values(sort_cols, ascending=[True] * len(sort_cols), kind="mergesort")

    d["selected"] = 0
    for day, g in d[d["signal"] == 1].groupby("date_tr", sort=False):
        keep = g.head(nmax).index
        d.loc[keep, "selected"] = 1

    # canonical row order for downstream joins
    return d.sort_values(["date_tr", "ticker"], kind="mergesort").reset_index(drop=True)
