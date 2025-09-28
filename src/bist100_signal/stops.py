import pandas as pd


def apply_atr_stop(trades: pd.DataFrame, daily_df: pd.DataFrame, atr_k: float) -> pd.DataFrame:
    d = daily_df.sort_values(["ticker", "dt_tr"]).copy()
    d = d[["ticker", "dt_tr", "low", "close", "atr14"]]
    d["dt_tr"] = pd.to_datetime(d["dt_tr"])

    out = trades.copy()
    # deterministic order for stop evaluation
    out = out.sort_values(["entry_date", "ticker"], kind="mergesort").reset_index(drop=True)
    out["stopped"] = 0

    for i, r in out.iterrows():
        tkr = r["ticker"]
        entry_dt = pd.to_datetime(r["entry_date"])
        exit_dt = pd.to_datetime(r["exit_date"])
        entry_open = r["next_open"]

        w = d[(d["ticker"] == tkr) & (d["dt_tr"] > entry_dt) & (d["dt_tr"] <= exit_dt)].copy()
        if w.empty:
            continue

        atr_entry = d[(d["ticker"] == tkr) & (d["dt_tr"] == entry_dt)].get("atr14")
        if atr_entry is None or len(atr_entry) == 0 or pd.isna(atr_entry.values[0]):
            continue
        stop_level = entry_open - atr_k * float(atr_entry.values[0])

        hit = w[w["low"] <= stop_level]
        if not hit.empty:
            first_hit = hit.iloc[0]
            out.at[i, "exit_date"] = first_hit["dt_tr"]
            out.at[i, "exit_close"] = first_hit["close"]
            out.at[i, "ret"] = out.at[i, "exit_close"] / entry_open - 1.0
            out.at[i, "stopped"] = 1
    return out
