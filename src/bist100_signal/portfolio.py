from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    initial_cash: float = 100_000.0
    max_names: int = 30  # mirrors exposure_cap
    slip_frac: float = 0.0  # extra slippage (fractional)
    commission_rate: float = 0.0010  # per side
    bsmv_rate: float = 0.05  # on commission
    exchange_fee_rate: float = 0.00001
    alloc_mode: str = "fixed_pct"  # NEW: use fixed 10% sizing by default
    alloc_pct: float = 0.075  # NEW: 10% of (cash + equity) per BUY


def _roundtrip_cost_frac(c: PortfolioConfig) -> float:
    return 2.0 * (c.commission_rate * (1.0 + c.bsmv_rate) + c.exchange_fee_rate)


def build_virtual_portfolio(
    trades: pd.DataFrame, daily: pd.DataFrame, cfg: PortfolioConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inputs:
      trades: must include ['ticker','entry_date','exit_date','entry_open','exit_open'] and 'ret_after_cost' (when closed)
      daily:  must include ['ticker','dt_tr','open','close']
    Returns:
      executions: rows for buys/sells (dt, ticker, qty, px, cash_after)
      ledger:     daily equity with cash, positions value, equity
    """
    # stable inputs
    trades = trades.sort_values(["entry_date", "ticker"], kind="mergesort").reset_index(drop=True)
    daily = daily.sort_values(["ticker", "dt_tr"], kind="mergesort").reset_index(drop=True)

    # normalize inputs
    t = trades.copy()
    t = t.sort_values("entry_date").reset_index(drop=True)
    px = daily[["ticker", "dt_tr", "open", "close"]].copy()
    px["dt_tr"] = pd.to_datetime(px["dt_tr"])
    px = px.sort_values(["dt_tr", "ticker"], kind="mergesort")

    # collapse any duplicate (dt_tr, ticker) rows deterministically
    px = (
        px.sort_values(["dt_tr", "ticker"], kind="mergesort")
        .groupby(["dt_tr", "ticker"], as_index=False)
        .agg(open=("open", "last"), close=("close", "last"))
    )

    # ---- NEW: forward-filled close (and open) matrices for robust MTM ----
    # Pivot to wide, sort, and forward-fill so missing closes reuse the latest known price.
    close_wide = (
        px.pivot_table(index="dt_tr", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )

    open_wide = (
        px.pivot_table(index="dt_tr", columns="ticker", values="open", aggfunc="last")
        .sort_index()
        .ffill()
    )

    def _latest_close(day: pd.Timestamp, tk: str) -> float:
        """Last available close up to 'day'. Falls back to latest open if close is missing."""
        try:
            val = close_wide.loc[day, tk]
            if pd.isna(val):
                # fall back to open (also ffilled) if for some reason close is NaN
                val = open_wide.loc[day, tk]
            return float(val) if pd.notna(val) else np.nan
        except KeyError:
            # day or ticker not present in table
            return np.nan

    # trading calendar
    cal = px["dt_tr"].sort_values().drop_duplicates().tolist()
    # map price lookup
    # map price lookup (unique by construction)
    px_open = px.groupby(["dt_tr", "ticker"])["open"].last()
    px_close = px.groupby(["dt_tr", "ticker"])["close"].last()

    cash = cfg.initial_cash
    positions = {}  # ticker -> qty
    executions = []
    equities = []

    # helper: active target count per day
    df_entries = t[["ticker", "entry_date"]].dropna().copy()
    df_entries["entry_effect"] = pd.to_datetime(df_entries["entry_date"]) + pd.Timedelta(days=1)
    df_exits = t[["ticker", "exit_date"]].dropna().copy()
    df_exits["exit_effect"] = pd.to_datetime(df_exits["exit_date"]) + pd.Timedelta(days=1)

    # compute desired equal weight per day
    # naive: target_names = min(max_names, number of open signals that day)
    # size on the morning of entry_effect
    open_set = set()
    for d in cal:
        # handle exits first (at next day open logic)
        exit_today = set(df_exits.loc[df_exits["exit_effect"] == d, "ticker"])
        for tk in sorted(exit_today):
            if tk in open_set:
                # sell at open
                px0 = px_open.get((d, tk), np.nan)
                if np.isnan(px0) or tk not in positions:
                    continue
                qty = positions.pop(tk)
                fill = px0 * (1.0 - cfg.slip_frac)
                proceed = qty * fill
                # costs per side on sell
                cost = proceed * (
                    cfg.commission_rate * (1.0 + cfg.bsmv_rate) + cfg.exchange_fee_rate
                )
                cash += proceed - cost
                executions.append(
                    {
                        "dt_tr": d,
                        "ticker": tk,
                        "side": "SELL",
                        "qty": qty,
                        "price": fill,
                        "cash_after": cash,
                    }
                )
                open_set.remove(tk)

        # handle new entries (at open)
        enter_today = df_entries.loc[df_entries["entry_effect"] == d, "ticker"].tolist()
        for tk in enter_today:
            open_set.add(tk)

        # compute target_per only when having actually names to size
        if len(open_set) > 0:
            target_names = min(cfg.max_names, len(open_set))
            if target_names > 0:

                eq = cash + sum(
                    positions[tk] * px_close.get((d, tk), np.nan) for tk in sorted(positions)
                )

                if not np.isfinite(eq):
                    eq = cash
                target_per = eq / float(target_names)
            else:
                target_per = np.nan
        else:
            target_per = np.nan

        # open new names not yet in positions (buy at open)
        for tk in sorted(open_set):
            if tk in positions:
                continue
            if len(positions) >= cfg.max_names:
                break

            px0 = px_open.get((d, tk), np.nan)
            if not np.isfinite(px0) or px0 <= 0:
                continue
            # --- sizing ---
            if cfg.alloc_mode == "fixed_pct":
                # Equity now = cash + MTM value of open positions (at today's close if available)

                eq_now = cash + sum(
                    positions[t] * px_close.get((d, t), np.nan) for t in sorted(positions)
                )

                if not np.isfinite(eq_now):
                    eq_now = cash
                alloc_cash = max(0.0, cfg.alloc_pct * eq_now)
                qty = int(np.floor(alloc_cash / px0))
            else:
                # fallback: equal-weight per active names (existing behavior)
                if not np.isfinite(target_per) or target_per <= 0:
                    continue
                qty = int(np.floor(target_per / px0))

            if qty <= 0:
                continue
            # --- end sizing ---

            fill = px0 * (1.0 + cfg.slip_frac)
            cost = (qty * fill) * (
                cfg.commission_rate * (1.0 + cfg.bsmv_rate) + cfg.exchange_fee_rate
            )
            spend = qty * fill + cost
            if spend <= cash:
                cash -= spend
                positions[tk] = qty
                executions.append(
                    {
                        "dt_tr": d,
                        "ticker": tk,
                        "side": "BUY",
                        "qty": qty,
                        "price": fill,
                        "cash_after": cash,
                    }
                )

        # daily marking to market at close (treat missing close as 0 contribution)
        # ---- NEW: MTM using forward-filled latest close (fallback to open) ----
        pos_val = 0.0

        for tk in sorted(positions):
            q = positions[tk]
            pxc = _latest_close(d, tk)

            if np.isfinite(pxc):
                pos_val += q * pxc

        equity = cash + pos_val
        equities.append(
            {
                "dt_tr": d,
                "cash": cash,
                "pos_val": pos_val,
                "equity": equity,
                "names": len(positions),
            }
        )

    executions_df = pd.DataFrame(executions)
    ledger_df = pd.DataFrame(equities)
    return executions_df, ledger_df
