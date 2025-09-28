import os

DB_PATH = os.getenv("DB_PATH", os.path.join("input", "bist100_prices.db"))

# --- ORB controls ---
# Primary: minutes (interval-agnostic). Fallback: bars.
ORB_MINUTES = 90  # 90 min opening range
ORB_WINDOW_BARS_FALLBACK = 3  # used only if bar size inference fails

CFG = {
    "prox_pct": 0.02,
    "vol_mult": 1.2,
    "orb_window_bars": 3,
    "vwap_strict": True,  # optional passthrough,
    "require_intraday": False,  ##If True, until 2025-04-15
    "exposure_cap": 30,
    "rank_key": "rank_prox_vol",
    "atr_k": 2.0,
    "hold_days": 3,
    "commission_rate": 0.0010,  # 0.10% per side (edit to broker’s actual)
    "bsmv_rate": 0.05,  # 5% of the commission (legal BSMV on fees)
    "exchange_fee_rate": 0.00001,  # 0.001% per side (BIST fee proxy; tiny)
    "min_adv_try": 5_000_000,  # e.g., 5M TRY/day; tune later
    "min_price": 5.0,  # optional price floor to avoid pennies
    "recent_intraday_days": 370,  # last ~12 months require proper intraday
    "min_intraday_bar": 240,  # minutes → 240 = 4h minimum for recent window
    "alloc_mode": "fixed_pct",  # new
    "alloc_pct": 0.075,  # new -> 10% of (cash + equity) per BUY
}
