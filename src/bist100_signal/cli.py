import argparse

import pandas as pd

from .config import DB_PATH
from .reporting import compare_runs
from .runner import run_pipeline


def main():
    p = argparse.ArgumentParser(prog="bist100", description="BIST100 signal prototype CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run strategy pipeline and save outputs")
    r.add_argument(
        "--db", default=DB_PATH, help="SQLite DB path (default: env DB_PATH or bist100_prices.db)"
    )

    rep = sub.add_parser("report", help="Show last runs from runs_summary")
    rep.add_argument("--limit", type=int, default=20, help="Number of rows to display")

    args = p.parse_args()

    if args.cmd == "run":
        # optional: DB_PATH override.
        out = run_pipeline()

        if not isinstance(out, dict):
            print("Warning: pipeline returned no result.")
            return

        print("== Portfolio metrics ==")
        print(out.get("metrics", {}))

        print("== Trade stats ==")
        print(out.get("trade_stats", {}))

        print("Files:", *out["files"])

    elif args.cmd == "report":
        df = compare_runs(DB_PATH, limit=args.limit)
        pd.set_option("display.max_columns", None)
        print(df.head(args.limit).to_string(index=False))
