"""Phase 3: Backtest CLI — Run backtests from the command line.

Usage:
    python3 scripts/run_backtest.py [--start 2024-06-01] [--end 2025-01-01] [--step 96]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import SYMBOL
from app.services.backtester import run_backtest, print_backtest_report


def progress(step: int, total: int) -> None:
    """Print progress bar."""
    pct = (step + 1) / total * 100 if total > 0 else 0
    bar_len = 40
    filled = int(bar_len * (step + 1) / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:.0f}% ({step + 1}/{total} windows)", end="", flush=True)


async def main(start_date: str, end_date: str, symbol: str, step_size: int) -> None:
    print(f"\nPhase 3 Backtest: {symbol} | {start_date} -> {end_date} | step={step_size}")
    print("=" * 60)
    print()

    result = await run_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        step_size=step_size,
        progress_callback=progress,
    )

    print()  # newline after progress bar

    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    print_backtest_report(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingMamba Phase 3 Backtest")
    parser.add_argument("--start", default="2024-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default=SYMBOL, help="Trading pair")
    parser.add_argument("--step", type=int, default=96, help="Step size in M15 candles (96 = 1 day)")
    args = parser.parse_args()

    asyncio.run(main(args.start, args.end, args.symbol, args.step))
