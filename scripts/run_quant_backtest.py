"""Quant Algo Bias Backtest CLI — Replay alert threshold crossings over historical data.

Tests 7/10 institutional algorithms (Options, Liquidation, OFI excluded — real-time only)
against Binance historical data. Evaluates price outcomes at T+1h, T+4h, T+24h.

Usage:
    python3 scripts/run_quant_backtest.py --start 2025-11-01 --end 2026-02-01
    python3 scripts/run_quant_backtest.py --start 2025-08-01 --end 2026-02-01 --step 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.quant.quant_backtester import run_quant_backtest, print_backtest_report


def progress(step_or_msg, total=None):
    """Print progress bar or status message."""
    if isinstance(step_or_msg, str):
        print(f"\r  {step_or_msg:<60}", end="", flush=True)
        return

    step = step_or_msg
    if total and total > 0:
        pct = (step + 1) / total * 100
        bar_len = 40
        filled = int(bar_len * (step + 1) / total)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:.0f}% ({step + 1}/{total} hours)", end="", flush=True)


async def main():
    parser = argparse.ArgumentParser(description="Run Quant Algo Bias Backtest")
    parser.add_argument("--start", default="2025-11-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--step", type=int, default=1, help="Step size in hours (default: 1)")
    parser.add_argument("--cooldown", type=int, default=4, help="Alert cooldown hours (default: 4)")
    args = parser.parse_args()

    print("\n  Quant Algo Bias Backtest")
    print(f"  {args.symbol} | {args.start} → {args.end} | step={args.step}h | cooldown={args.cooldown}h")
    print()

    result = await run_quant_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        step_hours=args.step,
        cooldown_hours=args.cooldown,
        progress_callback=progress,
    )

    print()  # newline after progress bar
    print_backtest_report(result)


if __name__ == "__main__":
    asyncio.run(main())
