"""Phase 3: Backtest Engine — Sliding window signal generation + outcome evaluation.

Runs the full multi-TF detection pipeline at regular intervals across historical data,
collects signals, and tracks whether each signal's TP or SL was hit first.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from dataclasses import asdict, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from app.config import SYMBOL
from app.core.engine import run_multi_tf_analysis, AnalysisResult
from app.models import (
    Candle, Direction, SignalGrade, TradingSignal,
    TradeOutcome, TradeRecord,
)
from app.services.data_fetcher import fetch_or_cache_historical

BACKTEST_DIR = Path(__file__).resolve().parents[3] / "data" / "backtest"

# Minimum candles needed per TF before we start generating signals
MIN_CANDLES = {"W1": 10, "D1": 60, "H4": 200, "M15": 500}


def _find_tf_index_at_timestamp(candles: list[Candle], timestamp: int) -> int:
    """Find the last candle index with open time <= timestamp."""
    timestamps = [c.timestamp for c in candles]
    idx = bisect_right(timestamps, timestamp) - 1
    return max(0, idx)


def _evaluate_outcome(
    signal: TradingSignal,
    future_candles: list[Candle],
    max_bars: int = 960,
) -> dict:
    """Track what happens after a signal is generated.

    Scans future candles to determine if TP or SL was hit first.
    max_bars=960 = 10 days of M15 candles.

    Returns dict with outcome fields.
    """
    mfe = 0.0  # max favorable excursion (%)
    mae = 0.0  # max adverse excursion (%)
    is_bullish = signal.direction == Direction.BULLISH

    for i, candle in enumerate(future_candles[:max_bars]):
        if is_bullish:
            # TP check: high reaches take_profit
            if candle.high >= signal.take_profit:
                return {
                    "outcome": TradeOutcome.WIN,
                    "exit_price": signal.take_profit,
                    "exit_candle_idx": candle.index,
                    "exit_timestamp": candle.timestamp,
                    "bars_held": i + 1,
                    "pnl_pct": (signal.take_profit - signal.entry_price) / signal.entry_price * 100,
                    "mfe": mfe,
                    "mae": mae,
                }
            # SL check: low hits stop_loss
            if candle.low <= signal.stop_loss:
                return {
                    "outcome": TradeOutcome.LOSS,
                    "exit_price": signal.stop_loss,
                    "exit_candle_idx": candle.index,
                    "exit_timestamp": candle.timestamp,
                    "bars_held": i + 1,
                    "pnl_pct": (signal.stop_loss - signal.entry_price) / signal.entry_price * 100,
                    "mfe": mfe,
                    "mae": mae,
                }
            # Track excursions
            fav = (candle.high - signal.entry_price) / signal.entry_price * 100
            adv = (signal.entry_price - candle.low) / signal.entry_price * 100
        else:
            # Bearish: TP when low hits take_profit
            if candle.low <= signal.take_profit:
                return {
                    "outcome": TradeOutcome.WIN,
                    "exit_price": signal.take_profit,
                    "exit_candle_idx": candle.index,
                    "exit_timestamp": candle.timestamp,
                    "bars_held": i + 1,
                    "pnl_pct": (signal.entry_price - signal.take_profit) / signal.entry_price * 100,
                    "mfe": mfe,
                    "mae": mae,
                }
            # SL when high hits stop_loss
            if candle.high >= signal.stop_loss:
                return {
                    "outcome": TradeOutcome.LOSS,
                    "exit_price": signal.stop_loss,
                    "exit_candle_idx": candle.index,
                    "exit_timestamp": candle.timestamp,
                    "bars_held": i + 1,
                    "pnl_pct": (signal.entry_price - signal.stop_loss) / signal.entry_price * 100,
                    "mfe": mfe,
                    "mae": mae,
                }
            fav = (signal.entry_price - candle.low) / signal.entry_price * 100
            adv = (candle.high - signal.entry_price) / signal.entry_price * 100

        mfe = max(mfe, max(fav, 0))
        mae = max(mae, max(adv, 0))

    # Timeout — neither hit
    last = future_candles[min(max_bars - 1, len(future_candles) - 1)] if future_candles else None
    exit_price = last.close if last else signal.entry_price
    if is_bullish:
        pnl = (exit_price - signal.entry_price) / signal.entry_price * 100
    else:
        pnl = (signal.entry_price - exit_price) / signal.entry_price * 100

    return {
        "outcome": TradeOutcome.TIMEOUT,
        "exit_price": exit_price,
        "exit_candle_idx": last.index if last else 0,
        "exit_timestamp": last.timestamp if last else 0,
        "bars_held": min(max_bars, len(future_candles)),
        "pnl_pct": pnl,
        "mfe": mfe,
        "mae": mae,
    }


def _signal_to_trade(signal: TradingSignal, outcome: dict) -> TradeRecord:
    """Convert a TradingSignal + outcome dict into a TradeRecord."""
    return TradeRecord(
        direction=signal.direction,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        risk_reward_ratio=signal.risk_reward_ratio,
        grade=signal.grade,
        confidence_score=signal.confidence_score,
        confluences=list(signal.confluences),
        entry_method=signal.entry_method.value if signal.entry_method else "",
        pattern_type=signal.pattern_type,
        is_counter_trend=signal.is_counter_trend,
        vsa_absorption=signal.vsa_absorption,
        entry_candle_idx=0,
        entry_timestamp=signal.timestamp,
        outcome=outcome["outcome"],
        exit_price=outcome["exit_price"],
        exit_candle_idx=outcome["exit_candle_idx"],
        exit_timestamp=outcome["exit_timestamp"],
        bars_held=outcome["bars_held"],
        pnl_pct=round(outcome["pnl_pct"], 4),
        max_favorable_excursion=round(outcome["mfe"], 4),
        max_adverse_excursion=round(outcome["mae"], 4),
    )


def _make_zone_key(signal: TradingSignal) -> str:
    """Create a deduplication key for a signal's zone."""
    return f"{signal.direction.value}_{round(signal.entry_price, -1)}"


def _compute_statistics(trades: list[TradeRecord]) -> dict:
    """Compute overall and breakdown statistics from trade records."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "avg_rr": 0,
            "profit_factor": 0, "total_pnl_pct": 0, "max_drawdown_pct": 0,
            "by_grade": [], "by_confluence": [], "by_entry_method": {},
            "by_session": {}, "equity_curve": [],
        }

    wins = [t for t in trades if t.outcome == TradeOutcome.WIN]
    losses = [t for t in trades if t.outcome == TradeOutcome.LOSS]
    timeouts = [t for t in trades if t.outcome == TradeOutcome.TIMEOUT]

    total = len(trades)
    win_rate = len(wins) / total * 100 if total else 0

    # Average realized R:R
    avg_win_pnl = mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss_pnl = mean([abs(t.pnl_pct) for t in losses]) if losses else 1
    avg_rr = avg_win_pnl / avg_loss_pnl if avg_loss_pnl > 0 else 0

    # Profit factor
    gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = sum(abs(t.pnl_pct) for t in trades if t.pnl_pct < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total P&L and equity curve
    total_pnl = sum(t.pnl_pct for t in trades)
    equity = []
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.entry_timestamp):
        cumulative += t.pnl_pct
        equity.append({"timestamp": t.entry_timestamp, "pnl": round(cumulative, 2)})
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # By grade
    by_grade = []
    for grade in ["A", "B", "C", "D"]:
        gt = [t for t in trades if t.grade.value == grade]
        if not gt:
            by_grade.append({
                "grade": grade, "trades": 0, "wins": 0, "losses": 0,
                "timeouts": 0, "win_rate": 0, "avg_pnl": 0,
                "recommendation": "N/A",
            })
            continue
        gw = sum(1 for t in gt if t.outcome == TradeOutcome.WIN)
        gl = sum(1 for t in gt if t.outcome == TradeOutcome.LOSS)
        gto = sum(1 for t in gt if t.outcome == TradeOutcome.TIMEOUT)
        gwr = gw / len(gt) * 100
        gpnl = mean([t.pnl_pct for t in gt])
        rec = "TRADE" if gwr >= 55 and gpnl > 0 else "SKIP"
        by_grade.append({
            "grade": grade, "trades": len(gt), "wins": gw, "losses": gl,
            "timeouts": gto, "win_rate": round(gwr, 1), "avg_pnl": round(gpnl, 2),
            "recommendation": rec,
        })

    # By confluence (contrastive analysis)
    all_confluences = set()
    for t in trades:
        all_confluences.update(t.confluences)

    by_confluence = []
    for conf in sorted(all_confluences):
        present = [t for t in trades if conf in t.confluences]
        absent = [t for t in trades if conf not in t.confluences]
        p_wr = sum(1 for t in present if t.outcome == TradeOutcome.WIN) / len(present) * 100 if present else 0
        a_wr = sum(1 for t in absent if t.outcome == TradeOutcome.WIN) / len(absent) * 100 if absent else 0
        edge = p_wr - a_wr
        by_confluence.append({
            "name": conf,
            "present_wr": round(p_wr, 1),
            "absent_wr": round(a_wr, 1),
            "edge": round(edge, 1),
            "present_count": len(present),
            "absent_count": len(absent),
        })
    by_confluence.sort(key=lambda x: x["edge"], reverse=True)

    # By entry method
    by_entry_method = {}
    methods = set(t.entry_method for t in trades if t.entry_method)
    for method in sorted(methods):
        mt = [t for t in trades if t.entry_method == method]
        mw = sum(1 for t in mt if t.outcome == TradeOutcome.WIN)
        mwr = mw / len(mt) * 100 if mt else 0
        mpnl = mean([t.pnl_pct for t in mt]) if mt else 0
        by_entry_method[method] = {
            "trades": len(mt), "win_rate": round(mwr, 1), "avg_pnl": round(mpnl, 2),
        }

    # By session (kill zone vs off-hours)
    kz_trades = [t for t in trades if "Kill zone active" in t.confluences]
    off_trades = [t for t in trades if "Kill zone active" not in t.confluences]
    kz_wr = sum(1 for t in kz_trades if t.outcome == TradeOutcome.WIN) / len(kz_trades) * 100 if kz_trades else 0
    off_wr = sum(1 for t in off_trades if t.outcome == TradeOutcome.WIN) / len(off_trades) * 100 if off_trades else 0
    by_session = {
        "kill_zone": {"trades": len(kz_trades), "win_rate": round(kz_wr, 1)},
        "off_hours": {"trades": len(off_trades), "win_rate": round(off_wr, 1)},
    }

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "timeouts": len(timeouts),
        "win_rate": round(win_rate, 1),
        "avg_rr": round(avg_rr, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        "total_pnl_pct": round(total_pnl, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "by_grade": by_grade,
        "by_confluence": by_confluence,
        "by_entry_method": by_entry_method,
        "by_session": by_session,
        "equity_curve": equity,
    }


async def run_backtest(
    symbol: str = SYMBOL,
    start_date: str = "2024-06-01",
    end_date: str = "2025-01-01",
    step_size: int = 96,
    max_bars_timeout: int = 960,
    progress_callback=None,
) -> dict:
    """Run the full Phase 3 backtest pipeline.

    1. Fetch historical data for all 4 TFs
    2. Slide a window across M15 candles in steps
    3. At each step, run multi-TF analysis and collect signals
    4. Evaluate each signal's outcome using future candles
    5. Compute statistics and return results

    Args:
        symbol: Trading pair
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        step_size: M15 candles between analysis windows (96 = 1 day)
        max_bars_timeout: Max M15 candles to wait for SL/TP (960 = 10 days)
        progress_callback: Optional callable(step, total_steps) for progress updates

    Returns:
        Complete backtest results dict
    """
    # Step 1: Fetch historical data
    candles_by_tf: dict[str, list[Candle]] = {}
    for tf in ["W1", "D1", "H4", "M15"]:
        candles_by_tf[tf] = await fetch_or_cache_historical(symbol, tf, start_date, end_date)

    m15_candles = candles_by_tf["M15"]
    if len(m15_candles) < MIN_CANDLES["M15"]:
        return {"error": f"Not enough M15 candles: {len(m15_candles)} < {MIN_CANDLES['M15']}"}

    # Build timestamp arrays for TF alignment
    tf_timestamps = {tf: [c.timestamp for c in cs] for tf, cs in candles_by_tf.items()}

    # Step 2: Slide window
    total_steps = (len(m15_candles) - MIN_CANDLES["M15"]) // step_size
    all_signals: list[TradingSignal] = []
    all_trades: list[TradeRecord] = []
    active_zones: dict[str, int] = {}  # zone_key -> entry_candle_idx
    total_signals_raw = 0

    for step_num, window_end in enumerate(
        range(MIN_CANDLES["M15"], len(m15_candles), step_size)
    ):
        if progress_callback:
            progress_callback(step_num, total_steps)

        # Current timestamp boundary
        current_ts = m15_candles[window_end - 1].timestamp

        # Slice each TF up to current timestamp
        windowed: dict[str, list[Candle]] = {}
        for tf in ["W1", "D1", "H4", "M15"]:
            if tf == "M15":
                windowed[tf] = m15_candles[:window_end]
            else:
                idx = _find_tf_index_at_timestamp(candles_by_tf[tf], current_ts)
                windowed[tf] = candles_by_tf[tf][:idx + 1]

        # Check minimum candles
        skip = False
        for tf, min_c in MIN_CANDLES.items():
            if len(windowed.get(tf, [])) < min_c:
                skip = True
                break
        if skip:
            continue

        # Run multi-TF analysis
        results = run_multi_tf_analysis(windowed)

        # Collect M15 signals
        m15_result = results.get("M15")
        if not m15_result or not m15_result.signals:
            continue

        for signal in m15_result.signals:
            total_signals_raw += 1
            zone_key = _make_zone_key(signal)

            # Dedup: skip if same zone has active trade within step_size candles
            if zone_key in active_zones:
                prev_idx = active_zones[zone_key]
                if window_end - prev_idx < step_size * 2:
                    continue

            # Evaluate outcome using future candles
            future = m15_candles[window_end:]
            if not future:
                continue

            outcome = _evaluate_outcome(signal, future, max_bars_timeout)
            trade = _signal_to_trade(signal, outcome)
            trade.entry_candle_idx = window_end - 1

            all_trades.append(trade)
            active_zones[zone_key] = window_end

            # Clear zone after trade resolves
            if outcome["outcome"] != TradeOutcome.TIMEOUT:
                active_zones.pop(zone_key, None)

    # Step 3: Compute statistics
    stats = _compute_statistics(all_trades)

    # Step 4: Build result
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "total_signals": total_signals_raw,
        "total_candles": {tf: len(cs) for tf, cs in candles_by_tf.items()},
        "step_size": step_size,
        "timestamp": timestamp,
        **stats,
        "trades": [
            {
                "direction": t.direction.value,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "risk_reward_ratio": round(t.risk_reward_ratio, 2),
                "grade": t.grade.value,
                "confidence_score": t.confidence_score,
                "confluences": t.confluences,
                "entry_method": t.entry_method,
                "pattern_type": t.pattern_type,
                "is_counter_trend": t.is_counter_trend,
                "vsa_absorption": t.vsa_absorption,
                "entry_candle_idx": t.entry_candle_idx,
                "entry_timestamp": t.entry_timestamp,
                "outcome": t.outcome.value,
                "exit_price": t.exit_price,
                "exit_candle_idx": t.exit_candle_idx,
                "exit_timestamp": t.exit_timestamp,
                "bars_held": t.bars_held,
                "pnl_pct": t.pnl_pct,
                "max_favorable_excursion": t.max_favorable_excursion,
                "max_adverse_excursion": t.max_adverse_excursion,
            }
            for t in all_trades
        ],
    }

    # Step 5: Save to disk
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    report_path = BACKTEST_DIR / f"backtest_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def get_latest_backtest() -> dict | None:
    """Load the most recent backtest result from disk."""
    if not BACKTEST_DIR.exists():
        return None
    files = sorted(BACKTEST_DIR.glob("backtest_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0], "r") as f:
        return json.load(f)


def print_backtest_report(result: dict) -> None:
    """Print human-readable backtest report to console."""
    print()
    print("=" * 60)
    print("  TradingMamba Phase 3 — Backtest Report")
    print(f"  {result['symbol']} | {result['start_date']} -> {result['end_date']}")
    print("=" * 60)
    print()

    print("OVERALL PERFORMANCE:")
    print(f"  Total signals: {result['total_signals']}  |  Total trades: {result['total_trades']}")
    print(f"  Win rate:    {result['win_rate']}%  ({result.get('wins', 0)} wins, {result.get('losses', 0)} losses, {result.get('timeouts', 0)} timeouts)")
    print(f"  Avg R:R:     {result['avg_rr']}")
    print(f"  Profit factor: {result['profit_factor']}")
    print(f"  Total P&L:  {result['total_pnl_pct']:+.2f}%")
    print(f"  Max drawdown: -{result['max_drawdown_pct']:.2f}%")
    print()

    print("BY GRADE:")
    print(f"  {'Grade':<8} {'Trades':>8} {'Win%':>8} {'Avg PnL':>10} {'Action':>12}")
    print(f"  {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 12}")
    for g in result.get("by_grade", []):
        print(f"  {g['grade']:<8} {g['trades']:>8} {g['win_rate']:>7.1f}% {g['avg_pnl']:>+9.2f}% {g['recommendation']:>12}")
    print()

    print("CONFLUENCE ANALYSIS:")
    print(f"  {'Confluence':<25} {'Present':>10} {'Absent':>10} {'Edge':>8}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 10} {'─' * 8}")
    for c in result.get("by_confluence", [])[:10]:
        print(f"  {c['name']:<25} {c['present_wr']:>9.1f}% {c['absent_wr']:>9.1f}% {c['edge']:>+7.1f}%")
    print()

    print("ENTRY METHOD:")
    for method, data in result.get("by_entry_method", {}).items():
        print(f"  {method:<20} {data['win_rate']:.1f}% WR ({data['trades']} trades)")
    print()

    print("SESSION:")
    by_session = result.get("by_session", {})
    kz = by_session.get("kill_zone", {})
    off = by_session.get("off_hours", {})
    print(f"  Kill Zone:  {kz.get('win_rate', 0):.1f}% WR ({kz.get('trades', 0)} trades)")
    print(f"  Off-Hours:  {off.get('win_rate', 0):.1f}% WR ({off.get('trades', 0)} trades)")
    print()

    # Recommendation
    ab_trades = [g for g in result.get("by_grade", []) if g["grade"] in ("A", "B")]
    if ab_trades:
        ab_total = sum(g["trades"] for g in ab_trades)
        ab_wins = sum(g["wins"] for g in ab_trades)
        ab_wr = ab_wins / ab_total * 100 if ab_total else 0
        print(f"RECOMMENDED: Trade Grade A + B only ({ab_total} trades, {ab_wr:.1f}% WR)")
    print()
