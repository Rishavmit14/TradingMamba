"""Phase 5: Background signal monitor.

Runs every 30 seconds to detect new M15 signals, deduplicate them,
create pending trades in the database, and send Telegram alerts.
"""

import asyncio
import hashlib
import logging
from typing import Optional

from app.config import (
    SYMBOL,
    DEMO_MIN_SIGNAL_GRADE,
    DEMO_PENDING_TIMEOUT_SECONDS,
)
from app.services.data_fetcher import fetch_all_timeframes
from app.core.engine import run_multi_tf_analysis
from app.services.database import (
    signal_exists,
    insert_signal,
    insert_trade,
    mark_signal_sent,
    timeout_pending_trades,
    get_pending_trades,
)

logger = logging.getLogger("tradingmamba.signal_monitor")

# Grade ordering for filtering
_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def _compute_signal_hash(signal) -> str:
    """Compute a dedup hash for a signal.

    Uses direction + rounded price (to nearest 100) + 15-minute timestamp bucket
    so the same signal in the same M15 candle window is only sent once.
    """
    direction = signal.direction.value
    price_bucket = round(signal.entry_price, -2)  # nearest $100
    time_bucket = signal.timestamp // 900_000  # 15 min in ms
    raw = f"{direction}_{price_bucket}_{time_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _compute_priority_score(signal_data: dict) -> float:
    """Compute a 0-100 priority score for ranking competing signals.

    Weights: Grade 35%, Confidence 25%, R:R 20%, Confluences 10%,
    Penalties: counter-trend -10, climax warning -5.
    """
    # Grade score (A=100, B=75, C=50, D=25)
    grade_scores = {"A": 100, "B": 75, "C": 50, "D": 25}
    grade_s = grade_scores.get(signal_data.get("grade", "D"), 25)

    # Confidence (already 0-100)
    conf_s = min(signal_data.get("confidence_score", 0), 100)

    # R:R score (1.0 → 25, 2.0 → 50, 3.0+ → 100)
    rr = signal_data.get("risk_reward_ratio", 1.0)
    rr_s = min(rr / 3.0 * 100, 100)

    # Confluence count (0 → 0, 3 → 60, 5+ → 100)
    n_conf = len(signal_data.get("confluences", []))
    conf_count_s = min(n_conf / 5.0 * 100, 100)

    score = (
        grade_s * 0.35
        + conf_s * 0.25
        + rr_s * 0.20
        + conf_count_s * 0.10
    )

    # Penalties
    if signal_data.get("is_counter_trend"):
        score -= 10
    if signal_data.get("climax_warning"):
        score -= 5

    return round(max(0, min(100, score)), 1)


def _serialize_signal(signal) -> dict:
    """Convert a TradingSignal dataclass to a dict for storage."""
    data = {
        "direction": signal.direction.value,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "risk_reward_ratio": round(signal.risk_reward_ratio, 2),
        "grade": signal.grade.value,
        "confidence_score": round(signal.confidence_score, 1),
        "confluences": signal.confluences,
        "entry_method": signal.entry_method.value if signal.entry_method else None,
        "pattern_type": signal.pattern_type,
        "timeframe": signal.timeframe,
        "is_counter_trend": signal.is_counter_trend,
        "climax_warning": signal.climax_warning,
    }
    data["priority_score"] = _compute_priority_score(data)
    return data


class SignalMonitor:
    """Background task that monitors for new signals every 30 seconds."""

    def __init__(self, bot=None, interval_seconds: int = 30):
        self.bot = bot
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None

    def start(self):
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self):
        """Main monitoring loop."""
        # Wait a bit on startup to let everything initialize
        await asyncio.sleep(5)

        while True:
            try:
                await self._scan_for_signals()
                await timeout_pending_trades(DEMO_PENDING_TIMEOUT_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal monitor error: {e}")

            await asyncio.sleep(self.interval)

    async def _scan_for_signals(self):
        """Run multi-TF analysis and process any new signals."""
        try:
            candles_by_tf = await fetch_all_timeframes(SYMBOL)
            results = run_multi_tf_analysis(candles_by_tf)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return

        m15 = results.get("M15")
        if not m15 or not m15.signals:
            return

        min_grade = _GRADE_ORDER.get(DEMO_MIN_SIGNAL_GRADE, 1)

        for signal in m15.signals:
            grade_rank = _GRADE_ORDER.get(signal.grade.value, 3)
            if grade_rank > min_grade:
                continue

            sig_hash = _compute_signal_hash(signal)

            # Skip if we've already processed this signal
            if await signal_exists(sig_hash):
                continue

            sig_data = _serialize_signal(signal)
            sig_data["signal_hash"] = sig_hash

            # Insert signal into DB
            signal_id = await insert_signal(sig_data)
            logger.info(
                f"New signal: {sig_data['direction']} {sig_data['grade']} "
                f"@ ${sig_data['entry_price']:,.0f} (ID: {signal_id})"
            )

            # Create pending trade
            trade_id = await insert_trade(signal_id, sig_data)

            # Send Telegram alert (if bot is available)
            if self.bot and self.bot.chat_id:
                try:
                    # Rank this signal among all pending trades
                    pending = await get_pending_trades()
                    scored = []
                    for p in pending:
                        ps = _compute_priority_score(p)
                        scored.append((ps, p["id"]))
                    scored.sort(key=lambda x: -x[0])
                    rank = next(
                        (i + 1 for i, (_, tid) in enumerate(scored) if tid == trade_id),
                        0,
                    )
                    msg_id = await self.bot.send_signal_alert(
                        sig_data, trade_id,
                        rank=rank, total_pending=len(scored),
                    )
                    await mark_signal_sent(signal_id, msg_id)
                    logger.info(f"Telegram alert sent for trade {trade_id} (rank {rank}/{len(scored)})")
                except Exception as e:
                    logger.error(f"Telegram alert failed: {e}")
