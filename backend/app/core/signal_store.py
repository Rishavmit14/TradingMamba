"""Signal Store — In-memory singleton that tracks signal lifecycle.

Handles:
- Cross-style deduplication (Day Trading + Scalping share M5 entry TF)
- Signal lifecycle: active → sl_hit / tp_hit / expired
- Outcome recording for performance analysis

Persists across API calls within the same server process. No DB changes.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from app.models import Candle, Direction, SignalGrade, TradingSignal


# Grade priority for selecting "best" signal when merging duplicates
_GRADE_PRIORITY = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass
class TrackedSignal:
    """A signal being tracked for its lifecycle."""
    signal_id: str                    # Hash of direction + rounded entry/SL (TP excluded for multi-TP merge)
    signal: TradingSignal             # Best-grade signal from merged group
    trading_styles: list[str]         # All styles that agree
    created_at: int                   # Unix ms when first generated
    status: str = "active"            # "active" | "sl_hit" | "tp_hit" | "expired"
    resolved_at: Optional[int] = None
    resolved_price: Optional[float] = None
    bars_active: int = 0
    entry_timeframe: str = ""         # TF of entry candles (for SL/TP checking)


class SignalStore:
    """Singleton in-memory store for signal lifecycle tracking."""

    _instance: Optional[SignalStore] = None
    EXPIRY_MS = 4 * 3600 * 1000  # 4 hours max
    MAX_RESOLVED = 200           # Ring buffer cap

    def __init__(self) -> None:
        self.active: dict[str, TrackedSignal] = {}
        self.resolved: list[TrackedSignal] = []

    @classmethod
    def get_instance(cls) -> SignalStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _compute_signal_id(sig: TradingSignal) -> str:
        """Hash of (direction, rounded entry, rounded SL).

        TP is EXCLUDED so signals from the same zone with different TPs merge
        into one signal with multiple TP levels (TP1, TP2, TP3).
        Entry (zone midpoint) and SL (zone boundary) define the zone identity.
        Rounding to nearest 10 ensures cross-style duplicates produce the same ID.
        """
        key = (
            sig.direction.value if isinstance(sig.direction, Direction) else sig.direction,
            round(sig.entry_price, -1),
            round(sig.stop_loss, -1),
        )
        return hashlib.md5(str(key).encode()).hexdigest()[:12]

    @staticmethod
    def _merge_take_profits(signals: list[TradingSignal], best: TradingSignal) -> list[dict]:
        """Merge take_profits from multiple signals into a deduplicated, sorted list."""
        entry = best.entry_price
        sl = best.stop_loss
        risk = abs(entry - sl)
        is_bull = (best.direction == Direction.BULLISH
                   if isinstance(best.direction, Direction)
                   else best.direction == "bullish")

        # Collect all TPs from all signals' take_profits lists + their primary TP
        # Filter out TPs that are on wrong side of entry or provide < 1R
        seen: set[float] = set()
        raw_tps: list[float] = []
        for sig in signals:
            # From take_profits list (already built by _deduplicate_signals)
            for tp_entry in (sig.take_profits or []):
                tp_price = tp_entry["price"]
                if risk > 0:
                    tp_rr = abs(tp_price - entry) / risk
                    if tp_rr < 1.0:
                        continue
                    if is_bull and tp_price <= entry:
                        continue
                    if not is_bull and tp_price >= entry:
                        continue
                tp_rounded = round(tp_price, -1)
                if tp_rounded not in seen:
                    seen.add(tp_rounded)
                    raw_tps.append(tp_price)
            # Also check the primary TP in case take_profits is empty
            tp_price = sig.take_profit
            if risk > 0:
                tp_rr = abs(tp_price - entry) / risk
                valid = tp_rr >= 1.0
                if is_bull and tp_price <= entry:
                    valid = False
                if not is_bull and tp_price >= entry:
                    valid = False
            else:
                valid = True
            if valid:
                tp_rounded = round(tp_price, -1)
                if tp_rounded not in seen:
                    seen.add(tp_rounded)
                    raw_tps.append(tp_price)

        # Sort: closest to entry first
        if is_bull:
            raw_tps.sort()
        else:
            raw_tps.sort(reverse=True)

        # Fallback: if all TPs got filtered, keep best signal's TP
        if not raw_tps:
            raw_tps = [best.take_profit]

        # Build labeled list
        result = []
        for i, tp in enumerate(raw_tps):
            rr = round(abs(tp - entry) / risk, 2) if risk > 0 else 0
            result.append({"price": tp, "rr": rr, "label": f"TP{i + 1}"})
        return result

    def _deduplicate_cross_style(
        self, signals: list[TradingSignal]
    ) -> dict[str, tuple[TradingSignal, list[str]]]:
        """Group signals by signal_id, merge trading_styles + take_profits, keep best grade.

        Returns dict mapping signal_id → (best_signal, merged_styles).
        """
        groups: dict[str, tuple[list[TradingSignal], list[str]]] = {}

        for sig in signals:
            sid = self._compute_signal_id(sig)
            style = sig.trading_style or "unknown"

            if sid not in groups:
                groups[sid] = ([sig], [style])
            else:
                sigs, styles = groups[sid]
                sigs.append(sig)
                if style not in styles:
                    styles.append(style)

        # For each group: pick best grade, merge take_profits
        result: dict[str, tuple[TradingSignal, list[str]]] = {}
        for sid, (sigs, styles) in groups.items():
            best = sigs[0]
            for sig in sigs[1:]:
                sig_grade = sig.grade.value if isinstance(sig.grade, SignalGrade) else sig.grade
                best_grade = best.grade.value if isinstance(best.grade, SignalGrade) else best.grade
                if _GRADE_PRIORITY.get(sig_grade, 9) < _GRADE_PRIORITY.get(best_grade, 9):
                    best = sig

            # Merge take_profits from all signals in this cross-style group
            merged_tps = self._merge_take_profits(sigs, best)
            best.take_profits = merged_tps
            if merged_tps:
                best.take_profit = merged_tps[0]["price"]
                best.risk_reward_ratio = merged_tps[0]["rr"]

            result[sid] = (best, styles)

        return result

    @staticmethod
    def _check_sl_tp_hit(
        tracked: TrackedSignal,
        candles: list[Candle],
    ) -> Optional[tuple[str, float]]:
        """Walk candles after signal trigger to check if SL or TP was hit.

        Returns ("sl_hit", price) or ("tp_hit", price) or None.
        Uses signal.timestamp (the zone-tap candle time, locked on first detection)
        instead of created_at (server wall clock) so SL/TP resolution survives
        server restarts.
        """
        direction = tracked.signal.direction
        is_bull = (direction == Direction.BULLISH) if isinstance(direction, Direction) else (direction == "bullish")
        sl = tracked.signal.stop_loss
        tp = tracked.signal.take_profit

        # Use the signal's trigger timestamp (zone tap time) to find the
        # starting point for SL/TP checking.  Falls back to created_at
        # for legacy signals that don't have a trigger timestamp.
        check_after = tracked.signal.timestamp or tracked.created_at

        for candle in candles:
            # Only check candles after the signal was triggered
            if candle.timestamp <= check_after:
                continue

            if is_bull:
                sl_hit = candle.low <= sl
                tp_hit = candle.high >= tp
            else:
                sl_hit = candle.high >= sl
                tp_hit = candle.low <= tp

            if sl_hit and tp_hit:
                # Both in same candle — use candle direction as heuristic
                if is_bull:
                    # Bull signal: if candle closed bearish, likely SL hit first
                    if candle.is_bearish:
                        return ("sl_hit", sl)
                    else:
                        return ("tp_hit", tp)
                else:
                    if candle.is_bullish:
                        return ("sl_hit", sl)
                    else:
                        return ("tp_hit", tp)
            elif sl_hit:
                return ("sl_hit", sl)
            elif tp_hit:
                return ("tp_hit", tp)

        return None

    def update(
        self,
        new_signals: list[TradingSignal],
        candles_by_tf: dict[str, list[Candle]],
    ) -> None:
        """Main entry point — called every analysis cycle.

        1. Cross-style dedup new signals
        2. Match against existing active signals (refresh if regenerated)
        3. Check SL/TP for all active signals
        4. Expire signals older than EXPIRY_MS
        5. Move resolved/expired to resolved ring buffer
        """
        now_ms = int(time.time() * 1000)

        # 1. Cross-style dedup
        deduped = self._deduplicate_cross_style(new_signals)

        # Track which active signals were regenerated this cycle
        regenerated_ids: set[str] = set()

        # 2. Match vs existing active signals
        for sid, (sig, styles) in deduped.items():
            if sid in self.active:
                # Signal regenerated — refresh but LOCK fields from first detection
                # so the signal identity stays stable across cycles.
                tracked = self.active[sid]
                locked_entry = tracked.signal.entry_price
                locked_rr = tracked.signal.risk_reward_ratio
                locked_tps = tracked.signal.take_profits
                locked_trigger = tracked.signal.trigger_candle_index
                locked_ts = tracked.signal.timestamp
                locked_style = tracked.signal.trading_style
                tracked.signal = sig
                # Restore locked fields
                tracked.signal.entry_price = locked_entry
                tracked.signal.risk_reward_ratio = locked_rr
                tracked.signal.take_profits = locked_tps
                tracked.signal.trigger_candle_index = locked_trigger
                tracked.signal.timestamp = locked_ts
                tracked.signal.trading_style = locked_style
                tracked.trading_styles = styles
                tracked.bars_active += 1
                tracked.entry_timeframe = sig.timeframe
                regenerated_ids.add(sid)
            else:
                # Before adding, check if an existing active signal overlaps
                # (same direction + similar SL). SBC signals drift entry with
                # current_price each cycle, producing different signal_ids for
                # the same zone. Detect this and replace the old signal.
                sig_dir = sig.direction.value if isinstance(sig.direction, Direction) else sig.direction
                replaced_old = None
                for old_sid, old_tracked in self.active.items():
                    if old_sid in regenerated_ids:
                        continue  # Already matched this cycle
                    old_sig = old_tracked.signal
                    old_dir = old_sig.direction.value if isinstance(old_sig.direction, Direction) else old_sig.direction
                    if old_dir != sig_dir:
                        continue
                    # Same direction — check if SL is within 0.5% (same zone)
                    sl_pct_diff = abs(old_sig.stop_loss - sig.stop_loss) / max(abs(sig.stop_loss), 1)
                    if sl_pct_diff < 0.005:
                        replaced_old = old_sid
                        break

                if replaced_old:
                    # Replace old signal with new one, preserving creation time
                    old_tracked = self.active.pop(replaced_old)
                    self.active[sid] = TrackedSignal(
                        signal_id=sid,
                        signal=sig,
                        trading_styles=styles,
                        created_at=old_tracked.created_at,
                        status="active",
                        bars_active=old_tracked.bars_active + 1,
                        entry_timeframe=sig.timeframe,
                    )
                else:
                    # Genuinely new signal
                    self.active[sid] = TrackedSignal(
                        signal_id=sid,
                        signal=sig,
                        trading_styles=styles,
                        created_at=now_ms,
                        status="active",
                        bars_active=0,
                        entry_timeframe=sig.timeframe,
                    )
                regenerated_ids.add(sid)

        # 3. Check SL/TP for ALL active signals (including regenerated ones)
        to_resolve: list[str] = []

        for sid, tracked in self.active.items():
            # Get candles for this signal's entry timeframe
            tf = tracked.entry_timeframe or "M15"
            candles = candles_by_tf.get(tf, [])

            if candles:
                result = self._check_sl_tp_hit(tracked, candles)
                if result:
                    status, price = result
                    tracked.status = status
                    tracked.resolved_at = now_ms
                    tracked.resolved_price = price
                    to_resolve.append(sid)
                    continue

            # 4. Expire signals older than EXPIRY_MS
            if not (sid in regenerated_ids) and (now_ms - tracked.created_at > self.EXPIRY_MS):
                tracked.status = "expired"
                tracked.resolved_at = now_ms
                to_resolve.append(sid)

        # 5. Move resolved to ring buffer
        for sid in to_resolve:
            tracked = self.active.pop(sid)
            self.resolved.append(tracked)

        # Trim ring buffer
        if len(self.resolved) > self.MAX_RESOLVED:
            self.resolved = self.resolved[-self.MAX_RESOLVED:]

    def get_active_as_trading_signals(self) -> list[TradingSignal]:
        """Return active signals as TradingSignal list with lifecycle fields populated."""
        result: list[TradingSignal] = []
        for tracked in self.active.values():
            sig = tracked.signal
            # Populate lifecycle fields
            sig.signal_id = tracked.signal_id
            sig.trading_styles = tracked.trading_styles
            sig.status = tracked.status
            sig.created_at = tracked.created_at
            sig.bars_active = tracked.bars_active
            result.append(sig)
        return result

    def get_resolved_signals(self, limit: int = 50) -> list[dict]:
        """Return recent resolved signals for performance display."""
        recent = self.resolved[-limit:] if limit < len(self.resolved) else self.resolved
        result = []
        for tracked in reversed(recent):
            sig = tracked.signal
            direction = sig.direction.value if isinstance(sig.direction, Direction) else sig.direction
            grade = sig.grade.value if isinstance(sig.grade, SignalGrade) else sig.grade
            entry_method = sig.entry_method.value if sig.entry_method else None
            result.append({
                "signal_id": tracked.signal_id,
                "direction": direction,
                "entry_price": sig.entry_price,
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
                "risk_reward_ratio": sig.risk_reward_ratio,
                "grade": grade,
                "trading_styles": tracked.trading_styles,
                "status": tracked.status,
                "created_at": tracked.created_at,
                "resolved_at": tracked.resolved_at,
                "resolved_price": tracked.resolved_price,
                "bars_active": tracked.bars_active,
                "confluences": sig.confluences,
                "entry_method": entry_method,
                "confidence_score": sig.confidence_score,
                "timeframe": sig.timeframe,
                "take_profits": sig.take_profits if sig.take_profits else [],
            })
        return result

    def get_stats(self) -> dict:
        """Return summary stats for the signal store."""
        sl_hits = sum(1 for t in self.resolved if t.status == "sl_hit")
        tp_hits = sum(1 for t in self.resolved if t.status == "tp_hit")
        expired = sum(1 for t in self.resolved if t.status == "expired")
        total = sl_hits + tp_hits + expired
        return {
            "active_count": len(self.active),
            "resolved_count": len(self.resolved),
            "sl_hits": sl_hits,
            "tp_hits": tp_hits,
            "expired": expired,
            "win_rate": round(tp_hits / total * 100, 1) if total > 0 else 0,
        }
