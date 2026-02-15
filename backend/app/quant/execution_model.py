"""Quant Execution Model (Layer 4) — Entry timing optimization.

Weight: 10% of combined quant score.
Uses session timing, VWAP distance, micro-trend alignment, and entry method quality.
"""

from __future__ import annotations

from app.models import Candle, Direction, Session, EntryMethod, MSSGrade
from app.config import KILL_ZONES


def compute_execution_score(
    session: Session | None,
    candles: list[Candle],
    signal_direction: Direction,
    entry_method: EntryMethod | None = None,
    mss_grade: MSSGrade = MSSGrade.NONE,
    l2_depth: dict | None = None,
) -> tuple[float, dict]:
    """Compute execution model score (-100 to +100).

    Components:
    1. Session timing quality (weight 3)
    2. Spread/slippage estimation (weight 2)
    3. VWAP distance (weight 2)
    4. Micro-trend alignment (weight 2)
    5. Entry method quality (weight 1)

    Returns (score, component_details).
    """
    is_bull = (
        signal_direction == Direction.BULLISH
        if isinstance(signal_direction, Direction)
        else signal_direction == "bullish"
    )

    components = {}
    weighted_sum = 0.0
    total_weight = 0.0

    # 1. Session timing quality
    session_score = _score_session(session)
    components["session"] = {"score": round(session_score, 1), "weight": 3}
    weighted_sum += session_score * 3
    total_weight += 3

    # 2. Spread/slippage estimation
    spread_score = _score_spread(l2_depth, candles)
    components["spread"] = {"score": round(spread_score, 1), "weight": 2}
    weighted_sum += spread_score * 2
    total_weight += 2

    # 3. VWAP distance
    vwap_score = _score_vwap_distance(candles, is_bull)
    components["vwap"] = {"score": round(vwap_score, 1), "weight": 2}
    weighted_sum += vwap_score * 2
    total_weight += 2

    # 4. Micro-trend alignment (5-candle EMA on entry TF)
    micro_score = _score_micro_trend(candles, is_bull)
    components["micro_trend"] = {"score": round(micro_score, 1), "weight": 2}
    weighted_sum += micro_score * 2
    total_weight += 2

    # 5. Entry method quality
    entry_score = _score_entry_method(entry_method, mss_grade)
    components["entry_method"] = {"score": round(entry_score, 1), "weight": 1}
    weighted_sum += entry_score * 1
    total_weight += 1

    # Normalize
    if total_weight > 0:
        raw = (weighted_sum / total_weight) * 10
    else:
        raw = 0.0

    score = max(-100.0, min(100.0, raw))
    return score, components


def _score_session(session: Session | None) -> float:
    """Session timing: kill zones are prime, off-hours are poor."""
    if not session:
        return 0.0

    if session.is_kill_zone:
        return 7.0  # Kill zone = institutional activity

    name = session.name.lower()
    if name == "ny":
        return 5.0   # NY session = highest volume
    elif name == "london":
        return 4.0   # London = good volume
    elif name == "asian":
        return -3.0  # Asian = low vol, wider spreads
    elif name == "late":
        return -5.0  # Late session = thin liquidity

    return 0.0


def _score_spread(l2_depth: dict | None, candles: list[Candle]) -> float:
    """Spread/slippage estimation from L2 depth or ATR proxy."""
    if l2_depth:
        # Use actual bid-ask spread
        best_bid = l2_depth.get("best_bid", 0)
        best_ask = l2_depth.get("best_ask", 0)
        if best_bid > 0 and best_ask > 0:
            spread_pct = (best_ask - best_bid) / best_bid * 100
            if spread_pct < 0.01:
                return 5.0   # Very tight spread
            elif spread_pct < 0.05:
                return 2.0   # Normal
            elif spread_pct > 0.1:
                return -5.0  # Wide spread
            return 0.0

    # Fallback: ATR-based proxy (narrow range = tight conditions)
    if len(candles) >= 5:
        recent_ranges = [c.high - c.low for c in candles[-5:]]
        avg_range = sum(recent_ranges) / len(recent_ranges)
        price = candles[-1].close
        if price > 0:
            range_pct = avg_range / price * 100
            if range_pct < 0.3:
                return 3.0   # Tight ranging
            elif range_pct > 1.0:
                return -3.0  # Wide ranging = slippage risk
    return 0.0


def _score_vwap_distance(candles: list[Candle], is_bull: bool) -> float:
    """VWAP distance: entry near VWAP is more favorable.

    Compute session VWAP from intraday candles (approx using volume-weighted price).
    """
    if len(candles) < 10:
        return 0.0

    # Simple VWAP over recent candles
    recent = candles[-96:]  # ~24h of M15 candles
    total_vol = 0.0
    vwap_num = 0.0
    for c in recent:
        typical = (c.high + c.low + c.close) / 3
        vwap_num += typical * c.volume
        total_vol += c.volume

    if total_vol == 0:
        return 0.0

    vwap = vwap_num / total_vol
    current = candles[-1].close
    distance_pct = (current - vwap) / vwap * 100

    # Bullish: entry below VWAP = discount, above = expensive
    if is_bull:
        if distance_pct < -0.5:
            return 5.0   # Below VWAP = good entry
        elif distance_pct > 1.0:
            return -3.0  # Extended above VWAP
    else:
        if distance_pct > 0.5:
            return 5.0   # Above VWAP = good entry for shorts
        elif distance_pct < -1.0:
            return -3.0  # Extended below VWAP

    return 1.0  # Near VWAP = fair value


def _score_micro_trend(candles: list[Candle], is_bull: bool) -> float:
    """5-candle EMA direction alignment with signal.

    If the micro-trend (short-term momentum) aligns with the signal,
    entry timing is better.
    """
    if len(candles) < 6:
        return 0.0

    # Compute 5-period EMA of closes
    closes = [c.close for c in candles[-10:]]
    ema = _ema(closes, 5)
    if len(ema) < 2:
        return 0.0

    ema_slope = ema[-1] - ema[-2]

    if is_bull and ema_slope > 0:
        return 5.0   # Micro uptrend aligns with bullish signal
    elif not is_bull and ema_slope < 0:
        return 5.0   # Micro downtrend aligns with bearish signal
    elif is_bull and ema_slope < 0:
        return -3.0  # Counter-micro-trend entry
    elif not is_bull and ema_slope > 0:
        return -3.0

    return 0.0


def _score_entry_method(entry_method: EntryMethod | None, mss_grade: MSSGrade) -> float:
    """Entry method quality: A++ MSS > A+ MSS > standard MSS > SBC > pullback."""
    if entry_method is None:
        return 0.0

    # MSS quality matters most
    if mss_grade == MSSGrade.A_PLUS_PLUS:
        return 10.0  # BPR — highest quality
    if mss_grade == MSSGrade.A_PLUS:
        return 7.0   # iFVG — high quality
    if mss_grade == MSSGrade.STANDARD:
        return 4.0   # Standard MSS

    # Entry method type
    if entry_method == EntryMethod.MSS:
        return 3.0
    elif entry_method == EntryMethod.SBC:
        return 5.0   # SBC is high conviction
    elif entry_method == EntryMethod.SCOB:
        return 4.0
    elif entry_method == EntryMethod.PULLBACK_BREAK:
        return 2.0

    return 0.0


def _ema(data: list[float], period: int) -> list[float]:
    """Compute EMA (Exponential Moving Average)."""
    if not data or period <= 0:
        return []

    multiplier = 2 / (period + 1)
    result = [data[0]]

    for i in range(1, len(data)):
        val = (data[i] - result[-1]) * multiplier + result[-1]
        result.append(val)

    return result
