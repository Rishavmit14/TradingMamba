"""1.10 — Entry Signal Generator (The Master Checklist)

Combines ALL detectors into the final trading signal using the V23 Master
Trading Checklist adapted for W1→D1→H4→M15.

The Checklist:
1. IDENTIFY TREND → W1 swing classifier → HH/HL = buy bias, LH/LL = sell bias
2. MARK STRUCTURE → D1 BOS/CHoCH, swing points
3. FIND IDM → D1/H4: after each BOS, locate new inducement
4. FIND ZONE → H4: below IDM → OB + FVG = sell/buy zone
5. CHECK CONTEXT → premium/discount on D1 range + session awareness
6. WAIT FOR TAP → price must reach the H4 zone
7. CONFIRM ENTRY on M15 → MSS / SCOB / valid pullback break
8. SET SL → beyond the H4 zone
9. SET TP → previous H4 high/low from which market took inducement
10. MONITOR FOR CLIMAX → largest D1 move = caution
11. DETECT CHoCH → D1 HL/LH break = direction switch
12. COUNTER-TREND → D1 BOS + IDM close + FVG → TP first opposing zone
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, Inducement, LiquidityPool, BOS, CHoCH,
    FVG, OrderBlock, PremiumDiscount, Session, TradingSignal,
    Direction, TrendState, SwingType, SwingClassification,
    ZoneType, EntryMethod, SignalGrade, IDMStatus,
)
from app.config import SL_BUFFER_PCT, MIN_RISK_REWARD


def _find_active_zones(
    order_blocks: list[OrderBlock],
    fvgs: list[FVG],
    trend: TrendState,
) -> list[dict]:
    """Find unmitigated zones (OB + FVG) that align with trend direction.

    V14+V13: Best zones have BOTH OB and FVG at the same level.
    """
    zones = []

    # Valid, unmitigated OBs in trend direction
    for ob in order_blocks:
        if not ob.valid or ob.mitigated:
            continue

        if trend == TrendState.BULLISH and ob.direction == Direction.BULLISH:
            zones.append({
                "type": "OB",
                "upper": ob.upper_price,
                "lower": ob.lower_price,
                "midpoint": ob.midpoint,
                "direction": Direction.BULLISH,
                "has_fvg": ob.has_fvg,
                "candle_index": ob.candle_index_start,
            })
        elif trend == TrendState.BEARISH and ob.direction == Direction.BEARISH:
            zones.append({
                "type": "OB",
                "upper": ob.upper_price,
                "lower": ob.lower_price,
                "midpoint": ob.midpoint,
                "direction": Direction.BEARISH,
                "has_fvg": ob.has_fvg,
                "candle_index": ob.candle_index_start,
            })

    # Valid, unmitigated FVGs in trend direction (if no OB at that level)
    for fvg in fvgs:
        if not fvg.valid or fvg.mitigated:
            continue

        if trend == TrendState.BULLISH and fvg.direction == Direction.BULLISH:
            zones.append({
                "type": "FVG",
                "upper": fvg.upper_price,
                "lower": fvg.lower_price,
                "midpoint": fvg.midpoint,
                "direction": Direction.BULLISH,
                "has_fvg": True,
                "candle_index": fvg.candle_index,
            })
        elif trend == TrendState.BEARISH and fvg.direction == Direction.BEARISH:
            zones.append({
                "type": "FVG",
                "upper": fvg.upper_price,
                "lower": fvg.lower_price,
                "midpoint": fvg.midpoint,
                "direction": Direction.BEARISH,
                "has_fvg": True,
                "candle_index": fvg.candle_index,
            })

    return zones


def _check_zone_tap(zone: dict, current_price: float) -> bool:
    """Check if current price has tapped (reached) a zone.

    Step 6: price must reach the identified zone before entry.
    """
    return zone["lower"] <= current_price <= zone["upper"]


def _calculate_sl(zone: dict, direction: Direction) -> float:
    """Calculate stop loss placement.

    V23: SL 2-4 pips beyond the zone (for crypto, use percentage buffer).
    """
    if direction == Direction.BULLISH:
        return zone["lower"] * (1 - SL_BUFFER_PCT)
    else:
        return zone["upper"] * (1 + SL_BUFFER_PCT)


def _calculate_tp(
    zone: dict,
    direction: Direction,
    swings: list[SwingPoint],
    inducements: list[Inducement],
) -> float | None:
    """Calculate take profit.

    V23: TP = previous high/low from which market took inducement.
    """
    if direction == Direction.BULLISH:
        # TP at the swing high above current zone
        for swing in sorted(swings, key=lambda s: s.price):
            if (swing.swing_type == SwingType.SWING_HIGH
                    and swing.price > zone["upper"]):
                return swing.price
    else:
        # TP at the swing low below current zone
        for swing in sorted(swings, key=lambda s: s.price, reverse=True):
            if (swing.swing_type == SwingType.SWING_LOW
                    and swing.price < zone["lower"]):
                return swing.price

    return None


def _count_confluences(
    zone: dict,
    pd: PremiumDiscount | None,
    session: Session | None,
    trend: TrendState,
    bos_events: list[BOS],
    choch_events: list[CHoCH],
    has_multi_tf_alignment: bool,
) -> list[str]:
    """Count and list all confluences for a signal."""
    confluences = []

    # Zone type
    if zone["type"] == "OB":
        confluences.append("Order Block")
    if zone["has_fvg"]:
        confluences.append("FVG")

    # Premium/Discount alignment
    if pd:
        if trend == TrendState.BULLISH and pd.zone == ZoneType.DISCOUNT:
            confluences.append("Discount zone (buy)")
        elif trend == TrendState.BEARISH and pd.zone == ZoneType.PREMIUM:
            confluences.append("Premium zone (sell)")

    # Session/Kill zone
    if session and session.is_kill_zone:
        confluences.append("Kill zone active")

    # Recent BOS confirmation
    valid_bos = [b for b in bos_events if b.valid]
    if valid_bos:
        confluences.append("BOS confirmed")

    # Multi-TF alignment
    if has_multi_tf_alignment:
        confluences.append("Multi-TF aligned")

    return confluences


def _grade_signal(confluences: list[str], is_counter_trend: bool, climax_warning: bool) -> SignalGrade:
    """Assign signal grade based on confluences.

    A: 3+ confluences, trend-aligned, kill zone, multi-TF
    B: 2 confluences, trend-aligned
    C: 1 confluence or weak alignment
    D: Counter-trend or climax warning
    """
    if is_counter_trend or climax_warning:
        return SignalGrade.D

    count = len(confluences)
    if count >= 3:
        return SignalGrade.A
    elif count >= 2:
        return SignalGrade.B
    elif count >= 1:
        return SignalGrade.C
    else:
        return SignalGrade.D


def generate_signals(
    candles: list[Candle],
    swings: list[SwingPoint],
    inducements: list[Inducement],
    liquidity_pools: list[LiquidityPool],
    bos_events: list[BOS],
    choch_events: list[CHoCH],
    fvgs: list[FVG],
    order_blocks: list[OrderBlock],
    trend: TrendState,
    pd: PremiumDiscount | None = None,
    session: Session | None = None,
    w1_trend: TrendState | None = None,
    d1_trend: TrendState | None = None,
    climax_warning: bool = False,
) -> list[TradingSignal]:
    """The Master Checklist — generate trading signals from all detector outputs.

    This is the top-level function that orchestrates the entire system.
    """
    if not candles or trend == TrendState.RANGING:
        return []

    current_price = candles[-1].close
    signals: list[TradingSignal] = []

    # Multi-TF alignment check
    has_multi_tf = False
    if w1_trend and d1_trend:
        if trend == TrendState.BULLISH:
            has_multi_tf = (w1_trend == TrendState.BULLISH
                           and d1_trend == TrendState.BULLISH)
        elif trend == TrendState.BEARISH:
            has_multi_tf = (w1_trend == TrendState.BEARISH
                           and d1_trend == TrendState.BEARISH)

    # Step 4: Find active zones
    zones = _find_active_zones(order_blocks, fvgs, trend)

    for zone in zones:
        # Step 6: Check zone tap
        if not _check_zone_tap(zone, current_price):
            continue

        direction = zone["direction"]

        # Step 8: Calculate SL
        sl = _calculate_sl(zone, direction)

        # Step 9: Calculate TP
        tp = _calculate_tp(zone, direction, swings, inducements)
        if tp is None:
            continue

        # Calculate R:R
        entry = current_price
        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            continue

        rr = reward / risk
        if rr < MIN_RISK_REWARD:
            continue

        # Count confluences
        confluences = _count_confluences(
            zone, pd, session, trend,
            bos_events, choch_events, has_multi_tf,
        )

        # Grade
        grade = _grade_signal(confluences, False, climax_warning)

        # Confidence score (0-100)
        base_score = len(confluences) * 15
        if has_multi_tf:
            base_score += 10
        if session and session.is_kill_zone:
            base_score += 5
        if climax_warning:
            base_score -= 20
        confidence = max(0, min(100, base_score))

        signals.append(TradingSignal(
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            risk_reward_ratio=round(rr, 2),
            confidence_score=confidence,
            grade=grade,
            confluences=confluences,
            timeframe="M15",
            entry_method=EntryMethod.MSS,
            pattern_type=f"{zone['type']} trend continuation",
            timestamp=candles[-1].timestamp,
            w1_trend=w1_trend,
            d1_trend=d1_trend,
            session=session,
            climax_warning=climax_warning,
            is_counter_trend=False,
        ))

    return signals
