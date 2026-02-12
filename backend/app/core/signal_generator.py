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

        # V12: Fibonacci qualification
        if pd.is_fib_qualified:
            confluences.append(f"Fib {pd.closest_fib} level")

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


def _determine_entry_method(
    zone: dict,
    bos_events: list[BOS],
    choch_events: list[CHoCH],
) -> EntryMethod:
    """V15/V17: Classify entry as MSS, SBC, or PULLBACK_BREAK.

    MSS = confirmed CHoCH with body close (swing-based model V10)
    SBC = confirmed CHoCH with wick only (sweep-based model V10)
    PULLBACK_BREAK = valid BOS trend continuation (no recent CHoCH)
    """
    zone_idx = zone.get("candle_index", 0)

    # Check for confirmed CHoCH near this zone
    confirmed_chochs = [
        c for c in choch_events
        if c.confirmed and not c.is_fake and c.candle_index <= zone_idx
    ]

    if confirmed_chochs:
        latest = max(confirmed_chochs, key=lambda c: c.candle_index)
        if latest.model == "sweep":
            return EntryMethod.SBC
        return EntryMethod.MSS

    # No confirmed CHoCH → check for valid BOS (trend continuation)
    valid_bos = [b for b in bos_events if b.valid and b.candle_index <= zone_idx]
    if valid_bos:
        return EntryMethod.PULLBACK_BREAK

    return EntryMethod.MSS  # default


def _check_counter_trend_conditions(
    candles: list[Candle],
    inducements: list[Inducement],
    fvgs: list[FVG],
    trend: TrendState,
) -> Direction | None:
    """V21: Check if counter-trend 3-condition entry is valid.

    Rule 1: Major trend candle body closes beyond first valid IDM
    Rule 2: An FVG must exist in the swing where IDM was closed
    Rule 3: (LTF confirmation — handled by entry method check)

    Returns counter-trend direction if conditions met, None otherwise.
    """
    if not inducements or not candles:
        return None

    last_candle = candles[-1]

    # Find the most recent taken IDM with body close
    body_closed_idms = [
        idm for idm in inducements
        if idm.status == IDMStatus.TAKEN and idm.body_closed
    ]
    if not body_closed_idms:
        return None

    latest_idm = max(body_closed_idms, key=lambda i: i.taken_at_candle or 0)

    # Check if an FVG exists near/after the IDM candle
    idm_fvg_exists = any(
        f.valid and not f.mitigated
        and abs(f.candle_index - (latest_idm.taken_at_candle or 0)) <= 5
        for f in fvgs
    )

    if not idm_fvg_exists:
        return None

    # Counter-trend direction is opposite to current trend
    if trend == TrendState.BULLISH:
        return Direction.BEARISH
    elif trend == TrendState.BEARISH:
        return Direction.BULLISH
    return None


def _get_counter_trend_tp(
    direction: Direction,
    zone: dict,
    order_blocks: list[OrderBlock],
    fvgs: list[FVG],
) -> float | None:
    """V21: TP at FIRST opposing zone only — never hold deeper.

    For counter-trend buy: TP = first sell zone (bearish OB/FVG) above entry
    For counter-trend sell: TP = first buy zone (bullish OB/FVG) below entry
    """
    opposing_dir = Direction.BEARISH if direction == Direction.BULLISH else Direction.BULLISH

    targets: list[float] = []

    for ob in order_blocks:
        if not ob.valid or ob.mitigated or ob.direction != opposing_dir:
            continue
        if direction == Direction.BULLISH and ob.midpoint > zone["upper"]:
            targets.append(ob.midpoint)
        elif direction == Direction.BEARISH and ob.midpoint < zone["lower"]:
            targets.append(ob.midpoint)

    for fvg in fvgs:
        if not fvg.valid or fvg.mitigated or fvg.direction != opposing_dir:
            continue
        if direction == Direction.BULLISH and fvg.midpoint > zone["upper"]:
            targets.append(fvg.midpoint)
        elif direction == Direction.BEARISH and fvg.midpoint < zone["lower"]:
            targets.append(fvg.midpoint)

    if not targets:
        return None

    # First opposing zone = closest target
    if direction == Direction.BULLISH:
        return min(targets)  # Closest above
    else:
        return max(targets)  # Closest below


def _deduplicate_signals(signals: list[TradingSignal]) -> list[TradingSignal]:
    """V23: One trade per zone — keep highest grade signal per overlapping zone.

    Two zones overlap when their price ranges intersect.
    Priority: Grade A > B > C > D, then trend-aligned > counter-trend, then R:R.
    """
    if len(signals) <= 1:
        return signals

    _grade_rank = {"A": 4, "B": 3, "C": 2, "D": 1}

    def _sort_key(sig: TradingSignal):
        return (
            _grade_rank.get(sig.grade.value, 0),
            0 if sig.is_counter_trend else 1,
            sig.risk_reward_ratio,
            sig.confidence_score,
        )

    # Group overlapping zones
    groups: list[list[TradingSignal]] = []
    for sig in signals:
        sig_lower = min(sig.entry_price, sig.stop_loss)
        sig_upper = max(sig.entry_price, sig.take_profit)
        added = False
        for group in groups:
            ref = group[0]
            ref_lower = min(ref.entry_price, ref.stop_loss)
            ref_upper = max(ref.entry_price, ref.take_profit)
            if sig_lower <= ref_upper and sig_upper >= ref_lower:
                group.append(sig)
                added = True
                break
        if not added:
            groups.append([sig])

    return [max(group, key=_sort_key) for group in groups]


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
    htf_zones: list[dict] | None = None,
) -> list[TradingSignal]:
    """The Master Checklist — generate trading signals from all detector outputs.

    This is the top-level function that orchestrates the entire system.
    V20: htf_zones passed from higher TF for zone alignment.
    """
    if not candles:
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

    # V20: Check if M15 CHoCH aligns with HTF trend direction
    has_ltf_choch_sync = False
    if htf_zones and choch_events:
        confirmed_chochs = [c for c in choch_events if c.confirmed and not c.is_fake]
        if confirmed_chochs:
            latest_choch = max(confirmed_chochs, key=lambda c: c.candle_index)
            # CHoCH direction should match trend (HTF bias)
            if (trend == TrendState.BULLISH and latest_choch.direction == Direction.BULLISH) or \
               (trend == TrendState.BEARISH and latest_choch.direction == Direction.BEARISH):
                has_ltf_choch_sync = True

    # Skip trend-aligned signals only if trend is RANGING (not for all cases)
    if trend != TrendState.RANGING:
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

            # V20: Add CHoCH sync confluence
            if has_ltf_choch_sync:
                confluences.append("LTF CHoCH sync")

            # Grade
            grade = _grade_signal(confluences, False, climax_warning)

            # Confidence score (0-100)
            base_score = len(confluences) * 15
            if has_multi_tf:
                base_score += 10
            if session and session.is_kill_zone:
                base_score += 5
            if has_ltf_choch_sync:
                base_score += 5
            if climax_warning:
                base_score -= 20
            confidence = max(0, min(100, base_score))

            # V15/V17: Determine entry method from recent structure events
            entry_method = _determine_entry_method(
                zone, bos_events, choch_events,
            )

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
                entry_method=entry_method,
                pattern_type=f"{zone['type']} trend continuation",
                timestamp=candles[-1].timestamp,
                w1_trend=w1_trend,
                d1_trend=d1_trend,
                session=session,
                climax_warning=climax_warning,
                is_counter_trend=False,
            ))

    # V21: Counter-trend signal generation
    ct_dir = _check_counter_trend_conditions(candles, inducements, fvgs, trend)
    if ct_dir:
        # Find zones in counter-trend direction
        ct_trend = TrendState.BULLISH if ct_dir == Direction.BULLISH else TrendState.BEARISH
        ct_zones = _find_active_zones(order_blocks, fvgs, ct_trend)

        for zone in ct_zones:
            if not _check_zone_tap(zone, current_price):
                continue

            sl = _calculate_sl(zone, ct_dir)
            tp = _get_counter_trend_tp(ct_dir, zone, order_blocks, fvgs)
            if tp is None:
                continue

            entry = current_price
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk == 0:
                continue
            rr = reward / risk
            if rr < MIN_RISK_REWARD:
                continue

            confluences = ["Counter-trend IDM body close"]
            if pd:
                if ct_dir == Direction.BULLISH and pd.zone == ZoneType.DISCOUNT:
                    confluences.append("Discount zone (buy)")
                elif ct_dir == Direction.BEARISH and pd.zone == ZoneType.PREMIUM:
                    confluences.append("Premium zone (sell)")

            grade = _grade_signal(confluences, True, climax_warning)
            confidence = max(0, min(100, len(confluences) * 10))

            signals.append(TradingSignal(
                direction=ct_dir,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                risk_reward_ratio=round(rr, 2),
                confidence_score=confidence,
                grade=grade,
                confluences=confluences,
                timeframe="M15",
                entry_method=EntryMethod.MSS,
                pattern_type=f"{zone['type']} counter-trend",
                timestamp=candles[-1].timestamp,
                w1_trend=w1_trend,
                d1_trend=d1_trend,
                session=session,
                climax_warning=climax_warning,
                is_counter_trend=True,
            ))

    # V23: Deduplicate — one signal per zone
    return _deduplicate_signals(signals)
