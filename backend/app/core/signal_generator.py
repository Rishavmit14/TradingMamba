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
10. VSA ABSORPTION → ultra-high volume institutional flow = confirmation
11. DETECT CHoCH → D1 HL/LH break = direction switch
12. COUNTER-TREND → D1 BOS + IDM close + FVG → TP first opposing zone
13. SBC ENTRY → sweep liquidity (wick) + body close opposite side = standalone entry
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, Inducement, LiquidityPool, BOS, CHoCH,
    FVG, OrderBlock, PremiumDiscount, Session, TradingSignal,
    Direction, TrendState, SwingType, SwingClassification,
    ZoneType, EntryMethod, SignalGrade, IDMStatus, MSSGrade,
    LiquidityType, LiquiditySource, LiquidityEvent,
)
from app.core.liquidity import prices_equal
from app.config import (
    SL_BUFFER_PCT, MIN_RISK_REWARD,
    SBC_MIN_RISK_REWARD, SBC_RECENCY_WINDOW, SBC_CONFIRM_WINDOW,
)


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


def _check_zone_tap(zone: dict, candles: list[Candle], lookback: int = 10) -> bool:
    """Check if price has tapped (reached) a zone in the last N candles.

    Step 6: price must reach the identified zone before entry.
    Uses the high/low range of recent candles — not just the last close —
    to catch wicks that tapped the zone.
    Default lookback=10 (2.5h on M15) balances recency with detection rate.
    """
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    for c in recent:
        # Any candle whose range overlaps the zone counts as a tap
        if c.low <= zone["upper"] and c.high >= zone["lower"]:
            return True
    return False


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
    entry: float = 0,
    sl: float = 0,
) -> float | None:
    """Calculate take profit.

    V23: TP = previous high/low from which market took inducement.
    Skips swings that are too close (would give R:R below minimum).
    """
    risk = abs(entry - sl) if entry and sl else 0

    if direction == Direction.BULLISH:
        # TP at the swing high above current zone
        for swing in sorted(swings, key=lambda s: s.price):
            if (swing.swing_type == SwingType.SWING_HIGH
                    and swing.price > zone["upper"]):
                # Skip swings too close for acceptable R:R
                if risk > 0:
                    reward = abs(swing.price - entry)
                    if reward / risk < MIN_RISK_REWARD:
                        continue
                return swing.price
    else:
        # TP at the swing low below current zone
        for swing in sorted(swings, key=lambda s: s.price, reverse=True):
            if (swing.swing_type == SwingType.SWING_LOW
                    and swing.price < zone["lower"]):
                # Skip swings too close for acceptable R:R
                if risk > 0:
                    reward = abs(swing.price - entry)
                    if reward / risk < MIN_RISK_REWARD:
                        continue
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


def _grade_signal(
    confluences: list[str],
    is_counter_trend: bool,
    vsa_match: bool = False,
    mss_grade: MSSGrade = MSSGrade.NONE,
) -> SignalGrade:
    """Assign signal grade based on confluences + V25 MSS quality.

    V25 MSS grading directly influences signal grade:
    - A++ MSS (BPR) → auto grade A (institutional confluence)
    - A+ MSS (iFVG) → promotes grade by 1 tier
    - Standard MSS → counted as extra confluence

    V22 VSA Absorption: counted as extra confluence (positive).

    Base grading from confluences:
    A: 3+ confluences, trend-aligned, kill zone, multi-TF
    B: 2 confluences, trend-aligned
    C: 1 confluence or weak alignment
    D: Counter-trend
    """
    if is_counter_trend:
        return SignalGrade.D

    count = len(confluences)

    # V25: A++ MSS with BPR = automatic grade A
    if mss_grade == MSSGrade.A_PLUS_PLUS:
        return SignalGrade.A

    # Base grade from confluences
    if count >= 3:
        base = SignalGrade.A
    elif count >= 2:
        base = SignalGrade.B
    elif count >= 1:
        base = SignalGrade.C
    else:
        base = SignalGrade.D

    # V25: A+ MSS promotes grade by one tier
    if mss_grade == MSSGrade.A_PLUS:
        promote = {SignalGrade.D: SignalGrade.C, SignalGrade.C: SignalGrade.B,
                   SignalGrade.B: SignalGrade.A, SignalGrade.A: SignalGrade.A}
        return promote[base]

    return base


def _determine_entry_method(
    zone: dict,
    bos_events: list[BOS],
    choch_events: list[CHoCH],
) -> tuple[EntryMethod, MSSGrade]:
    """V15/V17/V25: Classify entry based on the most recent structural event.

    Compares the latest confirmed CHoCH vs the latest valid BOS:
    - If CHoCH is more recent and is_mss=True → MSS (liq swept + expansion + body close)
    - If CHoCH is more recent and model="sweep" → SBC (wick-only break)
    - If CHoCH is more recent but NOT MSS → PULLBACK_BREAK (just a body-close CHoCH)
    - If BOS is more recent → PULLBACK_BREAK (trend continuation)

    Returns (entry_method, mss_grade) — mss_grade is NONE for non-MSS entries.
    """
    # Find the most recent confirmed CHoCH
    confirmed_chochs = [
        c for c in choch_events
        if c.confirmed and not c.is_fake
    ]
    latest_choch_idx = max(
        (c.candle_index for c in confirmed_chochs), default=-1
    )

    # Find the most recent valid BOS
    valid_bos = [b for b in bos_events if b.valid]
    latest_bos_idx = max(
        (b.candle_index for b in valid_bos), default=-1
    )

    # The most recent structural event determines the entry method
    if latest_choch_idx > latest_bos_idx and latest_choch_idx >= 0:
        latest = max(confirmed_chochs, key=lambda c: c.candle_index)
        if latest.model == "sweep":
            return EntryMethod.SBC, MSSGrade.NONE
        if latest.is_mss:
            return EntryMethod.MSS, latest.mss_grade
        return EntryMethod.PULLBACK_BREAK, MSSGrade.NONE

    return EntryMethod.PULLBACK_BREAK, MSSGrade.NONE


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


def _generate_sbc_signals(
    candles: list[Candle],
    swings: list[SwingPoint],
    liquidity_pools: list[LiquidityPool],
    fvgs: list[FVG],
    trend: TrendState,
    pd: PremiumDiscount | None,
    session: Session | None,
    trade_bias: TrendState,
    has_multi_tf: bool,
    vsa_absorptions: list | None,
    w1_trend: TrendState | None,
    d1_trend: TrendState | None,
) -> list[TradingSignal]:
    """V22 SBC (Sweep Based Change of Character) standalone entry signals.

    SBC = sweep one side's liquidity (wick only) + first candle body close
    on opposite side of the swept swing → direct entry.

    This generates signals INDEPENDENTLY of zone-tap signals. It uses
    existing liquidity sweep data to find SBC patterns and create entries
    at 50% fib or FVG within the sweep swing range.
    """
    if not candles or not liquidity_pools or not swings:
        return []

    signals: list[TradingSignal] = []
    last_idx = candles[-1].index
    current_price = candles[-1].close
    idx_map = {c.index: c for c in candles}

    # Step 1: Find recent wick-only sweeps (SWEEP, not GRAB)
    recent_sweeps = [
        pool for pool in liquidity_pools
        if pool.swept
        and pool.event_type == LiquidityEvent.SWEEP
        and pool.swept_at_candle is not None
        and (last_idx - pool.swept_at_candle) <= SBC_RECENCY_WINDOW
    ]

    if not recent_sweeps:
        return []

    # Sort by sweep time (earliest first) for two-sided resolution
    recent_sweeps.sort(key=lambda p: p.swept_at_candle or 0)

    for pool in recent_sweeps:
        # Step 2: Find the swing point that created this liquidity pool
        swept_swing = None
        for swing in swings:
            if swing.candle_index in pool.candle_indices:
                swept_swing = swing
                break
        if not swept_swing:
            for swing in swings:
                if prices_equal(swing.price, pool.price_level):
                    swept_swing = swing
                    break
        if not swept_swing:
            continue

        swept_swing_candle = idx_map.get(swept_swing.candle_index)
        if not swept_swing_candle:
            continue

        # Step 3: Determine SBC direction
        if pool.pool_type == LiquidityType.SELL_SIDE:
            sbc_dir = Direction.BULLISH   # Low swept → buy
        else:
            sbc_dir = Direction.BEARISH   # High swept → sell

        # Step 4: Check body close confirmation
        # V22: body must close ABOVE the swept swing price (buy)
        #      or BELOW the swept swing price (sell)
        # This confirms the sweep was absorbed and price reversed.
        sweep_idx = pool.swept_at_candle
        confirmation_candle = None
        for c in candles:
            if c.index < sweep_idx:
                continue
            if c.index > sweep_idx + SBC_CONFIRM_WINDOW:
                break
            if sbc_dir == Direction.BULLISH:
                if c.body_top > swept_swing.price:
                    confirmation_candle = c
                    break
            else:
                if c.body_bottom < swept_swing.price:
                    confirmation_candle = c
                    break

        if not confirmation_candle:
            continue

        # Step 5: Define sweep swing range for entry calculation
        sweep_candle = idx_map.get(sweep_idx)
        if not sweep_candle:
            continue

        if sbc_dir == Direction.BULLISH:
            sweep_extreme = sweep_candle.low      # The actual sweep wick low
            swing_opposite = swept_swing.price    # The swing price level
        else:
            sweep_extreme = sweep_candle.high     # The actual sweep wick high
            swing_opposite = swept_swing.price    # The swing price level

        # Entry: FVG within sweep range (preferred) or 50% fib (fallback)
        fvg_entry = None
        sweep_lo = min(sweep_extreme, swing_opposite)
        sweep_hi = max(sweep_extreme, swing_opposite)
        for fvg in fvgs:
            if not fvg.valid or fvg.mitigated:
                continue
            if fvg.direction != sbc_dir:
                continue
            if sweep_lo <= fvg.midpoint <= sweep_hi:
                fvg_entry = fvg.midpoint
                break

        fib_50_entry = (sweep_extreme + swing_opposite) / 2
        ideal_entry = fvg_entry if fvg_entry is not None else fib_50_entry

        # If price already passed the ideal entry, use current price
        # (SBC confirmed → market entry is valid).
        # Skip only if price ran past the sweep range by more than the
        # sweep range size itself (move already played out).
        sweep_range_size = sweep_hi - sweep_lo
        if sbc_dir == Direction.BULLISH:
            if current_price > sweep_hi + sweep_range_size * 3:
                continue  # Move already played out
            entry = max(current_price, ideal_entry)
        else:
            if current_price < sweep_lo - sweep_range_size * 3:
                continue  # Move already played out
            entry = min(current_price, ideal_entry)

        # SL: beyond the sweep extreme
        if sbc_dir == Direction.BULLISH:
            sl = sweep_extreme * (1 - SL_BUFFER_PCT)
        else:
            sl = sweep_extreme * (1 + SL_BUFFER_PCT)

        # TP: first swing beyond entry with >= SBC_MIN_RISK_REWARD R:R
        risk = abs(entry - sl)
        if risk == 0:
            continue

        tp = None
        if sbc_dir == Direction.BULLISH:
            for swing in sorted(swings, key=lambda s: s.price):
                if swing.swing_type == SwingType.SWING_HIGH and swing.price > entry:
                    reward = swing.price - entry
                    if reward / risk >= SBC_MIN_RISK_REWARD:
                        tp = swing.price
                        break
        else:
            for swing in sorted(swings, key=lambda s: s.price, reverse=True):
                if swing.swing_type == SwingType.SWING_LOW and swing.price < entry:
                    reward = entry - swing.price
                    if reward / risk >= SBC_MIN_RISK_REWARD:
                        tp = swing.price
                        break

        if tp is None:
            continue

        rr = abs(tp - entry) / risk

        # Step 6: Build confluences
        confluences: list[str] = ["SBC sweep entry"]

        is_trend_aligned = (
            (trade_bias == TrendState.BULLISH and sbc_dir == Direction.BULLISH)
            or (trade_bias == TrendState.BEARISH and sbc_dir == Direction.BEARISH)
        )
        is_counter_trend = not is_trend_aligned and trade_bias != TrendState.RANGING

        if is_trend_aligned:
            confluences.append("HTF trend aligned")
        if pd:
            if sbc_dir == Direction.BULLISH and pd.zone == ZoneType.DISCOUNT:
                confluences.append("Discount zone (buy)")
            elif sbc_dir == Direction.BEARISH and pd.zone == ZoneType.PREMIUM:
                confluences.append("Premium zone (sell)")
        if session and session.is_kill_zone:
            confluences.append("Kill zone active")
        if has_multi_tf:
            confluences.append("Multi-TF aligned")
        if fvg_entry is not None:
            confluences.append("FVG entry refinement")
        if pool.source in (LiquiditySource.EQUAL_HIGHS, LiquiditySource.EQUAL_LOWS):
            confluences.append("Major liquidity pool swept")

        # VSA absorption confluence
        vsa_match = False
        if vsa_absorptions:
            for vsa in vsa_absorptions:
                if vsa.direction == sbc_dir and abs(vsa.candle_index - last_idx) <= 20:
                    vsa_match = True
                    confluences.append("VSA Absorption")
                    break

        # Grade: CT SBC capped at B (V22 shows CT SBC works), otherwise normal
        if is_counter_trend:
            count = len(confluences)
            if count >= 3:
                grade = SignalGrade.B
            elif count >= 2:
                grade = SignalGrade.C
            else:
                grade = SignalGrade.D
        else:
            grade = _grade_signal(confluences, False, vsa_match)

        # Confidence score
        base_score = len(confluences) * 15
        if has_multi_tf:
            base_score += 10
        if session and session.is_kill_zone:
            base_score += 5
        if fvg_entry is not None:
            base_score += 10
        if vsa_match:
            base_score += 15
        if pool.source in (LiquiditySource.EQUAL_HIGHS, LiquiditySource.EQUAL_LOWS):
            base_score += 10
        confidence = max(0, min(100, base_score))

        signals.append(TradingSignal(
            direction=sbc_dir,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            risk_reward_ratio=round(rr, 2),
            confidence_score=confidence,
            grade=grade,
            confluences=confluences,
            timeframe="M15",
            entry_method=EntryMethod.SBC,
            pattern_type="SBC sweep entry",
            timestamp=candles[-1].timestamp,
            w1_trend=w1_trend,
            d1_trend=d1_trend,
            session=session,
            vsa_absorption=vsa_match,
            is_counter_trend=is_counter_trend,
        ))

        # Limit: max 1 SBC signal per analysis cycle to prevent over-generation
        if signals:
            break

    return signals


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
    vsa_absorptions: list | None = None,
    htf_zones: list[dict] | None = None,
) -> list[TradingSignal]:
    """The Master Checklist — generate trading signals from all detector outputs.

    This is the top-level function that orchestrates the entire system.
    V20: htf_zones passed from higher TF for zone alignment.
    V22: vsa_absorptions passed for institutional flow confluence.
    """
    if not candles:
        return []

    current_price = candles[-1].close
    signals: list[TradingSignal] = []

    # V23 Step 1: Trade bias comes from HTF (W1 → D1 → M15 fallback)
    # W1 sets the macro direction, D1 confirms, M15 is for entry timing
    trade_bias = trend  # default to M15 trend
    if w1_trend and w1_trend != TrendState.RANGING:
        trade_bias = w1_trend
    elif d1_trend and d1_trend != TrendState.RANGING:
        trade_bias = d1_trend

    # Multi-TF alignment: M15 trend matches the HTF bias
    has_multi_tf = (trade_bias == trend and trade_bias != TrendState.RANGING)

    # V20: Check if M15 CHoCH aligns with HTF trend direction
    has_ltf_choch_sync = False
    if htf_zones and choch_events:
        confirmed_chochs = [c for c in choch_events if c.confirmed and not c.is_fake]
        if confirmed_chochs:
            latest_choch = max(confirmed_chochs, key=lambda c: c.candle_index)
            if (trade_bias == TrendState.BULLISH and latest_choch.direction == Direction.BULLISH) or \
               (trade_bias == TrendState.BEARISH and latest_choch.direction == Direction.BEARISH):
                has_ltf_choch_sync = True

    # ── Build combined zone pool: M15 zones + HTF zones (V20) ──
    def _build_zone_pool(bias: TrendState) -> list[dict]:
        """Collect M15 zones + HTF zones matching the given bias direction."""
        zones = _find_active_zones(order_blocks, fvgs, bias)
        if htf_zones:
            bias_dir = Direction.BULLISH if bias == TrendState.BULLISH else Direction.BEARISH
            for hz in htf_zones:
                hz_dir = Direction.BULLISH if hz["direction"] == "bullish" else Direction.BEARISH
                if hz_dir == bias_dir:
                    zones.append({
                        "type": hz["type"],
                        "upper": hz["upper"],
                        "lower": hz["lower"],
                        "midpoint": (hz["upper"] + hz["lower"]) / 2,
                        "direction": hz_dir,
                        "has_fvg": hz["type"] == "FVG",
                        "candle_index": 0,
                    })
        return zones

    # Generate trend-aligned signals using HTF bias for zone selection
    if trade_bias != TrendState.RANGING:
        # Step 4: Find active zones matching HTF bias direction (M15 + HTF)
        zones = _build_zone_pool(trade_bias)

        for zone in zones:
            # Step 6: Check zone tap
            if not _check_zone_tap(zone, candles):
                continue

            direction = zone["direction"]

            # Step 8: Calculate SL
            sl = _calculate_sl(zone, direction)

            # Step 9: Calculate TP (skip swings too close for min R:R)
            tp = _calculate_tp(zone, direction, swings, inducements,
                               entry=current_price, sl=sl)
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
                zone, pd, session, trade_bias,
                bos_events, choch_events, has_multi_tf,
            )

            # V20: Add CHoCH sync confluence
            if has_ltf_choch_sync:
                confluences.append("LTF CHoCH sync")

            # V15/V17/V25: Determine entry method + MSS quality grade
            entry_method, mss_grade = _determine_entry_method(
                zone, bos_events, choch_events,
            )

            # V25: Add MSS quality as confluence
            if mss_grade == MSSGrade.A_PLUS_PLUS:
                confluences.append("MSS A++ (BPR)")
            elif mss_grade == MSSGrade.A_PLUS:
                confluences.append("MSS A+ (iFVG)")
            elif mss_grade == MSSGrade.STANDARD:
                confluences.append("MSS Standard")

            # V22: Check for VSA absorption matching signal direction
            vsa_match = False
            if vsa_absorptions:
                for vsa in vsa_absorptions:
                    if vsa.direction == direction and abs(vsa.candle_index - len(candles) + 1) <= 20:
                        vsa_match = True
                        confluences.append("VSA Absorption")
                        break

            # Grade (V25 MSS quality influences grading)
            grade = _grade_signal(confluences, False, vsa_match, mss_grade)

            # Confidence score (0-100) with V25 MSS quality boost
            base_score = len(confluences) * 15
            if has_multi_tf:
                base_score += 10
            if session and session.is_kill_zone:
                base_score += 5
            if has_ltf_choch_sync:
                base_score += 5
            # V25: MSS quality confidence boost
            if mss_grade == MSSGrade.A_PLUS_PLUS:
                base_score += 30
            elif mss_grade == MSSGrade.A_PLUS:
                base_score += 20
            elif mss_grade == MSSGrade.STANDARD:
                base_score += 10
            # V22: VSA absorption boosts confidence (institutional flow confirmation)
            if vsa_match:
                base_score += 15
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
                entry_method=entry_method,
                pattern_type=f"{zone['type']} trend continuation",
                timestamp=candles[-1].timestamp,
                w1_trend=w1_trend,
                d1_trend=d1_trend,
                session=session,
                vsa_absorption=vsa_match,
                is_counter_trend=False,
                mss_quality=mss_grade.value if mss_grade != MSSGrade.NONE else "",
            ))

    # ── Counter-trend signal generation ──
    # Use trade_bias (not M15 trend) to determine counter-trend direction.
    # This ensures CT signals are generated even when M15 is ranging.
    # V21 strict conditions (IDM body close + FVG) are a bonus, not a gate.
    if trade_bias in (TrendState.BULLISH, TrendState.BEARISH):
        ct_dir = Direction.BEARISH if trade_bias == TrendState.BULLISH else Direction.BULLISH
        ct_trend = TrendState.BEARISH if trade_bias == TrendState.BULLISH else TrendState.BULLISH
        ct_zones = _build_zone_pool(ct_trend)

        # V21: Check if strict counter-trend conditions are met (bonus confluence)
        v21_met = _check_counter_trend_conditions(
            candles, inducements, fvgs, trade_bias,
        ) is not None

        for zone in ct_zones:
            if not _check_zone_tap(zone, candles):
                continue

            sl = _calculate_sl(zone, ct_dir)

            # Try V21 opposing-zone TP first, fall back to swing-based TP
            tp = _get_counter_trend_tp(ct_dir, zone, order_blocks, fvgs)
            if tp is None:
                tp = _calculate_tp(zone, ct_dir, swings, inducements,
                                   entry=current_price, sl=sl)
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

            confluences = []
            if v21_met:
                confluences.append("Counter-trend IDM body close")
            confluences.append(f"Counter-trend {zone['type']}")
            if zone["has_fvg"]:
                confluences.append("FVG")
            if pd:
                if ct_dir == Direction.BULLISH and pd.zone == ZoneType.DISCOUNT:
                    confluences.append("Discount zone (buy)")
                elif ct_dir == Direction.BEARISH and pd.zone == ZoneType.PREMIUM:
                    confluences.append("Premium zone (sell)")
            if session and session.is_kill_zone:
                confluences.append("Kill zone active")

            grade = _grade_signal(confluences, True)
            base_score = len(confluences) * 10
            if v21_met:
                base_score += 15
            confidence = max(0, min(100, base_score))

            ct_entry_method, _ = _determine_entry_method(zone, bos_events, choch_events)

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
                entry_method=ct_entry_method,
                pattern_type=f"{zone['type']} counter-trend",
                timestamp=candles[-1].timestamp,
                w1_trend=w1_trend,
                d1_trend=d1_trend,
                session=session,
                vsa_absorption=False,
                is_counter_trend=True,
            ))

    # ── V22 SBC (Sweep Based Change) standalone signal generation ──
    # SBC entries are independent of zone-tapping: sweep + body close = entry
    sbc_signals = _generate_sbc_signals(
        candles=candles,
        swings=swings,
        liquidity_pools=liquidity_pools,
        fvgs=fvgs,
        trend=trend,
        pd=pd,
        session=session,
        trade_bias=trade_bias,
        has_multi_tf=has_multi_tf,
        vsa_absorptions=vsa_absorptions,
        w1_trend=w1_trend,
        d1_trend=d1_trend,
    )
    signals.extend(sbc_signals)

    # V23: Deduplicate — one signal per zone
    return _deduplicate_signals(signals)
