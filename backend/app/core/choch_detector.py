"""1.5 — Change of Character (CHoCH) Detection

Detects potential trend reversals where price breaks major swing points.

Core Rules (V07, V09, V10):
- CHoCH = break of the current MAJOR swing Higher Low (bullish→bearish)
         or Lower High (bearish→bullish)
- Candle body MUST close beyond the level
- Distinguish from inducement: what retail calls CHoCH is often just IDM being taken
- Fake CHoCH filters (V09): 3 conditions that invalidate apparent CHoCH
- CHoCH confirmation (V10): requires follow-through structure
- VSA Absorption + CHoCH = strong reversal signal (V22/V23)
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, BOS, CHoCH, Inducement, LiquidityPool, FVG,
    SwingType, SwingClassification, Direction, TrendState,
    IDMStatus, LiquiditySource, LiquidityType, MSSGrade,
)
from app.core.liquidity import prices_equal


def _add_choch_break(
    candles: list[Candle],
    key_swing: SwingPoint,
    trigger_swing: SwingPoint,
    direction: Direction,
    has_vsa: bool,
    choch_events: list[CHoCH],
) -> None:
    """Find the candle that broke the key swing level and add a CHoCH event.

    Scans candles between the key swing and the trigger swing.
    Detects both body-close breaks (swing-based → MSS) and wick-only
    breaks (sweep-based → SBC) per V10.
    """
    for candle in candles:
        if candle.index <= key_swing.candle_index:
            continue
        if candle.index > trigger_swing.candle_index + 2:
            break

        body_broke = False
        wick_broke = False
        if direction == Direction.BEARISH:
            body_broke = candle.body_bottom < key_swing.price
            wick_broke = not body_broke and candle.low < key_swing.price
        else:
            body_broke = candle.body_top > key_swing.price
            wick_broke = not body_broke and candle.high > key_swing.price

        if body_broke or wick_broke:
            confidence = 0.5 if body_broke else 0.35
            if has_vsa:
                confidence += 0.25
            if body_broke:
                if direction == Direction.BEARISH:
                    if candle.body_bottom < key_swing.price * 0.998:
                        confidence += 0.15
                else:
                    if candle.body_top > key_swing.price * 1.002:
                        confidence += 0.15

            choch_events.append(CHoCH(
                candle_index=candle.index,
                direction=direction,
                broken_swing_index=key_swing.candle_index,
                broken_price=key_swing.price,
                confidence=min(confidence, 1.0),
                has_vsa_confluence=has_vsa,
            ))
            break


def detect_choch(
    candles: list[Candle],
    swings: list[SwingPoint],
    bos_events: list[BOS],
    trend: TrendState,
    vsa_absorptions: list | None = None,
) -> list[CHoCH]:
    """Detect CHoCH by walking through swing structure chronologically.

    Instead of using a single end-state trend, walks through the swing
    sequence tracking the running trend and detects CHoCH at each reversal:
    - Bullish → Bearish: candle body closes below the key HL
    - Bearish → Bullish: candle body closes above the key LH

    Also checks if current price action is breaking the key level
    (potential CHoCH before new swing structure confirms it).

    vsa_absorptions: VSA absorption objects for confluence marking.
    A CHoCH within 5 candles of a matching VSA absorption gets has_vsa_confluence=True.
    """
    if len(swings) < 4:
        return []

    choch_events: list[CHoCH] = []
    vsa_absorptions = vsa_absorptions or []

    def _has_vsa_near(candle_idx: int, direction: Direction) -> bool:
        """Check if a VSA absorption exists within 5 candles before candle_idx
        with matching direction."""
        for vsa in vsa_absorptions:
            if vsa.direction == direction and candle_idx - 5 <= vsa.candle_index <= candle_idx:
                return True
        return False

    running_trend = TrendState.RANGING
    recent_high: SwingPoint | None = None
    recent_low: SwingPoint | None = None
    key_hl: SwingPoint | None = None  # Bullish defense level (break = bearish CHoCH)
    key_lh: SwingPoint | None = None  # Bearish defense level (break = bullish CHoCH)

    for i, swing in enumerate(swings):
        if swing.classification == SwingClassification.UNCLASSIFIED:
            if swing.swing_type == SwingType.SWING_HIGH:
                recent_high = swing
            else:
                recent_low = swing
            continue

        old_trend = running_trend

        if swing.swing_type == SwingType.SWING_HIGH:
            recent_high = swing
        else:
            recent_low = swing

        # Determine running trend from most recent classified high + low
        if (recent_high and recent_low
                and recent_high.classification != SwingClassification.UNCLASSIFIED
                and recent_low.classification != SwingClassification.UNCLASSIFIED):
            h_cls = recent_high.classification
            l_cls = recent_low.classification

            if h_cls == SwingClassification.HH and l_cls == SwingClassification.HL:
                running_trend = TrendState.BULLISH
            elif h_cls == SwingClassification.LH and l_cls == SwingClassification.LL:
                running_trend = TrendState.BEARISH
            elif h_cls == SwingClassification.HH and l_cls == SwingClassification.LL:
                # HL broke while highs still higher → bearish CHoCH
                if old_trend == TrendState.BULLISH:
                    running_trend = TrendState.BEARISH
            elif h_cls == SwingClassification.LH and l_cls == SwingClassification.HL:
                # LH broke while lows still higher → bullish CHoCH
                if old_trend == TrendState.BEARISH:
                    running_trend = TrendState.BULLISH

        # --- Handle trend transitions ---

        # RANGING → established trend: initialize key levels (no CHoCH)
        if old_trend == TrendState.RANGING and running_trend != TrendState.RANGING:
            if running_trend == TrendState.BULLISH:
                for s in reversed(swings[:i + 1]):
                    if s.classification == SwingClassification.HL and s.is_valid_smc:
                        key_hl = s
                        break
            elif running_trend == TrendState.BEARISH:
                for s in reversed(swings[:i + 1]):
                    if s.classification == SwingClassification.LH and s.is_valid_smc:
                        key_lh = s
                        break

        # BULLISH → BEARISH: bearish CHoCH (broke the key HL)
        elif old_trend == TrendState.BULLISH and running_trend == TrendState.BEARISH:
            if key_hl:
                _add_choch_break(candles, key_hl, swing,
                                 Direction.BEARISH, _has_vsa_near(swing.candle_index, Direction.BEARISH), choch_events)
            # Initialize key_lh for the new bearish trend
            for s in reversed(swings[:i + 1]):
                if s.classification == SwingClassification.LH and s.is_valid_smc:
                    key_lh = s
                    break
            key_hl = None

        # BEARISH → BULLISH: bullish CHoCH (broke the key LH)
        elif old_trend == TrendState.BEARISH and running_trend == TrendState.BULLISH:
            if key_lh:
                _add_choch_break(candles, key_lh, swing,
                                 Direction.BULLISH, _has_vsa_near(swing.candle_index, Direction.BULLISH), choch_events)
            # Initialize key_hl for the new bullish trend
            for s in reversed(swings[:i + 1]):
                if s.classification == SwingClassification.HL and s.is_valid_smc:
                    key_hl = s
                    break
            key_lh = None

        # Update key levels within current trend to track most recent HL/LH
        if running_trend == TrendState.BULLISH:
            if swing.classification == SwingClassification.HL and swing.is_valid_smc:
                key_hl = swing
        elif running_trend == TrendState.BEARISH:
            if swing.classification == SwingClassification.LH and swing.is_valid_smc:
                key_lh = swing

    # Edge case: check if current price breaks the key level (live CHoCH)
    # Since key_hl/key_lh are only set at trend transitions (not updated within
    # the trend), find the most recent HL/LH for live detection.
    if running_trend == TrendState.BULLISH:
        live_hl = None
        for s in reversed(swings):
            if s.classification == SwingClassification.HL and s.is_valid_smc:
                live_hl = s
                break
        if live_hl:
            last_swing_idx = swings[-1].candle_index
            for candle in candles[:-1]:  # exclude live (unfinished) candle
                if candle.index <= max(live_hl.candle_index, last_swing_idx):
                    continue
                body_broke = candle.body_bottom < live_hl.price
                wick_broke = not body_broke and candle.low < live_hl.price
                if body_broke or wick_broke:
                    vsa_here = _has_vsa_near(candle.index, Direction.BEARISH)
                    confidence = 0.5 if body_broke else 0.35
                    if vsa_here:
                        confidence += 0.25
                    if body_broke and candle.body_bottom < live_hl.price * 0.998:
                        confidence += 0.15
                    choch_events.append(CHoCH(
                        candle_index=candle.index,
                        direction=Direction.BEARISH,
                        broken_swing_index=live_hl.candle_index,
                        broken_price=live_hl.price,
                        confidence=min(confidence, 1.0),
                        has_vsa_confluence=vsa_here,
                    ))
                    break

    elif running_trend == TrendState.BEARISH:
        live_lh = None
        for s in reversed(swings):
            if s.classification == SwingClassification.LH and s.is_valid_smc:
                live_lh = s
                break
        if live_lh:
            last_swing_idx = swings[-1].candle_index
            for candle in candles[:-1]:  # exclude live (unfinished) candle
                if candle.index <= max(live_lh.candle_index, last_swing_idx):
                    continue
                body_broke = candle.body_top > live_lh.price
                wick_broke = not body_broke and candle.high > live_lh.price
                if body_broke or wick_broke:
                    vsa_here = _has_vsa_near(candle.index, Direction.BULLISH)
                    confidence = 0.5 if body_broke else 0.35
                    if vsa_here:
                        confidence += 0.25
                    if body_broke and candle.body_top > live_lh.price * 1.002:
                        confidence += 0.15
                    choch_events.append(CHoCH(
                        candle_index=candle.index,
                        direction=Direction.BULLISH,
                        broken_swing_index=live_lh.candle_index,
                        broken_price=live_lh.price,
                        confidence=min(confidence, 1.0),
                        has_vsa_confluence=vsa_here,
                    ))
                    break

    # Deduplicate: same (candle_index, broken_swing_index) can appear from
    # both the main loop and the live edge-case check
    seen = set()
    unique = []
    for ch in choch_events:
        key = (ch.candle_index, ch.broken_swing_index)
        if key not in seen:
            seen.add(key)
            unique.append(ch)
    return unique


def _classify_choch_model(
    choch: CHoCH,
    idx_map: dict[int, Candle],
) -> str:
    """Classify CHoCH as 'swing' or 'sweep' based (V10).

    Swing-Based: candle BODY closed beyond the broken swing level.
    Sweep-Based: only WICK went beyond (body stayed on original side).
    """
    break_candle = idx_map.get(choch.candle_index)
    if not break_candle:
        return "swing"  # default to swing-based

    if choch.direction == Direction.BEARISH:
        # Body closed below broken price → swing-based
        if break_candle.body_bottom < choch.broken_price:
            return "swing"
        # Only wick went below → sweep-based (liquidity sweep)
        return "sweep"
    else:
        if break_candle.body_top > choch.broken_price:
            return "swing"
        return "sweep"


def _check_weak_swing(
    choch: CHoCH,
    swings: list[SwingPoint],
    idx_map: dict[int, Candle],
) -> bool:
    """V09 Rule 1: Check if the broken swing point was WEAK.

    A swing is weak when its Swing Initiating Candle (SIC) simultaneously
    sweeps the immediately preceding opposite-type swing's liquidity.
    The energy was spent on the sweep, not building genuine structure.

    Only checks the IMMEDIATE structural predecessor (the directly preceding
    opposite-type swing), and only if that swing's price falls within the
    SIC candle's range — confirming the SIC actually reached for it.
    """
    swing_map = {s.candle_index: s for s in swings}
    broken_swing = swing_map.get(choch.broken_swing_index)
    if not broken_swing:
        return False

    sic = idx_map.get(broken_swing.candle_index)
    if not sic:
        return False

    # Find the IMMEDIATELY PRECEDING opposite-type swing (structural predecessor)
    opposite_type = (SwingType.SWING_LOW
                     if broken_swing.swing_type == SwingType.SWING_HIGH
                     else SwingType.SWING_HIGH)
    prior = None
    for s in reversed(swings):
        if s.swing_type == opposite_type and s.candle_index < broken_swing.candle_index:
            prior = s
            break

    if not prior:
        return False

    if broken_swing.swing_type == SwingType.SWING_HIGH:
        # SIC created a swing HIGH — did its LOW also sweep the prior swing low?
        # Prior swing low must be within SIC's range to be a genuine sweep
        return sic.low < prior.price <= sic.high
    else:
        # SIC created a swing LOW — did its HIGH also sweep the prior swing high?
        # Prior swing high must be within SIC's range to be a genuine sweep
        return sic.low <= prior.price < sic.high


def _check_engineered_liquidity(
    choch: CHoCH,
    liquidity_pools: list[LiquidityPool],
) -> bool:
    """V09 Rule 2: Check if the broken swing has engineered liquidity.

    If the broken swing price matches equal highs/lows (engineered liquidity),
    breaking it is just a liquidity sweep, not real CHoCH.
    """
    for pool in liquidity_pools:
        if pool.source not in (LiquiditySource.EQUAL_HIGHS, LiquiditySource.EQUAL_LOWS):
            continue
        if prices_equal(choch.broken_price, pool.price_level):
            return True
    return False


def _check_impulse_origin(
    choch: CHoCH,
    inducements: list[Inducement],
) -> bool:
    """V09 Rule 3: Check if the broken swing was an impulse-origin level.

    If the IDM for the broken swing has TRANSFERRED status, it means no
    pullback was found in the current swing range — the move was impulsive.
    Breaking an impulse-origin level is just IDM being taken, not real CHoCH.
    """
    for idm in inducements:
        if idm.parent_swing_index == choch.broken_swing_index:
            if idm.status == IDMStatus.TRANSFERRED:
                return True
            break
    return False


def _confirm_swing_based(
    choch: CHoCH,
    candles: list[Candle],
    idx_map: dict[int, Candle],
) -> tuple[bool, bool]:
    """V10 Swing-Based Confirmation (4 rules).

    Rules 1-2 are already satisfied by detect_choch() (major swing + body close).
    Here we check Rules 3-4:
      Rule 3: After CHoCH, price must create inducement (pullback in new direction)
      Rule 4: Market must take that inducement and create a BOS

    Returns:
        (confirmed, is_no_idm_trap)
        - confirmed=True if all 4 rules pass
        - is_no_idm_trap=True if no inducement forms after CHoCH (V10 trap)
    """
    choch_candle = idx_map.get(choch.candle_index)
    if not choch_candle:
        return False, False

    window = 20  # candles to look ahead for confirmation
    end_idx = choch.candle_index + window

    # Scan for inducement creation (a pullback in the new trend direction)
    pullback_price = None
    pullback_idx = None
    made_progress = False  # Did price move in CHoCH direction first?

    for candle in candles:
        if candle.index <= choch.candle_index:
            continue
        if candle.index > end_idx:
            break

        if choch.direction == Direction.BEARISH:
            # After bearish CHoCH: expect price to drop, then pull back UP
            if candle.low < choch_candle.low:
                made_progress = True
            if made_progress and pullback_price is None:
                # Looking for a rally (pullback up)
                if candle.high > candle.body_top:  # Has upper wick = some retracement
                    pullback_price = candle.high
                    pullback_idx = candle.index
                elif candle.is_bullish:  # Bullish candle = pullback
                    pullback_price = candle.high
                    pullback_idx = candle.index
        else:
            # After bullish CHoCH: expect price to rise, then pull back DOWN
            if candle.high > choch_candle.high:
                made_progress = True
            if made_progress and pullback_price is None:
                if candle.low < candle.body_bottom:
                    pullback_price = candle.low
                    pullback_idx = candle.index
                elif candle.is_bearish:
                    pullback_price = candle.low
                    pullback_idx = candle.index

    if pullback_price is None:
        # No pullback/inducement formed after CHoCH
        # If price kept moving in CHoCH direction without pullback → unconfirmed (live)
        # If price reversed back → no-IDM trap (fake)
        if not made_progress:
            # Price didn't even move in CHoCH direction → trap
            return False, True
        # Price is moving but no pullback yet → unconfirmed, not necessarily fake
        return False, False

    # Rule 4: Check if inducement was taken (price breaks past the pullback)
    for candle in candles:
        if candle.index <= pullback_idx:
            continue
        if candle.index > end_idx:
            break

        if choch.direction == Direction.BEARISH:
            # Bearish: inducement (pullback high) taken when price goes above then
            # creates BOS (new low below the post-pullback structure)
            if candle.low < choch_candle.low:
                return True, False  # Mini-BOS confirmed
        else:
            if candle.high > choch_candle.high:
                return True, False

    # Inducement exists but not yet broken → unconfirmed (awaiting Rule 4)
    return False, False


def _confirm_sweep_based(
    choch: CHoCH,
    swings: list[SwingPoint],
    candles: list[Candle],
    idx_map: dict[int, Candle],
) -> bool:
    """V10 Sweep-Based Confirmation (2 rules).

    Rule 1: Price swept liquidity (wick only) — already classified as sweep-based.
    Rule 2: Candle body must close below/above the valid pullback between
            the broken swing and the previous opposite-type swing.

    Returns True if confirmed.
    """
    swing_map = {s.candle_index: s for s in swings}
    broken_swing = swing_map.get(choch.broken_swing_index)
    if not broken_swing:
        return False

    # Find the previous opposite-type swing (the swing before the broken one)
    opposite_type = (SwingType.SWING_LOW
                     if broken_swing.swing_type == SwingType.SWING_HIGH
                     else SwingType.SWING_HIGH)
    prev_opposite = None
    for s in reversed(swings):
        if s.swing_type == opposite_type and s.candle_index < broken_swing.candle_index:
            prev_opposite = s
            break

    if not prev_opposite:
        return False

    # Find the valid pullback between prev_opposite and broken_swing
    # This is the internal retracement high (for bearish) or low (for bullish)
    start_idx = min(prev_opposite.candle_index, broken_swing.candle_index)
    end_idx = max(prev_opposite.candle_index, broken_swing.candle_index)

    pullback_level = None
    if choch.direction == Direction.BEARISH:
        # Bearish CHoCH: find the internal LOW between the two swings
        for candle in candles:
            if candle.index <= start_idx or candle.index >= end_idx:
                continue
            if pullback_level is None or candle.low < pullback_level:
                pullback_level = candle.low
    else:
        # Bullish CHoCH: find the internal HIGH between the two swings
        for candle in candles:
            if candle.index <= start_idx or candle.index >= end_idx:
                continue
            if pullback_level is None or candle.high > pullback_level:
                pullback_level = candle.high

    if pullback_level is None:
        return False

    # Rule 2: Check if break candle's body closed beyond the pullback level
    break_candle = idx_map.get(choch.candle_index)
    if not break_candle:
        return False

    if choch.direction == Direction.BEARISH:
        return break_candle.body_bottom < pullback_level
    else:
        return break_candle.body_top > pullback_level


def filter_fake_choch(
    choch_events: list[CHoCH],
    swings: list[SwingPoint],
    candles: list[Candle],
    inducements: list[Inducement] | None = None,
    liquidity_pools: list[LiquidityPool] | None = None,
    bos_events: list[BOS] | None = None,
) -> list[CHoCH]:
    """Apply V09 fake CHoCH filters and V10 confirmation models.

    V09 Fake CHoCH Rules (set is_fake=True):
      Rule 1: Weak Swing Point — SIC sweeps prior liquidity
      Rule 2: Engineered Liquidity — broken swing has equal highs/lows
      Rule 3: Impulse-Origin — broken swing's IDM was TRANSFERRED (no pullback)

    V10 CHoCH Confirmation (set confirmed=True/False):
      Swing-Based Model (body close): 4 rules including post-CHoCH IDM + BOS
      Sweep-Based Model (wick only): 2 rules — sweep + body close below pullback

    Note: V09 Rule 4 (multi-TF trap) is applied separately in run_multi_tf_analysis().
    """
    if inducements is None:
        inducements = []
    if liquidity_pools is None:
        liquidity_pools = []
    if bos_events is None:
        bos_events = []

    idx_map = {c.index: c for c in candles}

    for choch in choch_events:
        # ── V09: Fake CHoCH Detection ──

        # Rule 1: Weak swing point (SIC swept prior liquidity)
        if _check_weak_swing(choch, swings, idx_map):
            choch.is_fake = True
            continue

        # Rule 2: Engineered liquidity at broken swing level
        if _check_engineered_liquidity(choch, liquidity_pools):
            choch.is_fake = True
            continue

        # Rule 3: Impulse-origin inducement (TRANSFERRED IDM = no pullback)
        if _check_impulse_origin(choch, inducements):
            choch.is_fake = True
            continue

        # ── V10: CHoCH Confirmation ──

        model = _classify_choch_model(choch, idx_map)
        choch.model = model  # V15: store for MSS vs SBC entry classification

        if model == "swing":
            # Swing-Based: 4 rules (1-2 already met, check 3-4)
            confirmed, is_trap = _confirm_swing_based(choch, candles, idx_map)
            if is_trap:
                # No inducement after swing-based CHoCH = Smart Money Trap
                choch.is_fake = True
            else:
                choch.confirmed = confirmed
        else:
            # Sweep-Based: 2 rules (sweep + body close below pullback)
            choch.confirmed = _confirm_sweep_based(choch, swings, candles, idx_map)

    return choch_events


def classify_mss(
    choch_events: list[CHoCH],
    liquidity_pools: list[LiquidityPool],
    candles: list[Candle],
    fvgs: list[FVG] | None = None,
    sweep_lookback: int = 30,
    expansion_mult: float = 1.5,
) -> None:
    """V15 + V25: Classify MSS and grade quality.

    V15 3-rule framework (gate for is_mss=True):
      Rule 1: Opposite-side liquidity must be taken out BEFORE the break
      Rule 2: Market must expand (break candle body > 1.5x local avg)
      Rule 3: Body close beyond key level (confirmed swing-based CHoCH)

    V25 quality grading (sets mss_grade):
      NONE:        MSS passes V15 but no FVG in shift leg (weak, ~no reliable win rate)
      STANDARD:    FVG in shift leg (~50% win rate)
      A_PLUS:      FVG + iFVG (inverted opposite FVG) (~65% win rate)
      A_PLUS_PLUS: FVG + BPR (iFVG overlaps shift leg FVG) (~75-85% win rate)

    V25 also marks inverted FVGs (is_inverted=True) on FVG objects for chart display.
    """
    if not candles:
        return
    if fvgs is None:
        fvgs = []

    # Pre-compute body sizes for expansion check
    bodies = [abs(c.close - c.open) for c in candles]
    n = len(bodies)
    idx_map = {c.index: c for c in candles}

    for ch in choch_events:
        # Only confirmed body-close CHoCH can be MSS
        if not ch.confirmed or ch.is_fake or ch.model != "swing":
            continue

        # ── V15 Rule 1: Opposite-side liquidity swept before break ──
        needed_pool_type = (
            LiquidityType.BUY_SIDE if ch.direction == Direction.BEARISH
            else LiquidityType.SELL_SIDE
        )

        sweep_candle_idx: int | None = None
        for pool in liquidity_pools:
            if not pool.swept or pool.swept_at_candle is None:
                continue
            if pool.pool_type != needed_pool_type:
                continue
            if ch.candle_index - sweep_lookback <= pool.swept_at_candle < ch.candle_index:
                sweep_candle_idx = pool.swept_at_candle
                break

        if sweep_candle_idx is None:
            continue

        # ── V15 Rule 2: Expansion ──
        if ch.candle_index >= n:
            continue
        break_body = bodies[ch.candle_index]

        local_start = max(0, ch.candle_index - 20)
        local_end = ch.candle_index
        if local_end > local_start:
            local_avg = sum(bodies[local_start:local_end]) / (local_end - local_start)
        elif n > 0:
            local_avg = sum(bodies) / n
        else:
            continue

        if local_avg == 0:
            continue

        expanded = break_body >= local_avg * expansion_mult
        if not expanded and ch.candle_index >= 1:
            expanded = bodies[ch.candle_index - 1] >= local_avg * expansion_mult
        if not expanded:
            continue

        # V15 Rule 3: Body close — already satisfied by confirmed swing-based CHoCH
        ch.is_mss = True

        # ── V25: MSS Quality Grading ──

        # Define shift leg: from sweep candle to break candle
        shift_start = sweep_candle_idx
        shift_end = ch.candle_index

        # V25 Rule: FVG = identity of MSS.  Check for FVG in the shift leg
        # direction matching the MSS (bullish MSS → bullish FVGs in shift leg)
        shift_leg_fvgs = [
            f for f in fvgs
            if f.valid
            and f.direction == ch.direction
            and shift_start <= f.candle_index <= shift_end
        ]

        if not shift_leg_fvgs:
            # No FVG in shift leg → weak MSS (V25: "if no FVG → likely fails")
            ch.mss_grade = MSSGrade.NONE
            continue

        # Has FVG in shift leg → at least STANDARD
        ch.mss_grade = MSSGrade.STANDARD

        # ── V25: Check for iFVG (Inverse Fair Value Gap) ──
        # For bullish MSS: look for bearish FVGs that existed BEFORE shift
        #   and were closed THROUGH by shift leg candles (body above upper_price)
        # For bearish MSS: look for bullish FVGs closed through (body below lower_price)
        opposite_dir = (
            Direction.BEARISH if ch.direction == Direction.BULLISH
            else Direction.BULLISH
        )

        # Prior opposite-direction FVGs that were still active when shift started
        prior_fvgs = [
            f for f in fvgs
            if f.direction == opposite_dir
            and f.candle_index < shift_start
            and f.candle_index >= shift_start - 50  # reasonable lookback
            and (not f.mitigated or (f.mitigated_at_candle is not None
                                     and f.mitigated_at_candle >= shift_start))
        ]

        inverted_fvgs: list[FVG] = []
        for prior_fvg in prior_fvgs:
            for ci in range(shift_start, shift_end + 1):
                candle = idx_map.get(ci)
                if candle is None:
                    continue

                inverted = False
                if ch.direction == Direction.BULLISH:
                    # Bullish MSS: bearish FVG inverted when body closes ABOVE
                    inverted = candle.body_top > prior_fvg.upper_price
                else:
                    # Bearish MSS: bullish FVG inverted when body closes BELOW
                    inverted = candle.body_bottom < prior_fvg.lower_price

                if inverted:
                    prior_fvg.is_inverted = True
                    prior_fvg.inverted_at_candle = ci
                    inverted_fvgs.append(prior_fvg)
                    break

        if not inverted_fvgs:
            continue  # No iFVG → stays STANDARD

        # Has iFVG → at least A+
        ch.mss_grade = MSSGrade.A_PLUS

        # ── V25: Check for BPR (Balanced Price Range) ──
        # BPR = iFVG zone and a shift leg FVG overlap at the same price level
        for ifvg in inverted_fvgs:
            for sfvg in shift_leg_fvgs:
                overlap_lower = max(ifvg.lower_price, sfvg.lower_price)
                overlap_upper = min(ifvg.upper_price, sfvg.upper_price)
                if overlap_lower < overlap_upper:
                    # BPR found — zones overlap
                    ch.mss_grade = MSSGrade.A_PLUS_PLUS
                    break
            if ch.mss_grade == MSSGrade.A_PLUS_PLUS:
                break
