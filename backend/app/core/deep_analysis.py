"""Deep Analysis — Build structured multi-TF narrative for a signal.

Traces the signal's logic through its TF hierarchy (Bias → Setup → Entry),
extracting every market structure component with specific price points and
producing rich, context-aware narratives that explain WHY the signal exists.
"""

from __future__ import annotations

from app.config import TRADING_STYLES
from app.core.engine import AnalysisResult
from app.models import (
    Candle, Direction, TrendState, SwingType, SwingClassification,
    IDMStatus, MSSGrade, EntryMethod, SignalGrade, TradingSignal,
    ZoneType,
)


def _fmt(price: float) -> str:
    """Format price as $XX,XXX.XX."""
    return f"${price:,.2f}"


def _trend_label(trend: TrendState) -> str:
    if isinstance(trend, TrendState):
        return trend.value
    return str(trend)


def _dir_label(direction) -> str:
    if isinstance(direction, Direction):
        return direction.value
    return str(direction)


def _swing_label(s) -> str:
    """HH/HL/LH/LL + high/low."""
    cls = s.classification.value if hasattr(s.classification, "value") else str(s.classification)
    stype = "high" if s.swing_type == SwingType.SWING_HIGH else "low"
    strong = " (strong)" if s.is_strong else " (weak)"
    return f"{cls} swing {stype} at {_fmt(s.price)}{strong}"


def _is_counter_trend(signal: TradingSignal, trend: TrendState) -> bool:
    """Check if signal direction contradicts the TF trend."""
    if trend == TrendState.RANGING:
        return False
    return (
        (signal.direction == Direction.BULLISH and trend == TrendState.BEARISH) or
        (signal.direction == Direction.BEARISH and trend == TrendState.BULLISH)
    )


# ── Component extraction ────────────────────────────────────────────────

def _build_level(result: AnalysisResult, role: str, signal: TradingSignal) -> dict:
    """Build a deep analysis level from an AnalysisResult."""
    components = []

    # Swings — last 3 valid ones (latest structure)
    valid_swings = [s for s in result.swings if s.is_valid_smc]
    for s in valid_swings[-3:]:
        components.append({
            "type": "swing",
            "detail": _swing_label(s),
            "price": s.price,
        })

    # BOS events — last 2 valid ones
    valid_bos = [b for b in result.bos_events if b.valid]
    for b in valid_bos[-2:]:
        d = "Bullish" if b.direction == Direction.BULLISH else "Bearish"
        tier = " (IDM body-closed)" if b.idm_body_closed else ""
        components.append({
            "type": "bos",
            "detail": f"{d} BOS broke {_fmt(b.broken_price)}{tier}",
            "price": b.broken_price,
        })

    # CHoCH events — last 2 non-fake
    non_fake_chochs = [ch for ch in result.choch_events if not ch.is_fake]
    for ch in non_fake_chochs[-2:]:
        d = "Bullish" if ch.direction == Direction.BULLISH else "Bearish"
        mss = ""
        if ch.is_mss:
            grade = ch.mss_grade.value if hasattr(ch.mss_grade, "value") else str(ch.mss_grade)
            mss = f" — MSS {grade}" if grade != "none" else " — MSS"
        conf = " (confirmed)" if ch.confirmed else " (unconfirmed)"
        model = f" [{ch.model}]" if ch.model else ""
        components.append({
            "type": "choch",
            "detail": f"{d} CHoCH broke {_fmt(ch.broken_price)}{mss}{conf}{model}",
            "price": ch.broken_price,
        })

    # Fake CHoCH — last 1 trap
    fake_chochs = [ch for ch in result.choch_events if ch.is_fake]
    for ch in fake_chochs[-1:]:
        d = "Bullish" if ch.direction == Direction.BULLISH else "Bearish"
        components.append({
            "type": "choch",
            "detail": f"FAKE {d} CHoCH at {_fmt(ch.broken_price)} — liquidity trap",
            "price": ch.broken_price,
        })

    # Inducements — last 3
    for idm in result.inducements[-3:]:
        status = idm.status.value if hasattr(idm.status, "value") else str(idm.status)
        body = "body-closed" if idm.body_closed else "wick-only"
        major = "Major" if idm.is_major else "Minor"
        components.append({
            "type": "idm",
            "detail": f"{major} IDM at {_fmt(idm.price)} — {status} ({body})",
            "price": idm.price,
        })

    # FVGs — last 3 valid
    valid_fvgs = [fvg for fvg in result.fvgs if fvg.valid]
    for fvg in valid_fvgs[-3:]:
        d = "Bullish" if fvg.direction == Direction.BULLISH else "Bearish"
        status = "mitigated" if fvg.mitigated else "active"
        extreme = " (extreme candle)" if fvg.from_extreme_candle else ""
        components.append({
            "type": "fvg",
            "detail": f"{d} FVG {_fmt(fvg.lower_price)}–{_fmt(fvg.upper_price)} [{status}]{extreme}",
            "price": (fvg.upper_price + fvg.lower_price) / 2,
        })

    # Order Blocks — last 3 valid
    valid_obs = [ob for ob in result.order_blocks if ob.valid]
    for ob in valid_obs[-3:]:
        d = "Bullish" if ob.direction == Direction.BULLISH else "Bearish"
        status = "mitigated" if ob.mitigated else "active"
        quals = []
        if ob.has_fvg:
            quals.append("FVG")
        if ob.swept_liquidity:
            quals.append("swept liq")
        q = f" ({', '.join(quals)})" if quals else ""
        components.append({
            "type": "ob",
            "detail": f"{d} OB {_fmt(ob.lower_price)}–{_fmt(ob.upper_price)} [{status}]{q}",
            "price": (ob.upper_price + ob.lower_price) / 2,
        })

    # Liquidity pools — last 3
    for lp in result.liquidity_pools[-3:]:
        ltype = lp.pool_type.value if hasattr(lp.pool_type, "value") else str(lp.pool_type)
        source = lp.source.value if hasattr(lp.source, "value") else str(lp.source)
        swept = " — SWEPT" if lp.swept else ""
        event = ""
        if lp.event_type:
            event = f" ({lp.event_type.value})"
        components.append({
            "type": "liquidity",
            "detail": f"{ltype} liquidity at {_fmt(lp.price_level)} [{source}]{swept}{event}",
            "price": lp.price_level,
        })

    # Premium/Discount
    if result.premium_discount:
        pd = result.premium_discount
        zone = pd.zone.value if hasattr(pd.zone, "value") else str(pd.zone)
        fib = f", closest fib {pd.closest_fib}" if pd.is_fib_qualified else ""
        components.append({
            "type": "pd",
            "detail": f"P/D zone: {zone} (eq {_fmt(pd.equilibrium)}, depth {pd.depth_pct:.1f}%{fib})",
            "price": pd.equilibrium,
        })

    # VSA absorptions — last 2
    for vsa in result.vsa_absorptions[-2:]:
        d = "Bullish" if vsa.direction == Direction.BULLISH else "Bearish"
        conf = "confirmed" if vsa.confirmation else "unconfirmed"
        components.append({
            "type": "vsa",
            "detail": f"{d} VSA absorption — {vsa.volume_ratio:.1f}x volume ({conf})",
            "price": None,
        })

    # Build narrative
    narrative = _build_narrative(result, role, signal)

    return {
        "tf": result.timeframe,
        "role": role,
        "trend": _trend_label(result.trend),
        "narrative": narrative,
        "components": components,
    }


# ── Narrative generation ─────────────────────────────────────────────────

def _build_narrative(result: AnalysisResult, role: str, signal: TradingSignal) -> str:
    """Build a rich, context-aware narrative for a TF level.

    Unlike a flat data dump, this tells a STORY — explaining how the structural
    evidence on this timeframe supports (or provides context for) the signal.
    """
    tf = result.timeframe
    trend = _trend_label(result.trend)
    sig_bull = signal.direction == Direction.BULLISH
    sig_word = "long" if sig_bull else "short"
    counter = _is_counter_trend(signal, result.trend)

    parts: list[str] = []

    # ── 1. Role-aware opening sentence ──
    if role == "Bias":
        if counter:
            parts.append(
                f"{tf} shows {trend} structure — but critical conditions below "
                f"enable a counter-trend {sig_word}."
            )
        else:
            parts.append(
                f"{tf} confirms {trend} bias, aligning with the {sig_word} signal direction."
            )
    elif role == "Setup":
        parts.append(
            f"{tf} is the setup timeframe where the trade conditions formed. "
            f"Structure is {trend}."
        )
    else:  # Entry
        parts.append(
            f"{tf} is the entry timeframe providing the precise trigger. "
            f"Structure is {trend}."
        )

    # ── 2. Swing structure sequence ──
    valid_swings = [s for s in result.swings if s.is_valid_smc]
    highs = [s for s in valid_swings if s.swing_type == SwingType.SWING_HIGH]
    lows = [s for s in valid_swings if s.swing_type == SwingType.SWING_LOW]
    if highs and lows:
        latest_high = max(highs, key=lambda s: s.candle_index)
        latest_low = max(lows, key=lambda s: s.candle_index)
        h_cls = latest_high.classification.value if hasattr(latest_high.classification, "value") else str(latest_high.classification)
        l_cls = latest_low.classification.value if hasattr(latest_low.classification, "value") else str(latest_low.classification)
        parts.append(
            f"Latest structure: {h_cls} high at {_fmt(latest_high.price)}, "
            f"{l_cls} low at {_fmt(latest_low.price)}."
        )

    # ── 3. BOS context — with significance ──
    valid_bos = [b for b in result.bos_events if b.valid]
    if valid_bos:
        latest_bos = max(valid_bos, key=lambda b: b.candle_index)
        d = "Bullish" if latest_bos.direction == Direction.BULLISH else "Bearish"
        tier = " with IDM body-close confirmation" if latest_bos.idm_body_closed else ""

        if counter:
            parts.append(
                f"{d} BOS broke {_fmt(latest_bos.broken_price)}{tier} — "
                f"establishing the structural displacement that created the "
                f"counter-trend opportunity."
            )
        else:
            bos_aligns = (
                (latest_bos.direction == Direction.BULLISH and sig_bull) or
                (latest_bos.direction == Direction.BEARISH and not sig_bull)
            )
            if bos_aligns:
                parts.append(
                    f"{d} BOS broke {_fmt(latest_bos.broken_price)}{tier} — "
                    f"confirming trend continuation in signal direction."
                )
            else:
                parts.append(
                    f"{d} BOS broke {_fmt(latest_bos.broken_price)}{tier}."
                )

    # ── 4. CHoCH context — critical for direction changes ──
    confirmed_chochs = [ch for ch in result.choch_events if ch.confirmed and not ch.is_fake]
    if confirmed_chochs:
        latest_ch = max(confirmed_chochs, key=lambda c: c.candle_index)
        d = "Bullish" if latest_ch.direction == Direction.BULLISH else "Bearish"
        mss_note = ""
        if latest_ch.is_mss:
            grade_val = latest_ch.mss_grade.value if hasattr(latest_ch.mss_grade, "value") else str(latest_ch.mss_grade)
            if grade_val != "none":
                mss_note = f" (MSS {grade_val})"
            else:
                mss_note = " (MSS)"

        choch_aligns = (
            (latest_ch.direction == Direction.BULLISH and sig_bull) or
            (latest_ch.direction == Direction.BEARISH and not sig_bull)
        )
        if role == "Entry" and choch_aligns:
            parts.append(
                f"{d} CHoCH broke {_fmt(latest_ch.broken_price)}{mss_note} — "
                f"this is the direction change that confirmed the entry. "
                f"Price broke above prior structure, shifting bias from "
                f"{'bearish to bullish' if sig_bull else 'bullish to bearish'}."
            )
        elif choch_aligns:
            parts.append(
                f"{d} CHoCH broke {_fmt(latest_ch.broken_price)}{mss_note} — "
                f"validating the directional thesis."
            )
        else:
            parts.append(
                f"{d} CHoCH at {_fmt(latest_ch.broken_price)}{mss_note}."
            )

    # ── 5. IDM context — with V03 body-close rule ──
    taken_idms = [i for i in result.inducements if i.status == IDMStatus.TAKEN]
    active_idms = [i for i in result.inducements if i.status == IDMStatus.ACTIVE]
    if taken_idms:
        body_closed = [i for i in taken_idms if i.body_closed]
        if body_closed:
            latest_bc = max(body_closed, key=lambda i: i.taken_at_candle or 0)
            parts.append(
                f"IDM at {_fmt(latest_bc.price)} was swept with body close — "
                f"confirming genuine liquidity sweep, not just wick manipulation "
                f"(V03 body-close rule)."
            )
        else:
            latest_taken = max(taken_idms, key=lambda i: i.taken_at_candle or 0)
            parts.append(f"IDM at {_fmt(latest_taken.price)} was swept (wick-only).")
    if active_idms:
        parts.append(f"{len(active_idms)} active IDM(s) remain untaken above/below.")

    # ── 6. Premium/Discount — institutional context ──
    if result.premium_discount:
        pd = result.premium_discount
        zone = pd.zone.value if hasattr(pd.zone, "value") else str(pd.zone)

        in_favorable_zone = (
            (sig_bull and zone == "discount") or
            (not sig_bull and zone == "premium")
        )
        if in_favorable_zone:
            action = "accumulation (buying)" if sig_bull else "distribution (selling)"
            parts.append(
                f"Price in {zone} zone (eq {_fmt(pd.equilibrium)}, depth {pd.depth_pct:.0f}%) — "
                f"institutional {action} territory. Smart money typically "
                f"{'buys' if sig_bull else 'sells'} at these levels."
            )
        elif zone == "equilibrium":
            parts.append(
                f"Price near equilibrium at {_fmt(pd.equilibrium)} — neutral zone."
            )
        else:
            parts.append(
                f"Price in {zone} zone (eq {_fmt(pd.equilibrium)}, depth {pd.depth_pct:.0f}%)."
            )

    # ── 7. Active zones count ──
    active_obs = [ob for ob in result.order_blocks if ob.valid and not ob.mitigated]
    active_fvgs = [fvg for fvg in result.fvgs if fvg.valid and not fvg.mitigated]
    if active_obs or active_fvgs:
        parts.append(f"{len(active_obs)} active OB(s), {len(active_fvgs)} active FVG(s).")

    # ── 8. Role-specific closing takeaway ──
    if role == "Bias" and counter:
        parts.append(
            f"Despite {trend} trend, deep price displacement and swept liquidity "
            f"create a counter-trend {'accumulation' if sig_bull else 'distribution'} "
            f"scenario where smart money may {'buy' if sig_bull else 'sell'} at value."
        )
    elif role == "Setup":
        if counter:
            parts.append(
                f"The setup formed from the counter-trend conditions above — "
                f"structural displacement created the entry zone opportunity."
            )
        else:
            parts.append(
                f"Setup conditions aligned: trend-following entry zone formed "
                f"with structural confirmation."
            )
    elif role == "Entry":
        parts.append(
            f"Entry was triggered in this timeframe with structural confirmation."
        )

    return " ".join(parts)


# ── Entry zone ───────────────────────────────────────────────────────────

def _find_entry_zone(signal: TradingSignal, entry_result: AnalysisResult) -> dict:
    """Identify which OB/FVG zone the signal's entry price falls within."""
    entry = signal.entry_price
    sig_bull = signal.direction == Direction.BULLISH
    best_zone = None
    best_dist = float("inf")

    # Check OBs
    for ob in entry_result.order_blocks:
        if not ob.valid:
            continue
        mid = (ob.upper_price + ob.lower_price) / 2
        dist = abs(mid - entry)
        if dist < best_dist:
            best_dist = dist
            d = "Bullish" if ob.direction == Direction.BULLISH else "Bearish"
            quals = []
            if ob.has_fvg:
                quals.append("FVG-validated")
            if ob.swept_liquidity:
                quals.append("liquidity swept")
            q = f" ({', '.join(quals)})" if quals else ""
            best_zone = {
                "type": f"{d} OB",
                "upper": ob.upper_price,
                "lower": ob.lower_price,
                "method": "",
                "narrative": (
                    f"Entry at {d} Order Block {_fmt(ob.lower_price)}–{_fmt(ob.upper_price)}{q}. "
                    f"Entry price {_fmt(entry)} falls within this zone. "
                    f"OBs represent the last opposing candle before a displacement move — "
                    f"institutional {'buy' if sig_bull else 'sell'} orders are concentrated here."
                ),
            }

    # Check FVGs
    for fvg in entry_result.fvgs:
        if not fvg.valid:
            continue
        mid = (fvg.upper_price + fvg.lower_price) / 2
        dist = abs(mid - entry)
        if dist < best_dist:
            best_dist = dist
            d = "Bullish" if fvg.direction == Direction.BULLISH else "Bearish"
            mitigated = " (already mitigated)" if fvg.mitigated else " (unmitigated — fresh zone)"
            extreme = " Created by extreme candle displacement." if fvg.from_extreme_candle else ""
            best_zone = {
                "type": f"{d} FVG",
                "upper": fvg.upper_price,
                "lower": fvg.lower_price,
                "method": "",
                "narrative": (
                    f"Entry at {d} Fair Value Gap {_fmt(fvg.lower_price)}–{_fmt(fvg.upper_price)}"
                    f"{mitigated}. "
                    f"Entry price {_fmt(entry)} targets the gap midpoint. "
                    f"FVGs represent imbalanced price action where institutions left "
                    f"unfilled orders — price tends to return to fill these gaps.{extreme}"
                ),
            }

    if best_zone is None:
        best_zone = {
            "type": "Zone",
            "upper": entry * 1.001,
            "lower": entry * 0.999,
            "method": "",
            "narrative": f"Entry at {_fmt(entry)}.",
        }

    # Add entry method with explanation
    if signal.entry_method:
        method = signal.entry_method.value if hasattr(signal.entry_method, "value") else str(signal.entry_method)
        best_zone["method"] = method
        method_explanations = {
            "pullback_break": "Price pulled back into the zone and then broke structure, confirming directional intent.",
            "zone_tap": "Price tapped into the zone for a direct entry.",
            "mss_entry": "Market Structure Shift confirmed the entry — direction changed within the zone.",
            "sbc_entry": "Sweep-Based Change of Character: liquidity was swept before entry confirmation.",
            "choch_entry": "Change of Character within the zone confirmed the reversal.",
        }
        explanation = method_explanations.get(method, f"Entry confirmed via {method} method.")
        best_zone["narrative"] += f" {explanation}"
    else:
        best_zone["method"] = "zone_tap"

    # SL placement context
    risk = abs(entry - signal.stop_loss)
    risk_pct = (risk / entry) * 100 if entry else 0
    best_zone["narrative"] += (
        f" Stop loss at {_fmt(signal.stop_loss)} — placed "
        f"{'below' if sig_bull else 'above'} the zone boundary "
        f"(risk: {_fmt(risk)}, {risk_pct:.2f}%)."
    )

    return best_zone


# ── TP logic ─────────────────────────────────────────────────────────────

def _build_tp_logic(
    signal: TradingSignal,
    results_by_tf: dict[str, AnalysisResult],
    tf_chain: list[str],
) -> list[dict]:
    """Explain what each TP target is aiming for, with context."""
    sig_bull = signal.direction == Direction.BULLISH
    tps = signal.take_profits if signal.take_profits else []
    if not tps:
        risk = abs(signal.entry_price - signal.stop_loss)
        rr = round(abs(signal.take_profit - signal.entry_price) / risk, 2) if risk > 0 else 0
        tps = [{"price": signal.take_profit, "rr": rr, "label": "TP1"}]

    result = []
    for tp_entry in tps:
        price = tp_entry["price"]
        rr = tp_entry["rr"]
        label = tp_entry["label"]

        # Search all TFs in the chain for a matching swing
        target_desc = "swing target"
        best_match = None
        best_pct = 0.01  # 1% max tolerance

        for tf in tf_chain:
            r = results_by_tf.get(tf)
            if not r:
                continue
            for s in r.swings:
                if not s.is_valid_smc:
                    continue
                pct = abs(s.price - price) / price if price else 1
                if pct < best_pct:
                    best_pct = pct
                    cls = s.classification.value if hasattr(s.classification, "value") else str(s.classification)
                    stype = "high" if s.swing_type == SwingType.SWING_HIGH else "low"
                    strong = "strong" if s.is_strong else "weak"

                    # Explain WHY this swing is the target
                    if sig_bull and stype == "high":
                        reason = "nearest opposing swing high above entry"
                    elif not sig_bull and stype == "low":
                        reason = "nearest opposing swing low below entry"
                    elif sig_bull and stype == "low":
                        reason = "structural support level"
                    else:
                        reason = "structural resistance level"

                    best_match = (
                        f"{tf} {cls} swing {stype} at {_fmt(s.price)} "
                        f"({strong}) — {reason}"
                    )

        if best_match:
            target_desc = best_match

        result.append({
            "label": label,
            "price": price,
            "target": target_desc,
            "rr": rr,
        })

    return result


# ── Confluences ──────────────────────────────────────────────────────────

def _build_confluences(
    signal: TradingSignal,
    levels: list[dict],
    entry_zone: dict,
) -> list[dict]:
    """Build detailed confluence table with specific evidence from components."""
    result = []

    # Pre-extract relevant evidence from level components
    entry_fvg_detail = None
    entry_ob_detail = None
    pd_detail = None
    bos_detail = None
    choch_detail = None
    idm_detail = None

    entry_mid = (entry_zone["upper"] + entry_zone["lower"]) / 2

    for level in levels:
        for comp in level["components"]:
            # Find FVG nearest to entry zone
            if comp["type"] == "fvg" and comp["price"]:
                if abs(comp["price"] - entry_mid) / entry_mid < 0.02:
                    entry_fvg_detail = f"{level['tf']}: {comp['detail']}"

            # Find OB nearest to entry zone
            if comp["type"] == "ob" and comp["price"]:
                if abs(comp["price"] - entry_mid) / entry_mid < 0.05:
                    entry_ob_detail = f"{level['tf']}: {comp['detail']}"

            # Capture P/D detail
            if comp["type"] == "pd":
                pd_detail = f"{level['tf']}: {comp['detail']}"

            # Capture latest BOS
            if comp["type"] == "bos":
                bos_detail = f"{level['tf']}: {comp['detail']}"

            # Capture CHoCH aligned with signal
            if comp["type"] == "choch":
                sig_dir = "Bullish" if signal.direction == Direction.BULLISH else "Bearish"
                if sig_dir in comp["detail"] and "FAKE" not in comp["detail"]:
                    choch_detail = f"{level['tf']}: {comp['detail']}"

            # Capture body-closed IDM
            if comp["type"] == "idm" and "body-closed" in comp["detail"]:
                idm_detail = f"{level['tf']}: {comp['detail']}"

    # Build evidence-backed confluence entries
    evidence_map = {
        "FVG": (
            "strong",
            entry_fvg_detail or "Fair Value Gap present in/near entry zone",
        ),
        "Order Block": (
            "strong",
            entry_ob_detail or "Entry zone is a valid Order Block",
        ),
        "Discount zone (buy)": (
            "strong",
            pd_detail or "Price in discount zone — institutional buy area",
        ),
        "Premium zone (sell)": (
            "strong",
            pd_detail or "Price in premium zone — institutional sell area",
        ),
        "BOS confirmed": (
            "medium",
            bos_detail or "Break of Structure validates trend continuation",
        ),
        "LTF CHoCH sync": (
            "strong",
            choch_detail or "Lower timeframe CHoCH confirms HTF direction",
        ),
        "Multi-TF aligned": (
            "strong",
            "Multiple timeframes confirm same direction — high-probability setup",
        ),
        "Kill zone active": (
            "medium",
            "Trading during high-volatility session (London/NY kill zone)",
        ),
        "MSS A++ (BPR)": (
            "strong",
            "Market Structure Shift with Balanced Price Range — highest quality entry confirmation",
        ),
        "MSS A+ (iFVG)": (
            "strong",
            "Market Structure Shift with inverted FVG — high quality entry confirmation",
        ),
        "MSS Standard": (
            "medium",
            choch_detail or "Market Structure Shift confirmed — direction change validated",
        ),
        "VSA Absorption": (
            "strong",
            "Ultra-high volume institutional flow detected — smart money is actively trading this level",
        ),
        "SBC sweep entry": (
            "medium",
            "Sweep-Based Change of Character — liquidity was swept before directional confirmation",
        ),
        "HTF trend aligned": (
            "strong",
            "Signal direction matches higher timeframe trend — trading with the smart money flow",
        ),
        "Counter-trend IDM body close": (
            "medium",
            idm_detail or "Counter-trend IDM swept with body close — genuine liquidity sweep confirmed (V03 rule)",
        ),
        "Counter-trend FVG": (
            "medium",
            entry_fvg_detail or "Counter-trend FVG created during the reversal bounce — supports the counter-trend thesis",
        ),
    }

    for conf_name in signal.confluences:
        strength, detail = evidence_map.get(conf_name, ("medium", conf_name))
        result.append({
            "name": conf_name,
            "detail": detail,
            "strength": strength,
        })

    if not result:
        result.append({
            "name": "Minimal",
            "detail": "No significant confluences detected",
            "strength": "weak",
        })

    return result


# ── Main builder ─────────────────────────────────────────────────────────

def build_deep_analysis(
    signal: TradingSignal,
    signal_id: str,
    results_by_tf: dict[str, AnalysisResult],
    candles_by_tf: dict[str, list[Candle]],
) -> dict:
    """Build a complete deep analysis for a signal.

    Produces a structured, narrative-rich analysis that traces the signal's
    logic from Bias TF → Setup TF → Entry TF, explaining WHY each component
    matters for the trade thesis.
    """
    # Determine TF chain from trading style
    style_key = signal.trading_style or "intraday"
    if isinstance(style_key, str) and style_key in TRADING_STYLES:
        style_cfg = TRADING_STYLES[style_key]
    else:
        style_cfg = TRADING_STYLES.get("intraday", {
            "bias_tf": "H4", "setup_tf": "H1", "entry_tf": "M15", "label": "Intraday"
        })

    bias_tf = style_cfg["bias_tf"]
    setup_tf = style_cfg["setup_tf"]
    entry_tf = style_cfg["entry_tf"]
    style_label = style_cfg.get("label", style_key)

    # Build levels (pass signal for context-aware narratives)
    levels = []

    bias_result = results_by_tf.get(bias_tf)
    if bias_result:
        levels.append(_build_level(bias_result, "Bias", signal))

    setup_result = results_by_tf.get(setup_tf)
    if setup_result:
        levels.append(_build_level(setup_result, "Setup", signal))

    entry_result = results_by_tf.get(entry_tf)
    if entry_result:
        levels.append(_build_level(entry_result, "Entry", signal))

    # Signal summary
    direction = signal.direction.value if isinstance(signal.direction, Direction) else str(signal.direction)
    grade = signal.grade.value if isinstance(signal.grade, SignalGrade) else str(signal.grade)
    tps = signal.take_profits if signal.take_profits else []
    if not tps:
        risk = abs(signal.entry_price - signal.stop_loss)
        rr = round(abs(signal.take_profit - signal.entry_price) / risk, 2) if risk > 0 else 0
        tps = [{"price": signal.take_profit, "rr": rr, "label": "TP1"}]

    signal_summary = {
        "direction": direction,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profits": tps,
        "grade": grade,
        "style": style_label,
    }

    # Entry zone
    entry_zone = _find_entry_zone(signal, entry_result) if entry_result else {
        "type": "Unknown",
        "upper": signal.entry_price,
        "lower": signal.entry_price,
        "method": "unknown",
        "narrative": f"Entry at {_fmt(signal.entry_price)}.",
    }

    # TP logic
    tf_chain = [bias_tf, setup_tf, entry_tf]
    tp_logic = _build_tp_logic(signal, results_by_tf, tf_chain)

    # Confluences (now evidence-backed)
    confluences = _build_confluences(signal, levels, entry_zone)

    # Timing info
    timing = {
        "activated_at": signal.created_at,
        "trigger_timestamp": signal.timestamp,
        "trigger_candle_index": signal.trigger_candle_index,
        "bars_active": signal.bars_active,
        "entry_timeframe": signal.timeframe,
    }

    return {
        "signal_id": signal_id,
        "signal_summary": signal_summary,
        "levels": levels,
        "entry_zone": entry_zone,
        "tp_logic": tp_logic,
        "confluences": confluences,
        "timing": timing,
    }
