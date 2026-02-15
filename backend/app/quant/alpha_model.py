"""Quant Alpha Model (Layer 1) — Directional scoring from all data sources.

Weight: 55% of combined quant score.
Scores each signal direction using 11 components across TIER 1/2/3.
"""

from __future__ import annotations

from app.models import Direction, VolatilityRegime, PricePhase


def compute_alpha_score(
    quant_data: dict,
    futures_data: dict,
    signal_direction: Direction,
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL,
    price_cycle_phase: PricePhase | None = None,
) -> tuple[float, dict]:
    """Compute alpha model score for a signal direction.

    Components (each -10 to +10, weighted):
    TIER 1:
      1. OI-Price Divergence (weight 3)
      2. Funding Rate (weight 2)
      3. VPIN signal (weight 2)
      4. Liquidation cascade (weight 2)
      5. Volatility Regime (weight 1.5)
      6. Price Cycle Phase (weight 1)
    TIER 2 (when available):
      7. Options positioning (weight 1.5)
      8. Funding arbitrage (weight 1)
      9. COT net positioning (weight 1)
      10. On-chain flow (weight 1)
    TIER 3 (when available):
      11. Fear & Greed (weight 0.5)

    Returns (score: -100..+100, component_details: dict).
    """
    is_bull = (
        signal_direction == Direction.BULLISH
        if isinstance(signal_direction, Direction)
        else signal_direction == "bullish"
    )

    components = {}
    weighted_sum = 0.0
    total_weight = 0.0

    # ── TIER 1: Always available ──

    # 1. OI-Price Divergence
    oi_score = _score_oi_price_divergence(futures_data, is_bull)
    components["oi_divergence"] = {"score": round(oi_score, 1), "weight": 3}
    weighted_sum += oi_score * 3
    total_weight += 3

    # 2. Funding Rate
    funding_score = _score_funding_rate(futures_data, is_bull)
    components["funding_rate"] = {"score": round(funding_score, 1), "weight": 2}
    weighted_sum += funding_score * 2
    total_weight += 2

    # 3. VPIN signal
    vpin = quant_data.get("vpin", 0.0)
    taker_dir = quant_data.get("taker_direction", "neutral")
    vpin_score = _score_vpin(vpin, taker_dir, is_bull)
    components["vpin"] = {"score": round(vpin_score, 1), "weight": 2, "value": round(vpin, 3)}
    weighted_sum += vpin_score * 2
    total_weight += 2

    # 4. Liquidation cascade
    cascade = quant_data.get("liquidation_cascade")
    liq_score = _score_liquidation(cascade, is_bull)
    components["liquidation"] = {"score": round(liq_score, 1), "weight": 2}
    weighted_sum += liq_score * 2
    total_weight += 2

    # 5. Volatility Regime
    vol_score = _score_volatility_regime(volatility_regime, is_bull)
    components["volatility_regime"] = {"score": round(vol_score, 1), "weight": 1.5, "regime": volatility_regime.value}
    weighted_sum += vol_score * 1.5
    total_weight += 1.5

    # 6. Price Cycle Phase
    phase_score = _score_price_cycle(price_cycle_phase, is_bull)
    components["price_cycle"] = {"score": round(phase_score, 1), "weight": 1, "phase": price_cycle_phase.value if price_cycle_phase else "unknown"}
    weighted_sum += phase_score * 1
    total_weight += 1

    # ── TIER 2: Optional ──

    # 7. Options positioning (Deribit)
    options = quant_data.get("options_data", {})
    if options and options.get("pc_ratio") is not None:
        opt_score = _score_options(options, is_bull)
        components["options"] = {"score": round(opt_score, 1), "weight": 1.5}
        weighted_sum += opt_score * 1.5
        total_weight += 1.5

    # 8. Funding arbitrage (cross-exchange)
    cross_funding = quant_data.get("cross_exchange_funding", {})
    if cross_funding and cross_funding.get("dispersion", 0) > 0:
        arb_score = _score_funding_arbitrage(cross_funding, is_bull)
        components["funding_arb"] = {"score": round(arb_score, 1), "weight": 1}
        weighted_sum += arb_score * 1
        total_weight += 1

    # 9. COT net positioning
    cot = quant_data.get("cot_data", {})
    if cot and cot.get("leveraged_net") is not None:
        cot_score = _score_cot(cot, is_bull)
        components["cot"] = {"score": round(cot_score, 1), "weight": 1}
        weighted_sum += cot_score * 1
        total_weight += 1

    # 10. On-chain flow
    onchain = quant_data.get("onchain_flow", {})
    if onchain and onchain.get("net_flow") is not None:
        chain_score = _score_onchain(onchain, is_bull)
        components["onchain"] = {"score": round(chain_score, 1), "weight": 1}
        weighted_sum += chain_score * 1
        total_weight += 1

    # ── TIER 3: Nice to have ──

    # 11. Fear & Greed (contrarian)
    fng = quant_data.get("fear_greed", {})
    if fng and fng.get("value") is not None:
        fng_score = _score_fear_greed(fng, is_bull)
        components["fear_greed"] = {"score": round(fng_score, 1), "weight": 0.5}
        weighted_sum += fng_score * 0.5
        total_weight += 0.5

    # Normalize to -100..+100
    if total_weight > 0:
        raw = (weighted_sum / total_weight) * 10
    else:
        raw = 0.0

    score = max(-100.0, min(100.0, raw))
    return score, components


# ── Component Scoring Functions ──

def _score_oi_price_divergence(futures_data: dict, is_bull: bool) -> float:
    """OI rising + price falling = accumulation (bullish divergence)."""
    oi_hist = futures_data.get("open_interest", {}).get("history", [])
    if len(oi_hist) < 12:
        return 0.0

    recent_oi = oi_hist[-1].get("oi_usd", 0)
    earlier_oi = oi_hist[-12].get("oi_usd", 0)
    if earlier_oi <= 0:
        return 0.0

    oi_change = (recent_oi - earlier_oi) / earlier_oi * 100

    # Rising OI is generally conviction
    if oi_change > 3.0:
        return 6.0 if is_bull else 4.0  # More bullish bias (accumulation)
    elif oi_change > 1.0:
        return 3.0
    elif oi_change < -3.0:
        return -5.0  # Unwinding = weak conviction
    elif oi_change < -1.0:
        return -2.0
    return 0.0


def _score_funding_rate(futures_data: dict, is_bull: bool) -> float:
    """Funding rate: crowded positioning = contrarian signal."""
    rate = futures_data.get("funding_rate", {}).get("current", 0)
    if abs(rate) < 0.0001:
        return 0.0

    # Positive funding = longs pay shorts = longs crowded
    if is_bull:
        if rate < -0.0003:
            return 7.0   # Shorts crowded = bullish fuel
        elif rate < 0:
            return 3.0
        elif rate > 0.0005:
            return -6.0  # Longs very crowded = contrarian bearish
        elif rate > 0.0001:
            return -2.0
    else:
        if rate > 0.0003:
            return 7.0   # Longs crowded = bearish fuel
        elif rate > 0:
            return 3.0
        elif rate < -0.0005:
            return -6.0  # Shorts very crowded = contrarian bullish
        elif rate < -0.0001:
            return -2.0

    return 0.0


def _score_vpin(vpin: float, taker_dir: str, is_bull: bool) -> float:
    """VPIN: high = informed flow, direction from taker imbalance."""
    if vpin < 0.3:
        return 0.0  # Low informed flow

    if vpin >= 0.7:
        # High informed flow
        if (is_bull and taker_dir == "buy") or (not is_bull and taker_dir == "sell"):
            return 8.0  # Smart money aligned
        elif taker_dir == "neutral":
            return 2.0
        else:
            return -6.0  # Smart money opposing
    elif vpin >= 0.5:
        if (is_bull and taker_dir == "buy") or (not is_bull and taker_dir == "sell"):
            return 4.0
        elif taker_dir != "neutral":
            return -3.0
        return 1.0

    return 0.0


def _score_liquidation(cascade: dict | None, is_bull: bool) -> float:
    """Liquidation cascade: opposite-side exhaustion = fuel for signal."""
    if not cascade or not cascade.get("detected"):
        return 0.0

    dominant = cascade.get("dominant_side", "")
    total = cascade.get("total_usd", 0)

    # Cascade scale factor
    magnitude = 1.0
    if total > 200_000_000:
        magnitude = 2.0  # Massive cascade
    elif total > 100_000_000:
        magnitude = 1.5

    if is_bull and dominant == "sell":
        return min(10.0, 5.0 * magnitude)   # Longs liquidated = bottom near
    elif not is_bull and dominant == "buy":
        return min(10.0, 5.0 * magnitude)   # Shorts liquidated = top near
    elif is_bull and dominant == "buy":
        return max(-10.0, -4.0 * magnitude)  # Our side cascading
    elif not is_bull and dominant == "sell":
        return max(-10.0, -4.0 * magnitude)

    return 0.0


def _score_volatility_regime(regime: VolatilityRegime, is_bull: bool) -> float:
    """Volatility regime: low vol = reliable signals, extreme = unreliable."""
    scores = {
        VolatilityRegime.LOW: 5.0,       # Low vol = clear signals
        VolatilityRegime.NORMAL: 3.0,    # Normal conditions
        VolatilityRegime.HIGH: -3.0,     # Elevated risk
        VolatilityRegime.EXTREME: -8.0,  # Suppress signals
    }
    return scores.get(regime, 0.0)


def _score_price_cycle(phase: PricePhase | None, is_bull: bool) -> float:
    """Price cycle phase from existing price_cycle_detector.py."""
    if phase is None:
        return 0.0

    if phase == PricePhase.EXPANSION:
        return 5.0   # Momentum — trend continuation likely
    elif phase == PricePhase.RETRACEMENT:
        return 3.0   # Pullback — entry opportunity
    elif phase == PricePhase.CONSOLIDATION:
        return 0.0   # Ranging — unclear
    elif phase == PricePhase.REVERSAL:
        return -3.0  # Potential trend change

    return 0.0


def _score_options(options: dict, is_bull: bool) -> float:
    """Deribit options: P/C ratio, max pain magnet, GEX, 25d skew."""
    score = 0.0
    pc_ratio = options.get("pc_ratio", 1.0)
    max_pain_dist = options.get("max_pain_distance_pct", 0)
    net_gex = options.get("net_gex", 0)
    skew_25d = options.get("skew_25d")

    # P/C ratio: > 1.3 = very bearish sentiment, < 0.7 = very bullish
    if is_bull:
        if pc_ratio > 1.3:
            score += 4.0  # Contrarian bullish (extreme put buying)
        elif pc_ratio < 0.7:
            score -= 2.0  # Too bullish = complacent
    else:
        if pc_ratio < 0.7:
            score += 4.0  # Contrarian bearish (extreme call buying)
        elif pc_ratio > 1.3:
            score -= 2.0

    # Max pain proximity: price gravitates toward max pain near expiry
    if abs(max_pain_dist) < 2.0:
        score += 1.5  # Near max pain = stable, predictable
    elif max_pain_dist > 5.0:
        # Price above max pain — pull lower
        score += 1.0 if not is_bull else -0.5
    elif max_pain_dist < -5.0:
        # Price below max pain — pull higher
        score += 1.0 if is_bull else -0.5

    # GEX: Positive = dealers dampen (mean-reverting market), Negative = amplify (trending)
    if net_gex > 0:
        # Positive GEX = dampening → range-bound conditions
        score += 1.0  # Slightly favorable for well-placed signals
    elif net_gex < -100:
        # Strong negative GEX = dealers amplify moves
        score += 1.5  # Trend-following signals get a boost
    elif net_gex < 0:
        score += 0.5

    # 25-delta skew: put_IV - call_IV
    if skew_25d is not None:
        if skew_25d > 5.0:
            # Puts expensive = fear/hedging demand = bearish sentiment
            score += 2.0 if not is_bull else -1.5  # Contrarian or confirm short
        elif skew_25d > 2.0:
            score += 0.5 if not is_bull else -0.5
        elif skew_25d < -5.0:
            # Calls expensive = bullish positioning
            score += 2.0 if is_bull else -1.5
        elif skew_25d < -2.0:
            score += 0.5 if is_bull else -0.5

    return max(-10.0, min(10.0, score))


def _score_funding_arbitrage(cross_funding: dict, is_bull: bool) -> float:
    """Cross-exchange funding divergence: arbitrage pressure building."""
    dispersion = cross_funding.get("dispersion", 0)
    avg_rate = cross_funding.get("avg_rate", 0)

    if dispersion < 0.0005:
        return 0.0  # No meaningful divergence

    # High dispersion = one exchange out of line = convergence coming
    if is_bull and avg_rate < 0:
        return 5.0  # Average negative funding = shorts crowded = bullish
    elif not is_bull and avg_rate > 0:
        return 5.0  # Average positive funding = longs crowded = bearish
    elif is_bull and avg_rate > 0.0003:
        return -4.0  # Longs crowded
    elif not is_bull and avg_rate < -0.0003:
        return -4.0  # Shorts crowded

    return 2.0  # Divergence exists, slight positive


def _score_cot(cot: dict, is_bull: bool) -> float:
    """CME COT: institutional positioning from CFTC (weekly)."""
    leveraged_net = cot.get("leveraged_net", 0)
    percentile = cot.get("percentile", 50)

    # Extreme readings (>80th or <20th percentile) = contrarian
    if percentile > 80:
        # Extremely long — contrarian bearish
        return -6.0 if is_bull else 6.0
    elif percentile < 20:
        # Extremely short — contrarian bullish
        return 6.0 if is_bull else -6.0
    elif percentile > 60:
        return -2.0 if is_bull else 2.0
    elif percentile < 40:
        return 2.0 if is_bull else -2.0

    return 0.0


def _score_onchain(onchain: dict, is_bull: bool) -> float:
    """On-chain exchange flows: inflow = selling, outflow = accumulation."""
    net_flow = onchain.get("net_flow", 0)  # Positive = net inflow to exchanges

    if net_flow > 5000:  # Large net inflow (BTC units)
        return -6.0 if is_bull else 5.0   # Selling pressure
    elif net_flow > 1000:
        return -3.0 if is_bull else 3.0
    elif net_flow < -5000:  # Large net outflow
        return 6.0 if is_bull else -5.0   # Accumulation
    elif net_flow < -1000:
        return 3.0 if is_bull else -3.0

    return 0.0


def _score_fear_greed(fng: dict, is_bull: bool) -> float:
    """Fear & Greed Index: contrarian at extremes."""
    value = fng.get("value", 50)

    if value < 20:  # Extreme Fear
        return 7.0 if is_bull else -4.0   # Contrarian buy
    elif value < 35:  # Fear
        return 3.0 if is_bull else -1.0
    elif value > 80:  # Extreme Greed
        return -4.0 if is_bull else 7.0   # Contrarian sell
    elif value > 65:  # Greed
        return -1.0 if is_bull else 3.0

    return 0.0
