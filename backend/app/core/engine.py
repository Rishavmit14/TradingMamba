"""SMC Detection Engine — Orchestrates all detectors in sequence.

This is the single entry point that takes OHLCV candles and returns
detected patterns + trading signals. Each detector builds on the previous.

Pipeline:
  Candles → Swings → IDM → Liquidity → BOS → CHoCH → FVG → MSS(V25) → OB → P/D → Session → Signals
"""

from __future__ import annotations
from dataclasses import dataclass, field
from app.models import (
    Candle, SwingPoint, Inducement, LiquidityPool,
    BOS, CHoCH, FVG, OrderBlock, PremiumDiscount,
    Session, TradingSignal, TrendState,
)
from app.core.swing_detector import detect_and_classify
from app.core.inducement import detect_and_validate as detect_idm
from app.core.liquidity import detect_all_liquidity, prices_equal
from app.core.bos_detector import detect_bos
from app.core.choch_detector import detect_choch, filter_fake_choch, classify_mss
from app.core.vsa_detector import detect_vsa_absorptions
from app.core.fvg_detector import detect_all_fvgs
from app.core.order_block import detect_all_order_blocks
from app.core.premium_discount import calculate_premium_discount
from app.core.session import get_current_session
from app.core.amd_detector import detect_amd_patterns
from app.core.price_cycle_detector import detect_price_cycles
from app.core.signal_generator import generate_signals
from app.config import SWING_LOOKBACK
from app.models import AMDPattern, PriceCycleEvent, PricePhase


@dataclass
class AnalysisResult:
    """Complete analysis output for a single timeframe."""
    timeframe: str
    trend: TrendState
    swings: list[SwingPoint] = field(default_factory=list)
    inducements: list[Inducement] = field(default_factory=list)
    liquidity_pools: list[LiquidityPool] = field(default_factory=list)
    bos_events: list[BOS] = field(default_factory=list)
    choch_events: list[CHoCH] = field(default_factory=list)
    fvgs: list[FVG] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    premium_discount: PremiumDiscount | None = None
    session: Session | None = None
    vsa_active: bool = False
    vsa_absorptions: list = field(default_factory=list)
    amd_patterns: list[AMDPattern] = field(default_factory=list)
    price_cycles: list[PriceCycleEvent] = field(default_factory=list)
    current_phase: PricePhase = PricePhase.CONSOLIDATION
    signals: list[TradingSignal] = field(default_factory=list)


def analyze_timeframe(candles: list[Candle], timeframe: str) -> AnalysisResult:
    """Run the complete detection pipeline on a single timeframe.

    This is the core analysis function. It runs each detector in sequence,
    feeding results forward.
    """
    if not candles:
        return AnalysisResult(timeframe=timeframe, trend=TrendState.RANGING)

    # Ensure candles have sequential indices
    for i, c in enumerate(candles):
        c.index = i

    lookback = SWING_LOOKBACK.get(timeframe, 5)

    # 1.1: Swing detection + classification
    swings, trend = detect_and_classify(candles, lookback)

    # 1.2: Inducement detection + swing validation
    inducements, swings = detect_idm(candles, swings)

    # 1.3: Liquidity pool identification
    liquidity_pools = detect_all_liquidity(candles, swings, inducements)

    # 1.4: Break of Structure detection
    bos_events = detect_bos(candles, swings, inducements)

    # 1.5a: VSA Ultra High Volume absorption detection (V22)
    vsa_absorptions = detect_vsa_absorptions(candles)

    # 1.5: Change of Character detection (with VSA confluence)
    choch_events = detect_choch(candles, swings, bos_events, trend, vsa_absorptions=vsa_absorptions)
    choch_events = filter_fake_choch(choch_events, swings, candles,
                                     inducements, liquidity_pools, bos_events)

    # 1.6: Fair Value Gap detection (moved before MSS — V25 grading needs FVGs)
    fvgs = detect_all_fvgs(candles, swings, trend)

    # 1.5b: MSS classification + V25 quality grading (Standard/A+/A++)
    classify_mss(choch_events, liquidity_pools, candles, fvgs)

    # 1.7: Order Block detection
    order_blocks = detect_all_order_blocks(candles, swings, fvgs, inducements, trend)

    # 1.8: Premium/Discount
    current_price = candles[-1].close
    pd = calculate_premium_discount(swings, current_price)

    # 1.9: Session
    session = get_current_session(candles[-1].timestamp)

    # 1.10: AMD pattern detection (V17)
    amd_patterns = detect_amd_patterns(candles, swings, inducements, trend)

    # 1.11: Price Delivery Cycle (V11)
    price_cycles, current_phase = detect_price_cycles(
        candles, bos_events, choch_events, fvgs, swings
    )

    return AnalysisResult(
        timeframe=timeframe,
        trend=trend,
        swings=swings,
        inducements=inducements,
        liquidity_pools=liquidity_pools,
        bos_events=bos_events,
        choch_events=choch_events,
        fvgs=fvgs,
        order_blocks=order_blocks,
        premium_discount=pd,
        session=session,
        vsa_active=len(vsa_absorptions) > 0,
        vsa_absorptions=vsa_absorptions,
        amd_patterns=amd_patterns,
        price_cycles=price_cycles,
        current_phase=current_phase,
    )


def _apply_cross_tf_fake_choch(results: dict[str, AnalysisResult]) -> None:
    """V09 Rule 4: Mark lower TF CHoCH as fake if the broken price matches
    a higher TF active inducement level.

    Hierarchy: W1 > D1 > H4 > M15
    """
    tf_hierarchy = [("M15", ["H4", "D1", "W1"]),
                    ("H4", ["D1", "W1"])]

    for lower_tf, higher_tfs in tf_hierarchy:
        lower = results.get(lower_tf)
        if not lower:
            continue

        # Collect active inducement prices from all higher TFs
        higher_idm_prices: list[float] = []
        for htf in higher_tfs:
            higher = results.get(htf)
            if not higher:
                continue
            for idm in higher.inducements:
                higher_idm_prices.append(idm.price)

        if not higher_idm_prices:
            continue

        # Check each lower TF CHoCH against higher TF inducements
        for choch in lower.choch_events:
            if choch.is_fake:
                continue
            for htf_price in higher_idm_prices:
                if prices_equal(choch.broken_price, htf_price):
                    choch.is_fake = True
                    break


def run_multi_tf_analysis(
    candles_by_tf: dict[str, list[Candle]],
) -> dict[str, AnalysisResult]:
    """Run analysis across all timeframes (W1→D1→H4→M15).

    Higher TF results inform lower TF signal generation.
    """
    results: dict[str, AnalysisResult] = {}

    # Analyze each timeframe
    for tf in ["W1", "D1", "H4", "M15"]:
        candles = candles_by_tf.get(tf, [])
        if candles:
            results[tf] = analyze_timeframe(candles, tf)
        else:
            results[tf] = AnalysisResult(timeframe=tf, trend=TrendState.RANGING)

    # V09 Rule 4: Cross-TF fake CHoCH detection
    # Lower TF swing that is higher TF inducement = fake CHoCH on lower TF
    _apply_cross_tf_fake_choch(results)

    # Generate signals on M15 using higher TF context
    m15 = results.get("M15")
    if m15:
        w1_trend = results.get("W1", AnalysisResult(timeframe="W1", trend=TrendState.RANGING)).trend
        d1_trend = results.get("D1", AnalysisResult(timeframe="D1", trend=TrendState.RANGING)).trend

        # V20: Collect unmitigated HTF zones for LTF alignment
        htf_zones: list[dict] = []
        for htf_key in ["H4", "D1"]:
            htf = results.get(htf_key)
            if not htf:
                continue
            for ob in htf.order_blocks:
                if ob.valid and not ob.mitigated:
                    htf_zones.append({
                        "type": "OB", "upper": ob.upper_price,
                        "lower": ob.lower_price, "direction": ob.direction.value,
                        "tf": htf_key,
                    })
            for fvg in htf.fvgs:
                if fvg.valid and not fvg.mitigated:
                    htf_zones.append({
                        "type": "FVG", "upper": fvg.upper_price,
                        "lower": fvg.lower_price, "direction": fvg.direction.value,
                        "tf": htf_key,
                    })

        m15_candles = candles_by_tf.get("M15", [])
        m15.signals = generate_signals(
            candles=m15_candles,
            swings=m15.swings,
            inducements=m15.inducements,
            liquidity_pools=m15.liquidity_pools,
            bos_events=m15.bos_events,
            choch_events=m15.choch_events,
            fvgs=m15.fvgs,
            order_blocks=m15.order_blocks,
            trend=m15.trend,
            pd=m15.premium_discount,
            session=m15.session,
            w1_trend=w1_trend,
            d1_trend=d1_trend,
            vsa_absorptions=m15.vsa_absorptions,
            htf_zones=htf_zones,
        )

    return results
