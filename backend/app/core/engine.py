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
from app.config import SWING_LOOKBACK, TRADING_STYLES
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
    all_style_signals: list[TradingSignal] = field(default_factory=list)


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
    tf_hierarchy = [
        ("M5",  ["M15", "H1", "H4", "D1", "W1"]),
        ("M15", ["H1", "H4", "D1", "W1"]),
        ("H1",  ["H4", "D1", "W1"]),
        ("H4",  ["D1", "W1"]),
        ("D1",  ["W1"]),
    ]

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


def _collect_htf_zones(results: dict[str, AnalysisResult], tf_keys: list[str]) -> list[dict]:
    """Collect unmitigated OBs + FVGs from given TFs as htf_zones."""
    zones: list[dict] = []
    for tf_key in tf_keys:
        htf = results.get(tf_key)
        if not htf:
            continue
        for ob in htf.order_blocks:
            if ob.valid and not ob.mitigated:
                zones.append({
                    "type": "OB", "upper": ob.upper_price,
                    "lower": ob.lower_price, "direction": ob.direction.value,
                    "tf": tf_key,
                })
        for fvg in htf.fvgs:
            if fvg.valid and not fvg.mitigated:
                zones.append({
                    "type": "FVG", "upper": fvg.upper_price,
                    "lower": fvg.lower_price, "direction": fvg.direction.value,
                    "tf": tf_key,
                })
    return zones


def _generate_style_signals(
    style_key: str,
    style_cfg: dict,
    results: dict[str, AnalysisResult],
    candles_by_tf: dict[str, list[Candle]],
) -> list[TradingSignal]:
    """Generate signals for a single trading style using its TF hierarchy.

    Each style has: bias_tf → setup_tf → entry_tf.
    Bias TF provides trend direction, setup TF provides zones (OBs + FVGs),
    entry TF provides the candles and detectors for signal generation.
    """
    bias_tf = style_cfg["bias_tf"]
    setup_tf = style_cfg["setup_tf"]
    entry_tf = style_cfg["entry_tf"]

    entry_result = results.get(entry_tf)
    if not entry_result:
        return []

    entry_candles = candles_by_tf.get(entry_tf, [])
    if not entry_candles:
        return []

    # Get bias from bias TF trend
    bias_result = results.get(bias_tf)
    bias_trend = bias_result.trend if bias_result else TrendState.RANGING

    # Collect setup zones from setup TF (and bias TF for higher-level zones)
    setup_zones = _collect_htf_zones(results, [setup_tf])

    # w1_trend and d1_trend for context fields on TradingSignal
    w1_trend = results.get("W1", AnalysisResult(timeframe="W1", trend=TrendState.RANGING)).trend
    d1_trend = results.get("D1", AnalysisResult(timeframe="D1", trend=TrendState.RANGING)).trend

    return generate_signals(
        candles=entry_candles,
        swings=entry_result.swings,
        inducements=entry_result.inducements,
        liquidity_pools=entry_result.liquidity_pools,
        bos_events=entry_result.bos_events,
        choch_events=entry_result.choch_events,
        fvgs=entry_result.fvgs,
        order_blocks=entry_result.order_blocks,
        trend=entry_result.trend,
        pd=entry_result.premium_discount,
        session=entry_result.session,
        w1_trend=w1_trend,
        d1_trend=d1_trend,
        vsa_absorptions=entry_result.vsa_absorptions,
        htf_zones=setup_zones,
        trade_bias_override=bias_trend,
        trading_style=style_key,
        entry_timeframe=entry_tf,
    )


def run_multi_tf_analysis(
    candles_by_tf: dict[str, list[Candle]],
) -> dict[str, AnalysisResult]:
    """Run analysis across all 7 timeframes and generate multi-style signals.

    V20: Each TF is analyzed once, then results are reused across all 6
    trading styles (Positional, Swing, Short-Term, Intraday, Day Trading, Scalping).
    """
    results: dict[str, AnalysisResult] = {}

    # Analyze all 7 timeframes
    for tf in ["1M", "W1", "D1", "H4", "H1", "M15", "M5"]:
        candles = candles_by_tf.get(tf, [])
        if candles:
            results[tf] = analyze_timeframe(candles, tf)
        else:
            results[tf] = AnalysisResult(timeframe=tf, trend=TrendState.RANGING)

    # V09 Rule 4: Cross-TF fake CHoCH detection
    _apply_cross_tf_fake_choch(results)

    # ── Generate signals for each V20 trading style ──
    all_style_signals: list[TradingSignal] = []
    for style_key, style_cfg in TRADING_STYLES.items():
        style_signals = _generate_style_signals(
            style_key, style_cfg, results, candles_by_tf,
        )
        all_style_signals.extend(style_signals)

    # Store all style signals on M15 result (primary entry for backward compat)
    m15 = results.get("M15")
    if m15:
        # Backward compat: m15.signals = intraday-only signals
        m15.signals = [s for s in all_style_signals if s.trading_style == "intraday"]
        # All styles combined
        m15.all_style_signals = all_style_signals

    return results
