"""TradingMamba API — FastAPI backend serving ICT/SMC analysis.

Endpoints:
- GET /api/analyze          → Run full multi-TF analysis on BTCUSDT
- GET /api/analyze/{tf}     → Run single-TF analysis
- GET /api/signals          → Get current trading signals
- GET /api/signals/detailed → Get signals with multi-TF context + checklists
- GET /api/health           → Health check
"""

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import SYMBOL
from app.services.data_fetcher import fetch_klines, fetch_all_timeframes
from app.core.engine import analyze_timeframe, run_multi_tf_analysis, AnalysisResult
from app.models import TrendState, Direction, IDMStatus
from app.services.backtester import run_backtest, get_latest_backtest, print_backtest_report

app = FastAPI(
    title="TradingMamba",
    description="ICT/SMC Pattern Detection Engine — powered by 23 Hindi SMC training videos",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serialize_result(result: AnalysisResult, candles=None) -> dict:
    """Convert AnalysisResult to JSON-serializable dict.

    Field names match the frontend TypeScript types exactly.
    """
    return {
        "timeframe": result.timeframe,
        "trend": result.trend.value,
        "candles": [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "index": c.index,
            }
            for c in (candles or [])
        ],
        "swings": [
            {
                "candle_index": s.candle_index,
                "price": s.price,
                "swing_type": s.swing_type.value,
                "classification": s.classification.value,
                "is_valid_smc": s.is_valid_smc,
                "idm_taken": s.idm_taken,
                "is_strong": s.is_strong,
                "candle_closed_properly": s.candle_closed_properly,
            }
            for s in result.swings
        ],
        "inducements": [
            {
                "candle_index": idm.candle_index,
                "price": idm.price,
                "parent_swing_index": idm.parent_swing_index,
                "status": idm.status.value,
                "taken_at_candle": idm.taken_at_candle,
                "body_closed": idm.body_closed,
                "is_major": idm.is_major,
            }
            for idm in result.inducements
        ],
        "liquidity_pools": [
            {
                "price_level": lp.price_level,
                "pool_type": lp.pool_type.value,
                "source": lp.source.value,
                "candle_indices": lp.candle_indices,
                "swept": lp.swept,
                "swept_at_candle": lp.swept_at_candle,
                "event_type": lp.event_type.value if lp.event_type else None,
            }
            for lp in result.liquidity_pools
        ],
        "bos_events": [
            {
                "candle_index": b.candle_index,
                "direction": b.direction.value,
                "broken_swing_index": b.broken_swing_index,
                "broken_price": b.broken_price,
                "valid": b.valid,
                "invalidation_reason": b.invalidation_reason,
                "idm_body_closed": b.idm_body_closed,
            }
            for b in result.bos_events
        ],
        "choch_events": [
            {
                "candle_index": ch.candle_index,
                "direction": ch.direction.value,
                "broken_swing_index": ch.broken_swing_index,
                "broken_price": ch.broken_price,
                "confidence": ch.confidence,
                "has_climax_confluence": ch.has_climax_confluence,
                "is_fake": ch.is_fake,
                "confirmed": ch.confirmed,
                "model": ch.model,
            }
            for ch in result.choch_events
        ],
        "fvgs": [
            {
                "candle_index": f.candle_index,
                "upper_price": f.upper_price,
                "lower_price": f.lower_price,
                "direction": f.direction.value,
                "valid": f.valid,
                "from_extreme_candle": f.from_extreme_candle,
                "mitigated": f.mitigated,
                "mitigated_at_candle": f.mitigated_at_candle,
            }
            for f in result.fvgs
        ],
        "order_blocks": [
            {
                "candle_index_start": ob.candle_index_start,
                "candle_index_end": ob.candle_index_end,
                "upper_price": ob.upper_price,
                "lower_price": ob.lower_price,
                "direction": ob.direction.value,
                "valid": ob.valid,
                "has_fvg": ob.has_fvg,
                "swept_liquidity": ob.swept_liquidity,
                "is_trap": ob.is_trap,
                "mitigated": ob.mitigated,
                "mitigated_at_candle": ob.mitigated_at_candle,
            }
            for ob in result.order_blocks
        ],
        "premium_discount": {
            "swing_high": result.premium_discount.swing_high,
            "swing_low": result.premium_discount.swing_low,
            "equilibrium": result.premium_discount.equilibrium,
            "zone": result.premium_discount.zone.value,
            "depth_pct": round(result.premium_discount.depth_pct, 1),
            "fibonacci_levels": {
                k: round(v, 2) for k, v in result.premium_discount.fibonacci_levels.items()
            },
            "closest_fib": result.premium_discount.closest_fib,
            "fib_distance_pct": result.premium_discount.fib_distance_pct,
            "is_fib_qualified": result.premium_discount.is_fib_qualified,
        } if result.premium_discount else None,
        "session": {
            "name": result.session.name,
            "is_kill_zone": result.session.is_kill_zone,
            "volatility_expectation": result.session.volatility_expectation,
        } if result.session else None,
        "climax_warning": result.climax_warning,
        "climax_ratio": round(result.climax_ratio, 2),
        "amd_patterns": [
            {
                "range_start_idx": amd.range_start_idx,
                "range_end_idx": amd.range_end_idx,
                "range_high": amd.range_high,
                "range_low": amd.range_low,
                "phase": amd.phase.value,
                "sweep_direction": amd.sweep_direction.value if amd.sweep_direction else None,
                "sweep_candle_idx": amd.sweep_candle_idx,
                "mss_candle_idx": amd.mss_candle_idx,
                "mss_confirmed": amd.mss_confirmed,
            }
            for amd in result.amd_patterns
        ],
        "price_cycles": [
            {
                "candle_index": pc.candle_index,
                "phase": pc.phase.value,
                "start_index": pc.start_index,
                "end_index": pc.end_index,
                "high": pc.high,
                "low": pc.low,
                "valid_transition": pc.valid_transition,
            }
            for pc in result.price_cycles
        ],
        "current_phase": result.current_phase.value,
        "signals": [
            {
                "direction": sig.direction.value,
                "entry_price": sig.entry_price,
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
                "risk_reward_ratio": sig.risk_reward_ratio,
                "confidence_score": sig.confidence_score,
                "grade": sig.grade.value,
                "confluences": sig.confluences,
                "timeframe": sig.timeframe,
                "entry_method": sig.entry_method.value if sig.entry_method else None,
                "pattern_type": sig.pattern_type,
                "timestamp": sig.timestamp,
                "climax_warning": sig.climax_warning,
                "is_counter_trend": sig.is_counter_trend,
            }
            for sig in result.signals
        ],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "symbol": SYMBOL, "engine": "TradingMamba v0.1"}


@app.get("/api/analyze/{timeframe}")
async def analyze_single_tf(timeframe: str = "H4"):
    """Run analysis on a single timeframe."""
    tf = timeframe.upper()
    candles = await fetch_klines(SYMBOL, tf, limit=1000)
    result = analyze_timeframe(candles, tf)
    return _serialize_result(result, candles)


@app.get("/api/analyze")
async def analyze_all():
    """Run full multi-TF analysis (W1→D1→H4→M15) and generate signals."""
    candles_by_tf = await fetch_all_timeframes(SYMBOL)
    results = run_multi_tf_analysis(candles_by_tf)

    return {
        "symbol": SYMBOL,
        "timeframes": {
            tf: _serialize_result(r, candles_by_tf.get(tf, []))
            for tf, r in results.items()
        },
        "signals": _serialize_result(
            results["M15"], candles_by_tf.get("M15", [])
        )["signals"] if "M15" in results else [],
    }


@app.get("/api/signals")
async def get_signals():
    """Get current trading signals from M15 analysis."""
    candles_by_tf = await fetch_all_timeframes(SYMBOL)
    results = run_multi_tf_analysis(candles_by_tf)

    m15 = results.get("M15")
    if not m15:
        return {"signals": [], "trend": "ranging"}

    return {
        "symbol": SYMBOL,
        "trend": m15.trend.value,
        "climax_warning": m15.climax_warning,
        "signals": _serialize_result(m15, candles_by_tf.get("M15", []))["signals"],
    }


@app.post("/api/backtest")
async def run_backtest_endpoint(
    start_date: str = "2024-06-01",
    end_date: str = "2025-01-01",
    symbol: str = SYMBOL,
    step_size: int = 96,
):
    """Run Phase 3 backtest on historical data.

    This may take 5-15 minutes depending on date range.
    """
    result = await run_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        step_size=step_size,
    )
    return result


@app.get("/api/backtest/latest")
async def get_latest_backtest_endpoint():
    """Get the most recent backtest results from cache."""
    result = get_latest_backtest()
    if result is None:
        return {"error": "No backtest results found. Run a backtest first."}
    return result


# ──────────────────────────────────────────────
# Phase 4: Detailed Signals + Checklists
# ──────────────────────────────────────────────

def _compute_v24_checklist(
    results: dict[str, AnalysisResult],
    candles_by_tf: dict,
) -> list[dict]:
    """V24 6-Rule Framework: evaluate each rule against live multi-TF data.

    Rule 1: SMC Structure Mapping (W1/D1 trend established)
    Rule 2: Daily FVG + Three Bar Pattern
    Rule 3: H1/M15 Intraday Trend aligned with HTF
    Rule 4: Retail Patterns / Liquidity Spread identification
    Rule 5: Entry Zone active (OB / FVG / S&D)
    Rule 6: Entry Confirmation (MSS / SBC / VSA trigger)
    """
    w1 = results.get("W1")
    d1 = results.get("D1")
    h4 = results.get("H4")
    h1 = results.get("H1")
    m15 = results.get("M15")

    checklist: list[dict] = []

    # Rule 1: SMC Structure — W1/D1 must have a clear trend (not ranging)
    w1_trend = w1.trend if w1 else TrendState.RANGING
    d1_trend = d1.trend if d1 else TrendState.RANGING
    htf_clear = w1_trend != TrendState.RANGING and d1_trend != TrendState.RANGING
    htf_aligned = w1_trend == d1_trend
    if htf_clear and htf_aligned:
        status, detail = "passed", f"W1 {w1_trend.value} + D1 {d1_trend.value} — aligned"
    elif htf_clear:
        status, detail = "pending", f"W1 {w1_trend.value} vs D1 {d1_trend.value} — conflicting"
    else:
        rng = "W1" if w1_trend == TrendState.RANGING else "D1"
        status, detail = "failed", f"{rng} is ranging — no clear structure"
    checklist.append({"rule": 1, "name": "SMC Structure Mapping", "status": status, "detail": detail})

    # Rule 2: Daily FVG + Three Bar Pattern
    d1_fvgs = [f for f in (d1.fvgs if d1 else []) if f.valid and not f.mitigated]
    d1_candles = candles_by_tf.get("D1", [])
    three_bar = False
    if len(d1_candles) >= 3:
        c1, c2, c3 = d1_candles[-3], d1_candles[-2], d1_candles[-1]
        # Bullish 3-bar: c1 sell, c2 sweeps c1 low & closes inside, c3 closes above c1 high
        if c1.is_bearish and c2.low < c1.low and c2.close > c1.low and c3.close > c1.high:
            three_bar = True
        # Bearish 3-bar: c1 buy, c2 sweeps c1 high & closes inside, c3 closes below c1 low
        if c1.is_bullish and c2.high > c1.high and c2.close < c1.high and c3.close < c1.low:
            three_bar = True
    if d1_fvgs and three_bar:
        status, detail = "passed", f"{len(d1_fvgs)} active D1 FVG(s) + 3-bar pattern confirmed"
    elif d1_fvgs:
        status, detail = "pending", f"{len(d1_fvgs)} active D1 FVG(s), no 3-bar pattern"
    elif three_bar:
        status, detail = "pending", "3-bar pattern detected, no active D1 FVG"
    else:
        status, detail = "failed", "No D1 FVG or 3-bar pattern"
    checklist.append({"rule": 2, "name": "Daily FVG + Three Bar Pattern", "status": status, "detail": detail})

    # Rule 3: H1/M15 Intraday Trend aligned with HTF
    h1_trend = h1.trend if h1 else TrendState.RANGING
    m15_trend = m15.trend if m15 else TrendState.RANGING
    htf_bias = d1_trend if d1_trend != TrendState.RANGING else w1_trend
    ltf_aligned = (h1_trend == htf_bias) or (m15_trend == htf_bias)
    if htf_bias == TrendState.RANGING:
        status, detail = "failed", "No HTF bias to align with"
    elif h1_trend == htf_bias and m15_trend == htf_bias:
        status, detail = "passed", f"H1 + M15 both {htf_bias.value} — full alignment"
    elif ltf_aligned:
        aligned_tf = "H1" if h1_trend == htf_bias else "M15"
        status, detail = "pending", f"{aligned_tf} aligned ({htf_bias.value}), other TF diverging"
    else:
        status, detail = "failed", f"H1 {h1_trend.value} / M15 {m15_trend.value} vs HTF {htf_bias.value}"
    checklist.append({"rule": 3, "name": "H1/M15 Intraday Trend Alignment", "status": status, "detail": detail})

    # Rule 4: Retail Patterns / Liquidity Spread
    # Check for swept liquidity pools (retail stops hunted)
    swept_pools = [lp for lp in (m15.liquidity_pools if m15 else []) if lp.swept]
    taken_idms = [idm for idm in (m15.inducements if m15 else []) if idm.status == IDMStatus.TAKEN]
    if swept_pools and taken_idms:
        status, detail = "passed", f"{len(swept_pools)} swept pool(s), {len(taken_idms)} taken IDM(s) — retail exposed"
    elif swept_pools or taken_idms:
        what = f"{len(swept_pools)} swept" if swept_pools else f"{len(taken_idms)} IDM taken"
        status, detail = "pending", f"{what} — partial liquidity activity"
    else:
        status, detail = "failed", "No swept liquidity or taken IDM detected"
    checklist.append({"rule": 4, "name": "Retail Pattern Liquidity", "status": status, "detail": detail})

    # Rule 5: Entry Zone active (OB / FVG)
    active_obs = [ob for ob in (m15.order_blocks if m15 else []) if ob.valid and not ob.mitigated]
    active_fvgs = [f for f in (m15.fvgs if m15 else []) if f.valid and not f.mitigated]
    # Also count H4 zones
    h4_obs = [ob for ob in (h4.order_blocks if h4 else []) if ob.valid and not ob.mitigated]
    h4_fvgs = [f for f in (h4.fvgs if h4 else []) if f.valid and not f.mitigated]
    total_zones = len(active_obs) + len(active_fvgs) + len(h4_obs) + len(h4_fvgs)
    has_ob_fvg_combo = (active_obs and active_fvgs) or (h4_obs and h4_fvgs)
    if has_ob_fvg_combo:
        status, detail = "passed", f"{total_zones} active zone(s) — OB + FVG combo present"
    elif total_zones > 0:
        status, detail = "pending", f"{total_zones} active zone(s), no OB+FVG combo"
    else:
        status, detail = "failed", "No active entry zones"
    checklist.append({"rule": 5, "name": "Entry Zone Active", "status": status, "detail": detail})

    # Rule 6: Entry Confirmation (MSS / SBC / VSA)
    signals = m15.signals if m15 else []
    confirmed_chochs = [c for c in (m15.choch_events if m15 else []) if c.confirmed and not c.is_fake]
    if signals:
        methods = set(s.entry_method.value for s in signals if s.entry_method)
        status, detail = "passed", f"{len(signals)} signal(s) confirmed via {', '.join(methods).upper() or 'N/A'}"
    elif confirmed_chochs:
        status, detail = "pending", f"{len(confirmed_chochs)} confirmed CHoCH — awaiting zone tap"
    else:
        status, detail = "failed", "No entry confirmation — awaiting MSS/SBC trigger"
    checklist.append({"rule": 6, "name": "Entry Confirmation (MSS/SBC)", "status": status, "detail": detail})

    return checklist


def _compute_v23_checklist(
    results: dict[str, AnalysisResult],
    candles_by_tf: dict,
) -> list[dict]:
    """V23 11-Step Master Trading Checklist: evaluate each step against live data."""
    w1 = results.get("W1")
    d1 = results.get("D1")
    h4 = results.get("H4")
    m15 = results.get("M15")

    checklist: list[dict] = []

    # Step 1: Identify Trend (W1 swing classifier)
    w1_trend = w1.trend if w1 else TrendState.RANGING
    if w1_trend != TrendState.RANGING:
        sw = [s for s in (w1.swings if w1 else []) if s.is_valid_smc]
        status = "passed"
        detail = f"W1 {w1_trend.value} — {len(sw)} valid swing(s)"
    else:
        status, detail = "failed", "W1 ranging — no clear trend"
    checklist.append({"step": 1, "name": "Identify Trend", "status": status, "detail": detail})

    # Step 2: Mark Structure (D1 BOS/CHoCH + swing points)
    d1_bos = [b for b in (d1.bos_events if d1 else []) if b.valid]
    d1_choch = [c for c in (d1.choch_events if d1 else []) if c.confirmed]
    d1_swings = [s for s in (d1.swings if d1 else []) if s.is_valid_smc]
    if d1_bos or d1_choch:
        events = f"{len(d1_bos)} BOS" + (f" + {len(d1_choch)} CHoCH" if d1_choch else "")
        status, detail = "passed", f"D1 structure: {events}, {len(d1_swings)} swings"
    elif d1_swings:
        status, detail = "pending", f"{len(d1_swings)} D1 swings, no BOS/CHoCH yet"
    else:
        status, detail = "failed", "No D1 structure detected"
    checklist.append({"step": 2, "name": "Mark Structure", "status": status, "detail": detail})

    # Step 3: Find IDM (D1/H4 inducement after BOS)
    h4_idms = [i for i in (h4.inducements if h4 else []) if i.status != IDMStatus.TRANSFERRED]
    d1_idms = [i for i in (d1.inducements if d1 else []) if i.status != IDMStatus.TRANSFERRED]
    all_idms = h4_idms + d1_idms
    active_idms = [i for i in all_idms if i.status == IDMStatus.ACTIVE]
    if active_idms:
        status, detail = "passed", f"{len(active_idms)} active IDM(s) on D1/H4"
    elif all_idms:
        status, detail = "pending", f"IDMs found but all taken — awaiting new formation"
    else:
        status, detail = "failed", "No IDM detected on D1/H4"
    checklist.append({"step": 3, "name": "Find IDM", "status": status, "detail": detail})

    # Step 4: Find Zone (H4 OB + FVG below IDM)
    h4_zones = [ob for ob in (h4.order_blocks if h4 else []) if ob.valid and not ob.mitigated]
    h4_fvgs = [f for f in (h4.fvgs if h4 else []) if f.valid and not f.mitigated]
    if h4_zones and h4_fvgs:
        status, detail = "passed", f"{len(h4_zones)} OB + {len(h4_fvgs)} FVG on H4"
    elif h4_zones or h4_fvgs:
        what = f"{len(h4_zones)} OB" if h4_zones else f"{len(h4_fvgs)} FVG"
        status, detail = "pending", f"H4 has {what} only"
    else:
        status, detail = "failed", "No active zones on H4"
    checklist.append({"step": 4, "name": "Find Zone", "status": status, "detail": detail})

    # Step 5: Wait for Tap (price at zone)
    m15_candles = candles_by_tf.get("M15", [])
    current_price = m15_candles[-1].close if m15_candles else 0
    at_zone = False
    for ob in h4_zones:
        if ob.lower_price <= current_price <= ob.upper_price:
            at_zone = True
            break
    for f in h4_fvgs:
        if f.lower_price <= current_price <= f.upper_price:
            at_zone = True
            break
    if at_zone:
        status, detail = "passed", f"Price ${current_price:,.0f} inside active zone"
    else:
        status, detail = "pending", f"Price ${current_price:,.0f} — waiting for zone tap"
    checklist.append({"step": 5, "name": "Wait for Zone Tap", "status": status, "detail": detail})

    # Step 6: Confirm Entry (MSS / SBC / Pullback Break on M15)
    signals = m15.signals if m15 else []
    if signals:
        methods = set(s.entry_method.value for s in signals if s.entry_method)
        status, detail = "passed", f"Entry confirmed: {', '.join(methods).upper()}"
    else:
        status, detail = "pending", "Awaiting M15 entry confirmation"
    checklist.append({"step": 6, "name": "Confirm Entry", "status": status, "detail": detail})

    # Step 7: Set SL (check if signals have valid SL placement)
    if signals:
        avg_risk = sum(abs(s.entry_price - s.stop_loss) / s.entry_price * 100 for s in signals) / len(signals)
        status, detail = "passed", f"SL set — avg risk {avg_risk:.2f}%"
    else:
        status, detail = "pending", "No signal — SL not applicable yet"
    checklist.append({"step": 7, "name": "Set Stop Loss", "status": status, "detail": detail})

    # Step 8: Set TP (check if signals have TP)
    if signals:
        avg_rr = sum(s.risk_reward_ratio for s in signals) / len(signals)
        status, detail = "passed", f"TP set — avg R:R {avg_rr:.1f}"
    else:
        status, detail = "pending", "No signal — TP not applicable yet"
    checklist.append({"step": 8, "name": "Set Take Profit", "status": status, "detail": detail})

    # Step 9: Monitor Climax
    climax = m15.climax_warning if m15 else False
    ratio = m15.climax_ratio if m15 else 0
    if climax:
        status, detail = "failed", f"CLIMAX ACTIVE — ratio {ratio:.1f}x (caution!)"
    else:
        status, detail = "passed", f"No climax — ratio {ratio:.1f}x (safe)"
    checklist.append({"step": 9, "name": "Monitor Climax", "status": status, "detail": detail})

    # Step 10: Detect CHoCH (D1 direction switch)
    recent_d1_choch = [c for c in (d1.choch_events if d1 else []) if c.confirmed and not c.is_fake]
    if recent_d1_choch:
        latest = max(recent_d1_choch, key=lambda c: c.candle_index)
        status, detail = "passed", f"D1 CHoCH {latest.direction.value} detected — direction switch"
    else:
        status, detail = "pending", "No D1 CHoCH — trend continuation"
    checklist.append({"step": 10, "name": "Detect CHoCH", "status": status, "detail": detail})

    # Step 11: Counter-Trend conditions
    ct_signals = [s for s in signals if s.is_counter_trend]
    body_closed_idms = [i for i in (m15.inducements if m15 else []) if i.status == IDMStatus.TAKEN and i.body_closed]
    m15_fvgs = [f for f in (m15.fvgs if m15 else []) if f.valid and not f.mitigated]
    m15_bos = [b for b in (m15.bos_events if m15 else []) if b.valid]
    ct_ready = bool(body_closed_idms and m15_fvgs and m15_bos)
    if ct_signals:
        status, detail = "passed", f"{len(ct_signals)} counter-trend signal(s) active"
    elif ct_ready:
        status, detail = "pending", "BOS + IDM close + FVG present — CT possible"
    else:
        status, detail = "pending", "Counter-trend conditions not met"
    checklist.append({"step": 11, "name": "Counter-Trend Check", "status": status, "detail": detail})

    return checklist


@app.get("/api/signals/detailed")
async def get_detailed_signals():
    """Get signals with full multi-TF context + V24/V23 checklists for the Signals tab."""
    candles_by_tf = await fetch_all_timeframes(SYMBOL)
    results = run_multi_tf_analysis(candles_by_tf)

    w1 = results.get("W1")
    d1 = results.get("D1")
    h4 = results.get("H4")
    h1 = results.get("H1")
    m15 = results.get("M15")

    return {
        "symbol": SYMBOL,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "context": {
            "w1_trend": w1.trend.value if w1 else "ranging",
            "d1_trend": d1.trend.value if d1 else "ranging",
            "h4_trend": h4.trend.value if h4 else "ranging",
            "h1_trend": h1.trend.value if h1 else "ranging",
            "m15_trend": m15.trend.value if m15 else "ranging",
            "session": {
                "name": m15.session.name,
                "is_kill_zone": m15.session.is_kill_zone,
                "volatility_expectation": m15.session.volatility_expectation,
            } if m15 and m15.session else None,
            "climax_warning": m15.climax_warning if m15 else False,
            "climax_ratio": round(m15.climax_ratio, 2) if m15 else 0,
            "current_phase": m15.current_phase.value if m15 else "consolidation",
            "premium_discount": {
                "swing_high": m15.premium_discount.swing_high,
                "swing_low": m15.premium_discount.swing_low,
                "equilibrium": m15.premium_discount.equilibrium,
                "zone": m15.premium_discount.zone.value,
                "depth_pct": round(m15.premium_discount.depth_pct, 1),
                "is_fib_qualified": m15.premium_discount.is_fib_qualified,
                "closest_fib": m15.premium_discount.closest_fib,
            } if m15 and m15.premium_discount else None,
        },
        "checklist_v24": _compute_v24_checklist(results, candles_by_tf),
        "checklist_v23": _compute_v23_checklist(results, candles_by_tf),
        "signals": [
            {
                "direction": sig.direction.value,
                "entry_price": sig.entry_price,
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
                "risk_reward_ratio": sig.risk_reward_ratio,
                "confidence_score": sig.confidence_score,
                "grade": sig.grade.value,
                "confluences": sig.confluences,
                "timeframe": sig.timeframe,
                "entry_method": sig.entry_method.value if sig.entry_method else None,
                "pattern_type": sig.pattern_type,
                "timestamp": sig.timestamp,
                "climax_warning": sig.climax_warning,
                "is_counter_trend": sig.is_counter_trend,
            }
            for sig in (m15.signals if m15 else [])
        ],
    }
