"""TradingMamba API — FastAPI backend serving ICT/SMC analysis.

Endpoints:
- GET /api/analyze          → Run full multi-TF analysis on BTCUSDT
- GET /api/analyze/{tf}     → Run single-TF analysis
- GET /api/signals          → Get current trading signals
- GET /api/health           → Health check
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import SYMBOL
from app.services.data_fetcher import fetch_klines, fetch_all_timeframes
from app.core.engine import analyze_timeframe, run_multi_tf_analysis, AnalysisResult
from app.models import TrendState
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
