"""Quant Engine Orchestrator — applies 4-layer quant scoring to SMC signals.

This is the main entry point for quant mode. Called from core/engine.py
when mode="quant", AFTER SMC signals are generated.

Flow:
1. Compute TIER 1 metrics from existing data (ATR, DVOL, VPIN, liquidations)
2. Fetch TIER 2+3 data with graceful degradation
3. Score each signal through Alpha + Microstructure + Risk + Execution
4. Modify grades, SL/TP, position sizing
5. Return QuantContext + modified signals

Provides both sync (`apply_quant_layer_sync`) and async (`apply_quant_layer`)
versions. Sync uses TIER 1 only (pure computation). Async adds TIER 2+3 fetches.
"""

from __future__ import annotations

import logging
import time

import httpx

from app.models import (
    Candle, TradingSignal, Direction, VolatilityRegime,
    QuantScore, QuantContext, PricePhase,
)
from app.config import DEMO_RISK_PER_TRADE_PCT, CACHE_TTL_DVOL

from app.quant.risk_model import (
    compute_atr, detect_volatility_regime,
    compute_atr_sl_tp, compute_position_size,
    compute_risk_tradability, should_suppress_signal,
)
from app.quant.microstructure import (
    compute_vpin, compute_vpin_history, get_taker_imbalance_direction,
    detect_liquidation_cascade, check_liquidation_proximity,
    compute_taker_burst, compute_microstructure_score,
)
from app.quant.alpha_model import compute_alpha_score
from app.quant.execution_model import compute_execution_score
from app.quant.scoring import compute_quant_score, apply_grade_modification

logger = logging.getLogger(__name__)

# ── DVOL sync cache ──
_dvol_cache: dict = {"value": 0.0, "fetched_at": 0.0}


def _fetch_dvol_sync() -> float:
    """Fetch DVOL synchronously with TTL cache.

    Uses Deribit's volatility index data endpoint (OHLC format).
    Returns the latest BTC DVOL close value.
    """
    now = time.time()
    if _dvol_cache["value"] > 0 and (now - _dvol_cache["fetched_at"]) < CACHE_TTL_DVOL:
        return _dvol_cache["value"]

    try:
        now_ms = int(now * 1000)
        resp = httpx.get(
            "https://www.deribit.com/api/v2/public/get_volatility_index_data",
            params={
                "currency": "BTC",
                "resolution": "3600",
                "start_timestamp": str(now_ms - 7_200_000),  # last 2 hours
                "end_timestamp": str(now_ms),
            },
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            entries = data.get("result", {}).get("data", [])
            if entries:
                # Each entry: [timestamp, open, high, low, close]
                dvol = entries[-1][4]  # latest close
                _dvol_cache["value"] = dvol
                _dvol_cache["fetched_at"] = now
                return dvol
    except Exception as e:
        logger.debug("DVOL sync fetch failed: %s", e)

    return _dvol_cache["value"]


def _fetch_tier2_sync() -> dict:
    """Fetch TIER 2+3 data synchronously with graceful degradation."""
    result: dict = {}

    try:
        # Fear & Greed — simplest, most reliable
        resp = httpx.get(
            "https://api.alternative.me/fng/",
            params={"limit": 1},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            entries = data.get("data", [])
            if entries:
                entry = entries[0]
                result["fear_greed"] = {
                    "value": int(entry.get("value", 50)),
                    "classification": entry.get("value_classification", "Neutral"),
                }
    except Exception as e:
        logger.debug("Fear & Greed sync fetch failed: %s", e)

    try:
        # Cross-exchange funding: Bybit
        resp = httpx.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            timeout=5,
        )
        rates = {}
        if resp.status_code == 200:
            data = resp.json()
            tickers = data.get("result", {}).get("list", [])
            if tickers:
                rates["bybit"] = float(tickers[0].get("fundingRate", 0))

        # OKX
        resp2 = httpx.get(
            "https://www.okx.com/api/v5/public/funding-rate",
            params={"instId": "BTC-USDT-SWAP"},
            timeout=5,
        )
        if resp2.status_code == 200:
            data2 = resp2.json()
            entries2 = data2.get("data", [])
            if entries2:
                rates["okx"] = float(entries2[0].get("fundingRate", 0))

        if rates:
            all_rates = list(rates.values())
            avg = sum(all_rates) / len(all_rates)
            disp = max(all_rates) - min(all_rates) if len(all_rates) > 1 else 0
            result["cross_exchange_funding"] = {
                "bybit_rate": rates.get("bybit", 0),
                "okx_rate": rates.get("okx", 0),
                "avg_rate": round(avg, 8),
                "dispersion": round(disp, 8),
            }
    except Exception as e:
        logger.debug("Cross-exchange funding sync fetch failed: %s", e)

    try:
        # On-chain exchange flows (CoinMetrics, FREE)
        resp = httpx.get(
            "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
            params={
                "assets": "btc",
                "metrics": "FlowInExNtv,FlowOutExNtv",
                "frequency": "1d",
                "page_size": "7",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            rows = resp.json().get("data", [])
            if rows:
                latest = rows[-1]
                inflow = float(latest.get("FlowInExNtv", 0) or 0)
                outflow = float(latest.get("FlowOutExNtv", 0) or 0)
                net = inflow - outflow
                signal = "neutral"
                if net > 5000:
                    signal = "strong_inflow"
                elif net > 1000:
                    signal = "mild_inflow"
                elif net < -5000:
                    signal = "strong_outflow"
                elif net < -1000:
                    signal = "mild_outflow"
                result["onchain_flow"] = {
                    "inflow_btc": round(inflow, 2),
                    "outflow_btc": round(outflow, 2),
                    "net_flow": round(net, 2),
                    "signal": signal,
                }
    except Exception as e:
        logger.debug("On-chain sync fetch failed: %s", e)

    try:
        # CFTC COT data (weekly, 7-day cache)
        resp = httpx.get(
            "https://publicreporting.cftc.gov/resource/gpe5-46if.json",
            params={
                "$where": "contract_market_name like '%BITCOIN%'",
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": "200",
                "$select": (
                    "report_date_as_yyyy_mm_dd,contract_market_name,"
                    "lev_money_positions_long,lev_money_positions_short,"
                    "asset_mgr_positions_long,asset_mgr_positions_short,"
                    "dealer_positions_long_all,dealer_positions_short_all"
                ),
            },
            timeout=10,
        )
        if resp.status_code == 200:
            from app.services.quant_data import _compute_percentile, _cot_signal
            rows = resp.json()
            if rows:
                by_date: dict[str, dict] = {}
                for row in rows:
                    date = row.get("report_date_as_yyyy_mm_dd", "")[:10]
                    if date not in by_date:
                        by_date[date] = {"lev_long": 0, "lev_short": 0}
                    by_date[date]["lev_long"] += int(row.get("lev_money_positions_long", 0) or 0)
                    by_date[date]["lev_short"] += int(row.get("lev_money_positions_short", 0) or 0)
                sorted_dates = sorted(by_date.keys(), reverse=True)
                latest = by_date[sorted_dates[0]]
                lev_net = latest["lev_long"] - latest["lev_short"]
                hist = [by_date[d]["lev_long"] - by_date[d]["lev_short"] for d in sorted_dates]
                pct = _compute_percentile(lev_net, hist)
                result["cot_data"] = {
                    "leveraged_net": lev_net,
                    "percentile": round(pct, 1),
                    "signal": _cot_signal(pct),
                    "report_date": sorted_dates[0],
                }
    except Exception as e:
        logger.debug("COT sync fetch failed: %s", e)

    try:
        # Deribit options: P/C ratio, max pain, GEX, 25d skew
        resp = httpx.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
            params={"currency": "BTC", "kind": "option"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            instruments = data.get("result", [])
            if instruments:
                from app.services.quant_data import compute_options_analytics
                result["options_data"] = compute_options_analytics(instruments)
    except Exception as e:
        logger.debug("Deribit options sync fetch failed: %s", e)

    return result


def apply_quant_layer_sync(
    signals: list[TradingSignal],
    candles_by_tf: dict[str, list[Candle]],
    futures_data: dict | None,
    results: dict,
) -> tuple[QuantContext, list[TradingSignal]]:
    """Synchronous quant layer — called from core/engine.py.

    TIER 1 metrics are pure computation (always available).
    TIER 2+3 are fetched synchronously with graceful degradation.
    """
    if not signals:
        return QuantContext(), signals

    try:
        return _apply_quant_impl_sync(signals, candles_by_tf, futures_data, results)
    except Exception as e:
        logger.error("Quant layer failed, returning unmodified signals: %s", e)
        return QuantContext(), signals


def _apply_quant_impl_sync(
    signals: list[TradingSignal],
    candles_by_tf: dict[str, list[Candle]],
    futures_data: dict | None,
    results: dict,
) -> tuple[QuantContext, list[TradingSignal]]:
    """Internal sync implementation."""
    futures_data = futures_data or {}

    # ── Gather candles ──
    m15_candles = candles_by_tf.get("M15", [])
    h4_candles = candles_by_tf.get("H4", [])
    d1_candles = candles_by_tf.get("D1", [])

    # ── TIER 1: Pure computation ──
    atr_m15 = compute_atr(m15_candles)
    atr_h4 = compute_atr(h4_candles)
    atr_d1 = compute_atr(d1_candles)

    # DVOL (sync fetch with cache)
    dvol = _fetch_dvol_sync()

    # Volatility regime
    regime = detect_volatility_regime(m15_candles, dvol=dvol)

    # VPIN from existing taker volume
    taker_volume = futures_data.get("taker_volume", [])
    vpin = compute_vpin(taker_volume)
    vpin_history = compute_vpin_history(taker_volume)
    taker_dir, taker_ratio = get_taker_imbalance_direction(taker_volume)

    # Liquidations from WebSocket buffer
    try:
        from app.services.liquidation_ws import get_liquidation_manager
        liq_manager = get_liquidation_manager()
        recent_liquidations = liq_manager.get_recent(30)
    except Exception:
        recent_liquidations = []

    cascade = detect_liquidation_cascade(recent_liquidations)
    taker_burst = compute_taker_burst(taker_volume)

    # Price cycle phase from M15 result
    m15_result = results.get("M15")
    current_phase = m15_result.current_phase if m15_result else PricePhase.CONSOLIDATION

    # ── TIER 2+3: Sync fetch with degradation ──
    quant_external = _fetch_tier2_sync()

    # ── Build quant_data ──
    quant_data = {
        "vpin": vpin,
        "vpin_history": vpin_history,
        "taker_direction": taker_dir,
        "taker_ratio": taker_ratio,
        "taker_burst": taker_burst,
        "liquidation_cascade": cascade,
        "recent_liquidations": recent_liquidations,
        "volatility_regime": regime,
        "dvol": dvol,
        "cross_exchange_funding": quant_external.get("cross_exchange_funding", {}),
        "options_data": quant_external.get("options_data", {}),
        "cot_data": quant_external.get("cot_data", {}),
        "onchain_flow": quant_external.get("onchain_flow", {}),
        "fear_greed": quant_external.get("fear_greed", {}),
    }

    # ── Risk tradability ──
    tradability = compute_risk_tradability(regime)

    # ── Score each signal ──
    for signal in signals:
        _score_signal(
            signal, quant_data, futures_data, m15_candles,
            atr_m15, regime, tradability, current_phase,
        )

    # ── Build QuantContext ──
    context = QuantContext(
        vpin=round(vpin, 4),
        vpin_history=vpin_history,
        atr_m15=round(atr_m15, 2),
        atr_h4=round(atr_h4, 2),
        atr_d1=round(atr_d1, 2),
        dvol=round(dvol, 2),
        volatility_regime=regime,
        recent_liquidations=recent_liquidations[-50:],
        liquidation_clusters=[cascade] if cascade else [],
        cross_exchange_funding=quant_external.get("cross_exchange_funding", {}),
        options_data=quant_external.get("options_data", {}),
        cot_data=quant_external.get("cot_data", {}),
        onchain_flow=quant_external.get("onchain_flow", {}),
        fear_greed=quant_external.get("fear_greed", {}),
    )

    return context, signals


def _score_signal(
    signal: TradingSignal,
    quant_data: dict,
    futures_data: dict,
    candles: list[Candle],
    atr: float,
    regime: VolatilityRegime,
    tradability: float,
    price_cycle_phase: PricePhase,
):
    """Score a single signal through all 4 quant layers and modify it."""

    # Layer 1: Alpha Model (55%)
    alpha, alpha_components = compute_alpha_score(
        quant_data=quant_data,
        futures_data=futures_data,
        signal_direction=signal.direction,
        volatility_regime=regime,
        price_cycle_phase=price_cycle_phase,
    )

    # Layer 2: Microstructure Filter (20%)
    micro, micro_components = compute_microstructure_score(
        quant_data=quant_data,
        futures_data=futures_data,
        candles=candles,
        signal_direction=signal.direction,
    )

    # Layer 3: Risk tradability already computed (same for all signals)

    # Layer 4: Execution Model (10%)
    execution, exec_components = compute_execution_score(
        session=signal.session,
        candles=candles,
        signal_direction=signal.direction,
        entry_method=signal.entry_method,
        mss_grade=_get_mss_grade(signal),
    )

    # ── Combined scoring ──
    qscore = compute_quant_score(
        alpha_score=alpha,
        micro_score=micro,
        risk_tradability=tradability,
        execution_score=execution,
        alpha_components=alpha_components,
        micro_components=micro_components,
        exec_components=exec_components,
    )

    # ── Apply grade modification ──
    new_grade, confidence_delta = apply_grade_modification(signal.grade, qscore)
    signal.grade = new_grade
    signal.confidence_score = max(0, min(100, signal.confidence_score + confidence_delta))

    # ── ATR-based SL/TP ──
    if atr > 0:
        atr_sl, atr_tp = compute_atr_sl_tp(signal, atr, regime)
        signal.atr_stop_loss = round(atr_sl, 2)
        signal.atr_take_profit = round(atr_tp, 2)

    # ── Position sizing ──
    if atr > 0 and signal.atr_stop_loss > 0:
        signal.position_size_pct = round(
            compute_position_size(
                entry=signal.entry_price,
                sl=signal.atr_stop_loss,
                account_balance=10_000.0,
                regime=regime,
                base_risk_pct=DEMO_RISK_PER_TRADE_PCT,
            ), 2
        )

    # ── Signal suppression ──
    if should_suppress_signal(regime, tradability):
        signal.suppressed = True

    # ── Liquidation proximity ──
    liqs = quant_data.get("recent_liquidations", [])
    if liqs:
        prox = check_liquidation_proximity(liqs, signal.stop_loss, signal.take_profit)
        if prox.get("sl_risk_elevated"):
            qscore.components["liq_sl_risk"] = {"elevated": True, "usd": prox["near_sl_usd"]}
            signal.quant_confluences.append(
                f"Quant: Liq cluster ${prox['near_sl_usd']/1e6:.1f}M near SL"
            )
        if prox.get("tp_catalyst"):
            signal.quant_confluences.append(
                f"Quant: Liq cluster ${prox['near_tp_usd']/1e6:.1f}M near TP (catalyst)"
            )

    # ── Quant confluences ──
    _build_quant_confluences(signal, qscore, quant_data, regime)

    # ── Attach score ──
    signal.quant_score = qscore


def _build_quant_confluences(
    signal: TradingSignal,
    qscore: QuantScore,
    quant_data: dict,
    regime: VolatilityRegime,
):
    """Build human-readable quant confluence strings."""
    confs = signal.quant_confluences

    # VPIN
    vpin = quant_data.get("vpin", 0)
    if vpin >= 0.7:
        confs.append(f"Quant: VPIN elevated {vpin:.2f} (informed flow)")
    elif vpin >= 0.5:
        confs.append(f"Quant: VPIN moderate {vpin:.2f}")

    # Volatility regime
    if regime == VolatilityRegime.LOW:
        confs.append("Quant: Low vol regime (reliable signals)")
    elif regime == VolatilityRegime.HIGH:
        confs.append("Quant: High vol regime (wider SL)")
    elif regime == VolatilityRegime.EXTREME:
        confs.append("Quant: EXTREME vol (signal suppressed)")

    # Liquidation cascade
    cascade = quant_data.get("liquidation_cascade")
    if cascade and cascade.get("detected"):
        total_m = cascade["total_usd"] / 1e6
        side = cascade["dominant_side"]
        confs.append(f"Quant: Liq cascade ${total_m:.0f}M {side}")

    # Taker burst
    burst = quant_data.get("taker_burst", {})
    if burst.get("detected"):
        confs.append(f"Quant: Taker burst {burst['direction']} ({burst['magnitude']:.1f}σ)")

    # On-chain flows
    onchain = quant_data.get("onchain_flow", {})
    if onchain and onchain.get("signal"):
        sig = onchain["signal"]
        net = onchain.get("net_flow", 0)
        if "strong" in sig:
            label = "sell pressure" if net > 0 else "accumulation"
            confs.append(f"Quant: On-chain {net:+,.0f} BTC ({label})")
        elif sig != "neutral":
            label = "mild sell" if net > 0 else "mild accumulation"
            confs.append(f"Quant: On-chain {net:+,.0f} BTC ({label})")

    # COT data
    cot = quant_data.get("cot_data", {})
    if cot and cot.get("signal"):
        sig = cot["signal"]
        pct = cot.get("percentile", 50)
        if sig in ("extreme_long", "extreme_short"):
            confs.append(f"Quant: COT {sig.replace('_', ' ')} (P{pct:.0f})")
        elif sig not in ("neutral",):
            confs.append(f"Quant: COT {sig.replace('_', ' ')} (P{pct:.0f})")

    # Options data
    options = quant_data.get("options_data", {})
    if options.get("max_pain") and options.get("max_pain_distance_pct") is not None:
        mp = options["max_pain"]
        dist = options["max_pain_distance_pct"]
        if abs(dist) < 3.0:
            confs.append(f"Quant: Max pain ${mp:,.0f} (near, {dist:+.1f}%)")
        else:
            confs.append(f"Quant: Max pain ${mp:,.0f} ({dist:+.1f}%)")
    skew = options.get("skew_25d")
    if skew is not None and abs(skew) > 3.0:
        label = "fear/hedging" if skew > 0 else "call demand"
        confs.append(f"Quant: 25Δ skew {skew:+.1f} ({label})")
    gex = options.get("net_gex", 0)
    if gex != 0:
        label = "dampening" if gex > 0 else "amplifying"
        confs.append(f"Quant: GEX {gex:+.0f} ({label})")

    # Grade change
    if qscore.grade_change > 0:
        confs.append(f"Quant: Grade promoted (+{qscore.grade_change})")
    elif qscore.grade_change < 0:
        confs.append(f"Quant: Grade demoted ({qscore.grade_change})")

    # Combined score
    if qscore.combined_score > 40:
        confs.append(f"Quant: Strong confirm ({qscore.combined_score:+.0f})")
    elif qscore.combined_score < -40:
        confs.append(f"Quant: Strong contradict ({qscore.combined_score:+.0f})")


def _get_mss_grade(signal: TradingSignal):
    """Extract MSS grade from signal."""
    from app.models import MSSGrade
    quality = getattr(signal, "mss_quality", "")
    try:
        return MSSGrade(quality) if quality else MSSGrade.NONE
    except ValueError:
        return MSSGrade.NONE
