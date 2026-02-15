"""Quant Microstructure Filter (Layer 2) — VPIN, Liquidation Cascades, Taker Bursts, OI Divergence.

Weight: 20% of combined quant score.
All computations use existing data — no new API calls for TIER 1.
"""

from __future__ import annotations

import math
import statistics

from app.models import Candle, Direction, VolatilityRegime
from app.config import VPIN_BUCKET_COUNT, VPIN_THRESHOLD


def compute_vpin(taker_volume_data: list[dict], bucket_count: int = VPIN_BUCKET_COUNT) -> float:
    """Compute VPIN (Volume-Synchronized Probability of Informed Trading).

    VPIN measures flow toxicity — probability that volume is driven by
    informed (institutional) rather than uninformed (retail) traders.

    Algorithm:
    1. Take taker buy/sell volume data (hourly from Binance futures)
    2. Bucket into volume-synchronized bars
    3. For each bucket, compute |buy - sell| / total as order imbalance
    4. VPIN = rolling average of imbalances over N buckets

    Returns 0.0-1.0 where > 0.7 indicates high probability of informed trading.
    """
    if not taker_volume_data or len(taker_volume_data) < 2:
        return 0.0

    # Compute total volume per entry and imbalances
    imbalances: list[float] = []
    for entry in taker_volume_data:
        buy_vol = entry.get("buy_vol", 0)
        sell_vol = entry.get("sell_vol", 0)
        total = buy_vol + sell_vol
        if total > 0:
            imbalance = abs(buy_vol - sell_vol) / total
            imbalances.append(imbalance)

    if not imbalances:
        return 0.0

    # Use rolling window of bucket_count (or all available if fewer)
    window = imbalances[-min(bucket_count, len(imbalances)):]
    vpin = sum(window) / len(window)

    return max(0.0, min(1.0, vpin))


def compute_vpin_history(taker_volume_data: list[dict], window: int = 10) -> list[dict]:
    """Compute rolling VPIN values for chart display.

    Returns list of {timestamp, value} for the VPIN timeseries.
    """
    if not taker_volume_data or len(taker_volume_data) < window:
        return []

    result = []
    for i in range(window, len(taker_volume_data) + 1):
        chunk = taker_volume_data[i - window:i]
        vpin_val = compute_vpin(chunk, bucket_count=window)
        result.append({
            "timestamp": chunk[-1].get("timestamp", 0),
            "value": round(vpin_val, 4),
        })

    return result


def get_taker_imbalance_direction(taker_volume_data: list[dict], lookback: int = 4) -> tuple[str, float]:
    """Determine aggregate taker flow direction from recent data.

    Returns (direction, ratio) where direction is "buy" or "sell"
    and ratio is the buy/sell ratio.
    """
    if not taker_volume_data:
        return "neutral", 1.0

    recent = taker_volume_data[-lookback:]
    total_buy = sum(t.get("buy_vol", 0) for t in recent)
    total_sell = sum(t.get("sell_vol", 0) for t in recent)

    if total_sell == 0:
        return "buy", 999.0
    if total_buy == 0:
        return "sell", 0.0

    ratio = total_buy / total_sell
    if ratio > 1.05:
        return "buy", ratio
    elif ratio < 0.95:
        return "sell", ratio
    return "neutral", ratio


def detect_liquidation_cascade(
    liquidations: list[dict],
    threshold_usd: float = 50_000_000,
    window_min: int = 30,
) -> dict | None:
    """Detect liquidation cascade — $50M+ forced liquidations within a window.

    A cascade creates a self-reinforcing waterfall that quant funds front-run or fade.
    Returns cascade info dict or None if no cascade detected.
    """
    if not liquidations:
        return None

    import time
    now_ms = int(time.time() * 1000)
    window_ms = window_min * 60 * 1000

    # Bucket liquidations by side
    buy_total = 0.0
    sell_total = 0.0
    buy_count = 0
    sell_count = 0

    for liq in liquidations:
        ts = liq.get("timestamp", 0)
        if now_ms - ts > window_ms:
            continue

        qty_usd = liq.get("qty_usd", 0)
        side = liq.get("side", "")

        if side == "sell":  # Longs getting liquidated
            sell_total += qty_usd
            sell_count += 1
        elif side == "buy":  # Shorts getting liquidated
            buy_total += qty_usd
            buy_count += 1

    total = buy_total + sell_total
    if total < threshold_usd:
        return None

    dominant_side = "sell" if sell_total > buy_total else "buy"
    return {
        "detected": True,
        "total_usd": total,
        "buy_liquidations_usd": buy_total,
        "sell_liquidations_usd": sell_total,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "dominant_side": dominant_side,
        "window_minutes": window_min,
    }


def check_liquidation_proximity(
    liquidations: list[dict],
    sl_price: float,
    tp_price: float,
    tolerance_pct: float = 0.5,
) -> dict:
    """Check for liquidation clusters near SL/TP zones.

    Liquidation clusters near SL dramatically change signal quality:
    - Cluster near SL = SL may get run by cascade → higher risk
    - Cluster near TP = cascade may push price past TP → higher reward chance
    """
    near_sl = 0.0
    near_tp = 0.0
    near_sl_count = 0
    near_tp_count = 0

    for liq in liquidations:
        price = liq.get("price", 0)
        qty_usd = liq.get("qty_usd", 0)
        if price <= 0:
            continue

        # Check proximity to SL
        if sl_price > 0:
            dist_sl_pct = abs(price - sl_price) / sl_price * 100
            if dist_sl_pct <= tolerance_pct:
                near_sl += qty_usd
                near_sl_count += 1

        # Check proximity to TP
        if tp_price > 0:
            dist_tp_pct = abs(price - tp_price) / tp_price * 100
            if dist_tp_pct <= tolerance_pct:
                near_tp += qty_usd
                near_tp_count += 1

    return {
        "near_sl_usd": near_sl,
        "near_sl_count": near_sl_count,
        "near_tp_usd": near_tp,
        "near_tp_count": near_tp_count,
        "sl_risk_elevated": near_sl > 10_000_000,  # $10M+ near SL = elevated
        "tp_catalyst": near_tp > 10_000_000,       # $10M+ near TP = catalyst
    }


def compute_taker_burst(taker_data: list[dict]) -> dict:
    """Detect 3-sigma taker volume spikes indicating institutional flow.

    A burst in buy or sell volume, significantly above the rolling average,
    signals smart money positioning.
    """
    if len(taker_data) < 10:
        return {"detected": False, "direction": "neutral", "magnitude": 0.0}

    buy_vols = [t.get("buy_vol", 0) for t in taker_data]
    sell_vols = [t.get("sell_vol", 0) for t in taker_data]

    # Rolling stats on total volume
    total_vols = [b + s for b, s in zip(buy_vols, sell_vols)]
    mean_vol = statistics.mean(total_vols[:-1])  # Exclude last for comparison
    if len(total_vols) > 2:
        stdev_vol = statistics.stdev(total_vols[:-1])
    else:
        stdev_vol = mean_vol * 0.3

    if stdev_vol == 0:
        return {"detected": False, "direction": "neutral", "magnitude": 0.0}

    latest_total = total_vols[-1]
    z_score = (latest_total - mean_vol) / stdev_vol

    if z_score < 3.0:
        return {"detected": False, "direction": "neutral", "magnitude": round(z_score, 2)}

    # Determine direction of the burst
    latest_buy = buy_vols[-1]
    latest_sell = sell_vols[-1]
    direction = "buy" if latest_buy > latest_sell else "sell"

    return {
        "detected": True,
        "direction": direction,
        "magnitude": round(z_score, 2),
        "latest_total": latest_total,
        "mean": mean_vol,
    }


def _score_oi_divergence(futures_data: dict, candles: list[Candle], signal_direction: Direction) -> float:
    """OI Delta + Price Divergence scoring.

    OI rising + price falling = accumulation (bullish)
    OI rising + price rising = momentum continuation
    OI falling = unwinding (weaker conviction)
    """
    oi_hist = futures_data.get("open_interest", {}).get("history", [])
    if len(oi_hist) < 12 or len(candles) < 12:
        return 0.0

    recent_oi = oi_hist[-1].get("oi_usd", 0)
    earlier_oi = oi_hist[-12].get("oi_usd", 0)
    if earlier_oi <= 0:
        return 0.0

    oi_change_pct = (recent_oi - earlier_oi) / earlier_oi * 100
    price_change_pct = (candles[-1].close - candles[-12].close) / candles[-12].close * 100

    is_bull = (
        signal_direction == Direction.BULLISH
        if isinstance(signal_direction, Direction)
        else signal_direction == "bullish"
    )

    score = 0.0

    # Rising OI + falling price = accumulation
    if oi_change_pct > 2.0 and price_change_pct < -1.0:
        score = 8.0 if is_bull else -5.0  # Bullish divergence
    # Rising OI + rising price = momentum
    elif oi_change_pct > 2.0 and price_change_pct > 1.0:
        score = 5.0 if is_bull else -3.0
    # Falling OI = unwinding
    elif oi_change_pct < -2.0:
        score = -5.0  # Weaker conviction either way
    # Rising OI neutral
    elif oi_change_pct > 2.0:
        score = 3.0

    return max(-10.0, min(10.0, score))


def compute_microstructure_score(
    quant_data: dict,
    futures_data: dict,
    candles: list[Candle],
    signal_direction: Direction,
) -> tuple[float, dict]:
    """Compute microstructure filter score (-100 to +100).

    Components (Layer 2):
    1. VPIN flow toxicity (weight 3)
    2. OI Delta + Divergence (weight 3)
    3. Liquidation heat map (weight 2)
    4. Taker flow imbalance (weight 2)
    5. Funding cross-exchange (weight 2) — added when TIER 2 available
    6. Options Skew + GEX (weight 2) — added when TIER 2 available

    Returns (score, component_details).
    """
    is_bull = (
        signal_direction == Direction.BULLISH
        if isinstance(signal_direction, Direction)
        else signal_direction == "bullish"
    )

    components = {}
    weighted_sum = 0.0
    total_weight = 0.0

    # 1. VPIN flow toxicity
    vpin = quant_data.get("vpin", 0.0)
    taker_dir, taker_ratio = get_taker_imbalance_direction(
        futures_data.get("taker_volume", [])
    )
    vpin_score = 0.0
    if vpin >= VPIN_THRESHOLD:
        # High VPIN = informed flow — direction matters
        if (is_bull and taker_dir == "buy") or (not is_bull and taker_dir == "sell"):
            vpin_score = 8.0  # Aligned with signal
        elif taker_dir == "neutral":
            vpin_score = 3.0  # Informed flow, unclear direction
        else:
            vpin_score = -6.0  # Informed flow opposing signal
    elif vpin >= 0.5:
        vpin_score = 2.0  # Moderate informed flow
    components["vpin"] = {"score": round(vpin_score, 1), "value": round(vpin, 3)}
    weighted_sum += vpin_score * 3
    total_weight += 3

    # 2. OI Delta + Divergence
    oi_score = _score_oi_divergence(futures_data, candles, signal_direction)
    components["oi_divergence"] = {"score": round(oi_score, 1)}
    weighted_sum += oi_score * 3
    total_weight += 3

    # 3. Liquidation heat map
    cascade = quant_data.get("liquidation_cascade")
    liq_score = 0.0
    if cascade and cascade.get("detected"):
        dominant = cascade.get("dominant_side", "")
        # Opposite-side cascade = exhaustion = supports signal
        if (is_bull and dominant == "sell") or (not is_bull and dominant == "buy"):
            liq_score = 8.0  # Cascade exhausted opposing side
        else:
            liq_score = -6.0  # Cascade on our side
    components["liquidation_cascade"] = {"score": round(liq_score, 1)}
    weighted_sum += liq_score * 2
    total_weight += 2

    # 4. Taker flow imbalance
    taker_score = 0.0
    if taker_ratio > 1.15 and is_bull:
        taker_score = 6.0
    elif taker_ratio < 0.85 and not is_bull:
        taker_score = 6.0
    elif taker_ratio > 1.15 and not is_bull:
        taker_score = -5.0
    elif taker_ratio < 0.85 and is_bull:
        taker_score = -5.0
    components["taker_flow"] = {"score": round(taker_score, 1), "ratio": round(taker_ratio, 3)}
    weighted_sum += taker_score * 2
    total_weight += 2

    # 5. Cross-exchange funding divergence (TIER 2 — may be empty)
    cross_funding = quant_data.get("cross_exchange_funding", {})
    if cross_funding and cross_funding.get("dispersion", 0) > 0:
        dispersion = cross_funding["dispersion"]
        avg_rate = cross_funding.get("avg_rate", 0)
        funding_score = 0.0
        if dispersion > 0.001:  # Significant divergence
            if is_bull and avg_rate < 0:
                funding_score = 5.0  # Shorts crowded across exchanges
            elif not is_bull and avg_rate > 0:
                funding_score = 5.0  # Longs crowded across exchanges
            else:
                funding_score = -3.0
        components["cross_funding"] = {"score": round(funding_score, 1), "dispersion": round(dispersion, 5)}
        weighted_sum += funding_score * 2
        total_weight += 2

    # 6. Options Skew + GEX (TIER 2 — may be empty)
    options = quant_data.get("options_data", {})
    if options and options.get("skew_25d") is not None:
        skew = options["skew_25d"]
        gex = options.get("net_gex", 0)
        opt_score = 0.0
        # Skew > 5 = fear (bearish), < -5 = complacency (bullish)
        if is_bull and skew > 5:
            opt_score = -4.0  # Market fearful, opposing bullish signal
        elif is_bull and skew < -5:
            opt_score = 4.0  # Complacent, potential upside
        elif not is_bull and skew > 5:
            opt_score = 4.0  # Fear supports bearish signal
        elif not is_bull and skew < -5:
            opt_score = -4.0  # Complacent, opposing bearish
        components["options_skew_gex"] = {"score": round(opt_score, 1), "skew": round(skew, 2), "gex": gex}
        weighted_sum += opt_score * 2
        total_weight += 2

    # Normalize to -100..+100
    if total_weight > 0:
        raw = (weighted_sum / total_weight) * 10  # Scale from ±10 to ±100
    else:
        raw = 0.0

    score = max(-100.0, min(100.0, raw))
    return score, components
