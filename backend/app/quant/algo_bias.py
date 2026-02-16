"""Quant Algo Bias — 10 From-Scratch Institutional Algorithms + Meta-Ensemble.

Every formula is built from first principles using only raw API data.
Academic references noted per algorithm.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

@dataclass
class AlgoBiasResult:
    algo_id: str
    algo_name: str
    direction: str          # "bullish" | "bearish" | "neutral"
    score: float            # -100 to +100
    confidence: float       # 0 to 100
    components: dict = field(default_factory=dict)
    explanation: str = ""
    data_age_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompositeBias:
    direction: str          # "strong_bullish"|"bullish"|"neutral"|"bearish"|"strong_bearish"
    score: float
    confidence: float
    algo_count: int
    agreement_count: int
    regime: str             # "trending"|"mean_reverting"|"random"
    vol_regime: str         # "low"|"normal"|"high"|"extreme"
    entropy: float
    algos: list = field(default_factory=list)
    timestamp: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 1),
            "algo_count": self.algo_count,
            "agreement_count": self.agreement_count,
            "regime": self.regime,
            "vol_regime": self.vol_regime,
            "entropy": round(self.entropy, 3),
            "algos": [a.to_dict() if isinstance(a, AlgoBiasResult) else a for a in self.algos],
            "timestamp": self.timestamp,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0001
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return max(math.sqrt(var), 0.0001)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _covariance(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)


def _variance(x: list[float]) -> float:
    if len(x) < 2:
        return 0.0001
    m = sum(x) / len(x)
    return max(sum((v - m) ** 2 for v in x) / (len(x) - 1), 0.0001)


def _percentile_rank(value: float, values: list[float]) -> float:
    if not values:
        return 50.0
    count_below = sum(1 for v in values if v < value)
    return count_below / len(values) * 100


def _direction_from_score(score: float, threshold: float = 5.0) -> str:
    if score > threshold:
        return "bullish"
    elif score < -threshold:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Algorithm 1: VPIN — Easley-López de Prado-O'Hara (2012)
# ---------------------------------------------------------------------------

def algo_vpin(raw: dict) -> AlgoBiasResult:
    """Volume-Synchronized Probability of Informed Trading."""
    taker = raw.get("taker_volume", [])
    if len(taker) < 5:
        return AlgoBiasResult("vpin", "VPIN Flow Toxicity", "neutral", 0, 0,
                              explanation="Insufficient taker volume data")

    # Step 1: Total volume and bucket sizing
    total_buy = sum(e.get("buy_vol", 0) for e in taker)
    total_sell = sum(e.get("sell_vol", 0) for e in taker)
    v_total = total_buy + total_sell
    if v_total <= 0:
        return AlgoBiasResult("vpin", "VPIN Flow Toxicity", "neutral", 0, 0,
                              explanation="Zero volume")

    n_buckets = 20
    v_bucket = v_total / n_buckets

    # Step 2-3: Fill volume-synchronized buckets from hourly bars
    buckets_buy: list[float] = []
    buckets_sell: list[float] = []
    cur_buy = 0.0
    cur_sell = 0.0
    cur_vol = 0.0

    for entry in taker:
        bv = entry.get("buy_vol", 0)
        sv = entry.get("sell_vol", 0)
        bar_vol = bv + sv
        if bar_vol <= 0:
            continue

        remaining = bar_vol
        buy_frac = bv / bar_vol if bar_vol > 0 else 0.5

        while remaining > 0:
            space = v_bucket - cur_vol
            fill = min(remaining, space)
            cur_buy += fill * buy_frac
            cur_sell += fill * (1 - buy_frac)
            cur_vol += fill
            remaining -= fill

            if cur_vol >= v_bucket - 1e-10:
                buckets_buy.append(cur_buy)
                buckets_sell.append(cur_sell)
                cur_buy = 0.0
                cur_sell = 0.0
                cur_vol = 0.0

    if not buckets_buy:
        return AlgoBiasResult("vpin", "VPIN Flow Toxicity", "neutral", 0, 0,
                              explanation="Could not form volume buckets")

    # Step 4-5: VPIN = mean of |buy - sell| / bucket_volume
    order_imbalances = [abs(b - s) for b, s in zip(buckets_buy, buckets_sell)]
    vpin = sum(oi / v_bucket for oi in order_imbalances) / len(order_imbalances)
    vpin = _clamp(vpin, 0, 1)

    # Step 6: Signed imbalance for direction
    signed_oi = sum(b - s for b, s in zip(buckets_buy, buckets_sell))
    median_vol = sorted(order_imbalances)[len(order_imbalances) // 2] if order_imbalances else 1
    signed_norm = math.tanh(signed_oi / max(median_vol, 1))

    # Step 7: Score
    score = _clamp(signed_norm * vpin * 100, -100, 100)

    # Step 8: Confidence
    confidence = _clamp(vpin * 120, 0, 95)

    direction = _direction_from_score(score)
    explanation = f"VPIN={vpin:.3f}, informed {'buying' if signed_oi > 0 else 'selling'}"
    if vpin >= 0.7:
        explanation += " — HIGH toxicity (institutional flow detected)"

    return AlgoBiasResult(
        "vpin", "VPIN Flow Toxicity", direction,
        round(score, 2), round(confidence, 1),
        {"vpin": round(vpin, 4), "signed_imbalance": round(signed_norm, 3),
         "n_buckets_filled": len(buckets_buy)},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 2: Funding Rate Ornstein-Uhlenbeck Mean Reversion
# ---------------------------------------------------------------------------

def algo_funding_ou(raw: dict) -> AlgoBiasResult:
    """Model funding as OU process; z-score for contrarian signal."""
    history = raw.get("funding_history", [])
    rates = [e.get("rate", 0) for e in history if e.get("rate") is not None]
    if len(rates) < 5:
        return AlgoBiasResult("funding_ou", "Funding Rate OU", "neutral", 0, 0,
                              explanation="Insufficient funding history")

    current = raw.get("funding_current", rates[-1])

    # Step 2: Estimate OU parameters via OLS — ΔX = a + b*X
    x_t = rates[:-1]
    dx_t = [rates[i + 1] - rates[i] for i in range(len(rates) - 1)]
    n = len(x_t)
    dt = 1.0  # normalized (each step = 1 funding interval ≈ 8h)

    sum_x = sum(x_t)
    sum_dx = sum(dx_t)
    sum_xx = sum(x * x for x in x_t)
    sum_xdx = sum(x * d for x, d in zip(x_t, dx_t))

    denom = n * sum_xx - sum_x ** 2
    if abs(denom) < 1e-30:
        return AlgoBiasResult("funding_ou", "Funding Rate OU", "neutral", 0, 0,
                              explanation="Degenerate funding data (constant)")

    b = (n * sum_xdx - sum_x * sum_dx) / denom
    a = (sum_dx - b * sum_x) / n

    # OU parameters
    theta = -b / dt if b < 0 else 0.01  # mean reversion speed
    mu = -a / b if abs(b) > 1e-15 else _safe_mean(rates)  # long-term mean

    # Residuals for volatility
    residuals = [dx_t[i] - (a + b * x_t[i]) for i in range(n)]
    sigma_e = _safe_std(residuals)
    sigma_ou = sigma_e * math.sqrt(2 * theta / max(1 - math.exp(-2 * theta * dt), 0.001))

    # Step 3: Half-life
    half_life = math.log(2) / max(theta, 0.001)

    # Step 4: Z-score of current deviation
    stationary_std = sigma_ou / math.sqrt(2 * max(theta, 0.001))
    z = (current - mu) / max(stationary_std, 1e-8)

    # Step 5: Cross-exchange spread z-score
    bybit = raw.get("bybit_rate")
    okx = raw.get("okx_rate")
    exchange_rates = [r for r in [current, bybit, okx] if r is not None]
    if len(exchange_rates) >= 2:
        spread = max(exchange_rates) - min(exchange_rates)
        z_spread = (spread - 0.0002) / max(0.0003, 0.0001)
    else:
        z_spread = 0.0

    # Step 7: Score (contrarian: positive z → bearish)
    score = -math.tanh(z) * min(100, abs(z) * 35)
    score *= (1 + 0.3 * min(1, max(z_spread, 0)))
    score = _clamp(score, -100, 100)

    # Step 8: Confidence
    confidence = _clamp(abs(z) * 30 + max(z_spread, 0) * 15, 0, 90)

    direction = _direction_from_score(score, threshold=8)
    explanation = f"Funding z={z:.2f} (μ={mu:.6f}), half-life={half_life:.1f} intervals"
    if abs(z) > 2:
        side = "longs" if z > 0 else "shorts"
        explanation += f" — {side} overleveraged, mean reversion expected"

    return AlgoBiasResult(
        "funding_ou", "Funding Rate OU", direction,
        round(score, 2), round(confidence, 1),
        {"z_score": round(z, 3), "theta": round(theta, 4), "mu": round(mu, 7),
         "half_life": round(half_life, 1), "z_spread": round(z_spread, 2),
         "current_rate": current},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 3: Options Greeks Flow — Dealer Gamma Exposure
# ---------------------------------------------------------------------------

def algo_options_greeks(raw: dict) -> AlgoBiasResult:
    """GEX regime + P/C z-score + max pain gravity + skew + DVOL."""
    opts = raw.get("options", {})
    dvol = raw.get("dvol", 0)

    if not opts or opts.get("pc_ratio") is None:
        return AlgoBiasResult("options_greeks", "Options Greeks Flow", "neutral", 0, 0,
                              explanation="No options data available")

    spot = opts.get("spot_price", 0) or raw.get("mark_price", 1)
    net_gex = opts.get("net_gex", 0)
    pc_ratio = opts.get("pc_ratio", 0.85)
    max_pain_dist = opts.get("max_pain_distance_pct", 0)
    skew_25d = opts.get("skew_25d") or 0

    # Sub-model A: GEX regime
    gex_norm = net_gex / max(spot, 1)
    # Use recent price direction from candles
    h1 = raw.get("candles_h1", [])
    recent_return = 0
    if len(h1) >= 2:
        recent_return = (h1[-1].get("close", h1[-1]) - h1[-2].get("close", h1[-2])) if isinstance(h1[-1], dict) else 0
        if isinstance(h1[-1], dict):
            c1 = h1[-1].get("close", 0)
            c0 = h1[-2].get("close", 0)
            recent_return = (c1 - c0) / max(c0, 1) if c0 else 0

    sign_ret = 1 if recent_return > 0 else (-1 if recent_return < 0 else 0)

    if net_gex < 0:  # Short gamma — amplifying (trending)
        gex_score = sign_ret * 10
    else:  # Long gamma — dampening (mean-reverting)
        gex_score = -sign_ret * 10
    gex_score *= min(2, abs(gex_norm) / max(0.001, 0.001))
    gex_score = _clamp(gex_score, -10, 10)

    # Sub-model B: P/C ratio z-score (BTC mean ≈ 0.85, std ≈ 0.20)
    z_pc = (pc_ratio - 0.85) / 0.20
    pc_score = _clamp(-math.tanh(z_pc) * 10, -10, 10)

    # Sub-model C: Max pain gravity
    gravity_score = _clamp(-math.tanh(max_pain_dist / 3) * 8, -10, 10)

    # Sub-model D: 25-delta skew
    skew_score = _clamp(-math.tanh(skew_25d / 8) * 8, -10, 10)

    # Sub-model E: DVOL
    dvol_score = _clamp(-math.tanh((dvol - 65) / 20) * 5, -10, 10) if dvol > 0 else 0

    # Weighted combination
    weights = {"gex": 3, "pc": 2, "max_pain": 2, "skew": 2, "dvol": 1}
    sub_scores = {"gex": gex_score, "pc": pc_score, "max_pain": gravity_score,
                  "skew": skew_score, "dvol": dvol_score}
    total_w = sum(weights.values())
    raw_score = sum(sub_scores[k] * weights[k] for k in weights) / total_w
    score = _clamp(raw_score * 10, -100, 100)

    # Confidence: count agreeing sub-signals
    final_sign = 1 if score > 0 else (-1 if score < 0 else 0)
    agreeing = sum(1 for s in sub_scores.values() if (s > 0.5) == (final_sign > 0))
    confidence = _clamp(agreeing * 18 + abs(score) / 100 * 15, 0, 85)

    direction = _direction_from_score(score)
    gex_label = "short-γ (amplifying)" if net_gex < 0 else "long-γ (dampening)"
    explanation = f"GEX={gex_label}, P/C={pc_ratio:.2f}, Skew={skew_25d:.1f}, MaxPain dist={max_pain_dist:.1f}%"

    return AlgoBiasResult(
        "options_greeks", "Options Greeks Flow", direction,
        round(score, 2), round(confidence, 1),
        {k: round(v, 2) for k, v in sub_scores.items()} | {
            "net_gex": net_gex, "pc_ratio": pc_ratio, "dvol": dvol},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 4: Kyle's Lambda + Amihud Illiquidity
# ---------------------------------------------------------------------------

def algo_kyle_amihud(raw: dict) -> AlgoBiasResult:
    """Price impact + illiquidity for directional flow amplification."""
    h1 = raw.get("candles_h1", [])
    taker = raw.get("taker_volume", [])
    l2 = raw.get("l2", {})

    # Need candles as dicts
    candles = []
    for c in h1:
        if isinstance(c, dict):
            candles.append(c)
        elif hasattr(c, "close"):
            candles.append({"open": c.open, "high": c.high, "low": c.low,
                            "close": c.close, "volume": c.volume})

    min_len = min(len(candles), len(taker))
    if min_len < 6:
        return AlgoBiasResult("kyle_amihud", "Kyle's Lambda + Amihud", "neutral", 0, 0,
                              explanation="Insufficient data for price impact analysis")

    # Align to last min_len points
    candles = candles[-min_len:]
    taker = taker[-min_len:]

    # Step 1: Hourly returns
    returns = []
    for i in range(1, len(candles)):
        prev_c = candles[i - 1].get("close", 0)
        cur_c = candles[i].get("close", 0)
        returns.append((cur_c - prev_c) / prev_c if prev_c else 0)

    # Step 2: Signed order flow
    order_flow = []
    for e in taker[1:]:  # align with returns (skip first)
        order_flow.append(e.get("buy_vol", 0) - e.get("sell_vol", 0))

    n = min(len(returns), len(order_flow))
    if n < 4:
        return AlgoBiasResult("kyle_amihud", "Kyle's Lambda + Amihud", "neutral", 0, 0,
                              explanation="Too few data points")

    returns = returns[-n:]
    order_flow = order_flow[-n:]

    # Step 3: Kyle's lambda (full period)
    var_of = _variance(order_flow)
    cov_r_of = _covariance(returns, order_flow)
    lam = cov_r_of / var_of if var_of > 0 else 0

    # Step 4: Lambda over sub-windows for z-score
    half = n // 2
    if half >= 3:
        lam_recent = _covariance(returns[-half:], order_flow[-half:]) / max(_variance(order_flow[-half:]), 1e-15)
        lam_older = _covariance(returns[:half], order_flow[:half]) / max(_variance(order_flow[:half]), 1e-15)
        lambdas = [lam_recent, lam_older, lam]
        z_lam = (lam_recent - _safe_mean(lambdas)) / _safe_std(lambdas)
    else:
        z_lam = 0.0

    # Step 5: Amihud
    illiq_vals = []
    for i in range(n):
        total_vol = taker[i + 1].get("buy_vol", 0) + taker[i + 1].get("sell_vol", 0)
        if total_vol > 0:
            illiq_vals.append(abs(returns[i]) / total_vol)
    illiq_avg = _safe_mean(illiq_vals)

    # Step 6: L2 depth
    depth_imb = l2.get("imbalance", 0) if l2 else 0
    spread_pct = l2.get("spread_pct", 0.01) if l2 else 0.01
    spread_penalty = min(1, spread_pct / 0.02)

    # Step 7: Score
    flow_6h = sum(order_flow[-min(6, n):])
    flow_dir = 1 if flow_6h > 0 else (-1 if flow_6h < 0 else 0)
    score = math.tanh(flow_dir * max(abs(z_lam), 0.5)) * 50 + depth_imb * 30
    if illiq_avg > 0:
        illiq_boost = min(0.5, illiq_avg * 1e6)
        score *= (1 + illiq_boost)
    score = _clamp(score, -100, 100)

    # Step 8: Confidence
    confidence = _clamp(abs(z_lam) * 20 + abs(depth_imb) * 25 + (1 - spread_penalty) * 15, 0, 80)

    direction = _direction_from_score(score)
    explanation = f"λ={lam:.2e}, z(λ)={z_lam:.2f}, L2 imb={depth_imb:.2f}, Amihud={illiq_avg:.2e}"

    return AlgoBiasResult(
        "kyle_amihud", "Kyle's Lambda + Amihud", direction,
        round(score, 2), round(confidence, 1),
        {"lambda": round(lam, 8), "z_lambda": round(z_lam, 3),
         "amihud": round(illiq_avg, 10), "depth_imbalance": round(depth_imb, 3),
         "flow_direction": flow_dir},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 5: Hurst Exponent Regime Detector
# ---------------------------------------------------------------------------

def algo_hurst(raw: dict) -> AlgoBiasResult:
    """R/S analysis for trending vs mean-reverting regime detection."""
    h1 = raw.get("candles_h1", [])

    prices = []
    for c in h1:
        if isinstance(c, dict):
            prices.append(c.get("close", 0))
        elif hasattr(c, "close"):
            prices.append(c.close)

    if len(prices) < 20:
        return AlgoBiasResult("hurst", "Hurst Exponent", "neutral", 0, 50,
                              {"hurst": 0.5}, "Insufficient data for R/S analysis")

    # Step 1: Log returns
    log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))
                   if prices[i - 1] > 0 and prices[i] > 0]

    if len(log_returns) < 12:
        return AlgoBiasResult("hurst", "Hurst Exponent", "neutral", 0, 50,
                              {"hurst": 0.5}, "Too few returns for R/S")

    # Step 2: R/S analysis for multiple sub-series lengths
    ns = [n for n in [6, 8, 12, 16, 24] if n <= len(log_returns) // 2]
    if len(ns) < 2:
        ns = [n for n in [4, 6, 8, 10] if n <= len(log_returns) // 2]

    log_ns = []
    log_rs = []

    for n in ns:
        k = len(log_returns) // n
        if k < 1:
            continue
        rs_values = []
        for j in range(k):
            sub = log_returns[j * n:(j + 1) * n]
            mean_sub = _safe_mean(sub)
            # Cumulative deviation
            y = []
            cum = 0
            for r in sub:
                cum += (r - mean_sub)
                y.append(cum)
            r_range = max(y) - min(y) if y else 0
            s = _safe_std(sub)
            if s > 1e-15:
                rs_values.append(r_range / s)
        if rs_values:
            log_ns.append(math.log(n))
            log_rs.append(math.log(_safe_mean(rs_values)))

    # Step 3: Linear regression for H
    if len(log_ns) < 2:
        hurst = 0.5
    else:
        n_pts = len(log_ns)
        sum_x = sum(log_ns)
        sum_y = sum(log_rs)
        sum_xy = sum(x * y for x, y in zip(log_ns, log_rs))
        sum_xx = sum(x * x for x in log_ns)
        denom = n_pts * sum_xx - sum_x ** 2
        hurst = (n_pts * sum_xy - sum_x * sum_y) / denom if abs(denom) > 1e-15 else 0.5
        hurst = _clamp(hurst, 0.01, 0.99)

    # Step 4: Regime classification
    if hurst > 0.65:
        regime = "trending"
    elif hurst > 0.55:
        regime = "mild_trending"
    elif hurst > 0.45:
        regime = "random"
    elif hurst > 0.35:
        regime = "mild_reverting"
    else:
        regime = "mean_reverting"

    # Step 5: Directional component
    recent_sum = sum(log_returns[-min(12, len(log_returns)):])
    trend_dir = 1 if recent_sum > 0 else (-1 if recent_sum < 0 else 0)

    if hurst > 0.5:
        score = trend_dir * (hurst - 0.5) * 200  # momentum
    else:
        score = -trend_dir * (0.5 - hurst) * 200  # contrarian
    score = _clamp(score, -60, 60)

    # Step 6: Confidence
    confidence = _clamp(abs(hurst - 0.5) * 150, 0, 70)

    direction = _direction_from_score(score, threshold=8)
    explanation = f"H={hurst:.3f} ({regime})"
    if hurst > 0.55:
        explanation += " — momentum strategies favored"
    elif hurst < 0.45:
        explanation += " — mean-reversion strategies favored"

    return AlgoBiasResult(
        "hurst", "Hurst Exponent", direction,
        round(score, 2), round(confidence, 1),
        {"hurst": round(hurst, 4), "regime": regime,
         "recent_trend": "up" if trend_dir > 0 else ("down" if trend_dir < 0 else "flat")},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 6: Liquidation Cascade Probability Model
# ---------------------------------------------------------------------------

def algo_liquidation(raw: dict) -> AlgoBiasResult:
    """Logistic probability model for cascade detection."""
    liquidations = raw.get("liquidations", [])
    oi_history = raw.get("oi_history", [])
    funding_current = raw.get("funding_current", 0)
    funding_history = raw.get("funding_history", [])
    h1 = raw.get("candles_h1", [])

    # Current price
    current_price = 0
    if h1:
        last = h1[-1]
        current_price = last.get("close", 0) if isinstance(last, dict) else (last.close if hasattr(last, "close") else 0)
    if not current_price:
        current_price = raw.get("mark_price", 0)

    # Step 1: OI percentile
    oi_values = [e.get("oi_usd", 0) for e in oi_history if e.get("oi_usd")]
    current_oi = oi_values[-1] if oi_values else 0
    oi_pct = _percentile_rank(current_oi, oi_values) if oi_values else 50

    # Step 2: Funding z-score
    rates = [e.get("rate", 0) for e in funding_history if e.get("rate") is not None]
    funding_std = _safe_std(rates) if rates else 0.0001
    funding_z = funding_current / funding_std if funding_std > 1e-8 else 0

    # Step 3: Liquidation clustering by $100 price bins
    # Use backtest timestamp when available, fall back to real time for live
    now = raw.get("current_timestamp") or time.time() * 1000
    recent_30m = [e for e in liquidations
                  if now - e.get("timestamp", 0) < 30 * 60 * 1000]

    cluster_volume: dict[int, float] = {}
    buy_liq = 0.0   # shorts liquidated
    sell_liq = 0.0   # longs liquidated
    for e in recent_30m:
        usd = e.get("qty_usd", 0)
        price = e.get("price", 0)
        if price > 0:
            bin_key = int(price // 100) * 100
            cluster_volume[bin_key] = cluster_volume.get(bin_key, 0) + usd
        if e.get("side") == "buy":
            buy_liq += usd
        else:
            sell_liq += usd

    max_cluster = max(cluster_volume.values()) if cluster_volume else 0
    recent_volume = buy_liq + sell_liq

    # Cluster proximity
    if current_price > 0 and cluster_volume:
        max_bin = max(cluster_volume, key=cluster_volume.get)
        cluster_proximity = abs(current_price - max_bin) / current_price * 100
    else:
        cluster_proximity = 10  # far away

    # Step 5: Pre-cascade probability (logistic)
    z_logistic = (0.03 * (oi_pct - 50)
                  + 0.8 * abs(funding_z)
                  + 2.0 * (1 if max_cluster > 10_000_000 else 0)
                  - 1.5 * cluster_proximity)
    p_cascade = 1 / (1 + math.exp(-z_logistic))

    # Step 6: Active cascade detection
    cascade_active = recent_volume > 50_000_000

    # Step 7-8: Direction and score
    if cascade_active:
        # Exhaustion: dominant side liquidated → reversal
        if sell_liq > buy_liq:
            direction = "bullish"  # longs liquidated → exhaustion → bounce
            score = 60 + min(40, recent_volume / 5e7 * 20)
        else:
            direction = "bearish"
            score = -(60 + min(40, recent_volume / 5e7 * 20))
        explanation = f"CASCADE ACTIVE: ${recent_volume / 1e6:.0f}M in 30min"
        if sell_liq > buy_liq:
            explanation += f" — longs liquidated, exhaustion → bounce expected"
        else:
            explanation += f" — shorts liquidated, exhaustion → drop expected"
    elif p_cascade > 0.6:
        # Pre-cascade anticipation
        if funding_current > 0:
            direction = "bearish"
            score = -(p_cascade * 60)
            explanation = f"Pre-cascade P={p_cascade:.0%}, longs vulnerable (funding={funding_current:.5f})"
        else:
            direction = "bullish"
            score = p_cascade * 60
            explanation = f"Pre-cascade P={p_cascade:.0%}, shorts vulnerable (funding={funding_current:.5f})"
    else:
        direction = "neutral"
        score = 0
        explanation = f"P(cascade)={p_cascade:.0%}, no imminent threat"

    score = _clamp(score, -100, 100)

    # Step 9: Confidence
    confidence = _clamp(p_cascade * 80 + (30 if cascade_active else 0), 0, 90)

    return AlgoBiasResult(
        "liquidation", "Liquidation Cascade", direction,
        round(score, 2), round(confidence, 1),
        {"p_cascade": round(p_cascade, 3), "cascade_active": cascade_active,
         "recent_volume_usd": round(recent_volume), "oi_percentile": round(oi_pct, 1),
         "funding_z": round(funding_z, 2), "buy_liq_usd": round(buy_liq),
         "sell_liq_usd": round(sell_liq)},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 7: Volatility Regime — ATR Ratio + DVOL
# ---------------------------------------------------------------------------

def algo_vol_regime(raw: dict) -> AlgoBiasResult:
    """ATR ratio + DVOL + realized-vs-implied vol spread."""
    h4 = raw.get("candles_h4", [])
    dvol = raw.get("dvol", 0)

    candles = []
    for c in h4:
        if isinstance(c, dict):
            candles.append(c)
        elif hasattr(c, "close"):
            candles.append({"open": c.open, "high": c.high, "low": c.low,
                            "close": c.close, "volume": c.volume})

    if len(candles) < 15:
        return AlgoBiasResult("vol_regime", "Volatility Regime", "neutral", 0, 50,
                              {"vol_regime": "unknown"}, "Insufficient candle data")

    # Step 1: True Range
    true_ranges = []
    for i in range(1, len(candles)):
        h = candles[i].get("high", 0)
        l = candles[i].get("low", 0)
        prev_c = candles[i - 1].get("close", 0)
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        true_ranges.append(tr)

    # Step 2: ATR(14) via EMA, ATR(long) via SMA
    alpha = 2 / (14 + 1)
    atr_14 = true_ranges[0]
    for tr in true_ranges[1:min(14, len(true_ranges))]:
        atr_14 = alpha * tr + (1 - alpha) * atr_14
    for tr in true_ranges[14:]:
        atr_14 = alpha * tr + (1 - alpha) * atr_14

    long_period = min(100, len(true_ranges))
    atr_long = _safe_mean(true_ranges[-long_period:])

    # Step 3: ATR ratio
    ratio = atr_14 / max(atr_long, 1e-10)

    # Step 4: Realized volatility (annualized from H4)
    log_returns = []
    for i in range(1, len(candles)):
        c0 = candles[i - 1].get("close", 0)
        c1 = candles[i].get("close", 0)
        if c0 > 0 and c1 > 0:
            log_returns.append(math.log(c1 / c0))

    rv_14 = _safe_std(log_returns[-14:]) * math.sqrt(6 * 365) * 100 if len(log_returns) >= 14 else 0

    # Step 5: IV-RV spread
    vol_spread = dvol - rv_14 if dvol > 0 else 0

    # Step 6: Regime classification
    if ratio > 2.0 or (dvol > 100):
        vol_regime = "extreme"
    elif ratio > 1.3 or (dvol > 80):
        vol_regime = "high"
    elif ratio < 0.7 and (dvol < 40 or dvol == 0):
        vol_regime = "low"
    else:
        vol_regime = "normal"

    # Step 7: Directional component (weak)
    if len(log_returns) >= 6:
        ret_24h = sum(log_returns[-6:])
    else:
        ret_24h = 0

    if vol_regime == "low" and ratio < 0.6:
        score = 15  # compression → BTC breakouts are historically 60% upward
    elif vol_regime == "extreme":
        score = -math.copysign(25, ret_24h) if ret_24h != 0 else 0
    else:
        score = 0

    confidence = 50  # Regime indicator, not directional predictor

    direction = _direction_from_score(score, threshold=10)
    explanation = f"Regime={vol_regime}, ATR ratio={ratio:.2f}, RV={rv_14:.1f}%, DVOL={dvol:.0f}"
    if vol_spread != 0:
        explanation += f", IV-RV={vol_spread:+.1f}"

    return AlgoBiasResult(
        "vol_regime", "Volatility Regime", direction,
        round(score, 2), round(confidence, 1),
        {"vol_regime": vol_regime, "atr_ratio": round(ratio, 3),
         "atr_14": round(atr_14, 2), "atr_long": round(atr_long, 2),
         "realized_vol": round(rv_14, 1), "dvol": dvol,
         "iv_rv_spread": round(vol_spread, 1)},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 8: Order Flow Imbalance — Cont-Kukanov-Stoikov (2014)
# ---------------------------------------------------------------------------

def algo_ofi(raw: dict) -> AlgoBiasResult:
    """L2 depth + taker aggression + price-flow regression."""
    l2 = raw.get("l2", {})
    taker = raw.get("taker_volume", [])
    h1 = raw.get("candles_h1", [])

    if not l2 or len(taker) < 4:
        return AlgoBiasResult("ofi", "Order Flow Imbalance", "neutral", 0, 0,
                              explanation="Insufficient order flow data")

    # Step 1: Static OFI from L2
    bid_wall = l2.get("bid_wall_usd", 0)
    ask_wall = l2.get("ask_wall_usd", 0)
    total_walls = bid_wall + ask_wall
    static_ofi = (bid_wall - ask_wall) / total_walls if total_walls > 0 else 0

    # Step 2: Dynamic OFI from taker (last 4h)
    recent = taker[-4:]
    net_takers = [e.get("buy_vol", 0) - e.get("sell_vol", 0) for e in recent]
    cumulative_of = sum(net_takers)
    avg_vol = _safe_mean([e.get("buy_vol", 0) + e.get("sell_vol", 0) for e in recent])
    normalized_of = cumulative_of / max(avg_vol, 1)

    # Step 3: Taker acceleration
    if len(taker) >= 4:
        accel = net_takers[-1] - net_takers[0]
        accel_norm = accel / max(avg_vol, 1)
    else:
        accel_norm = 0

    # Step 4: Price-flow regression β
    candles_list = []
    for c in h1:
        if isinstance(c, dict):
            candles_list.append(c)
        elif hasattr(c, "close"):
            candles_list.append({"close": c.close})

    beta = 0
    min_n = min(len(candles_list) - 1, len(taker) - 1, 6)
    if min_n >= 3:
        rets = [(candles_list[-(i)].get("close", 0) - candles_list[-(i + 1)].get("close", 0))
                / max(candles_list[-(i + 1)].get("close", 1), 1)
                for i in range(1, min_n + 1)]
        flows = [taker[-(i)].get("buy_vol", 0) - taker[-(i)].get("sell_vol", 0)
                 for i in range(1, min_n + 1)]
        rets.reverse()
        flows.reverse()
        var_f = _variance(flows)
        beta = _covariance(rets, flows) / var_f if var_f > 1e-15 else 0

    # Step 5: Spread quality
    spread_pct = l2.get("spread_pct", 0.01)
    spread_quality = max(0, 1 - spread_pct / 0.03)

    # Step 6: Combined score
    score = (static_ofi * 25 + math.tanh(normalized_of) * 45 + math.tanh(accel_norm) * 20)
    score *= (0.5 + 0.5 * spread_quality)
    if beta > 0:
        score *= (1 + min(0.5, abs(beta) * 500))
    score = _clamp(score, -100, 100)

    # Step 8: Confidence
    confidence = _clamp(abs(static_ofi) * 30 + abs(normalized_of) * 15
                        + spread_quality * 15 + (10 if beta > 0 else 0), 0, 80)

    direction = _direction_from_score(score)
    explanation = (f"L2 imb={static_ofi:.2f}, Taker flow={normalized_of:+.2f}×avg, "
                   f"Accel={accel_norm:+.2f}, β={beta:.2e}")

    return AlgoBiasResult(
        "ofi", "Order Flow Imbalance", direction,
        round(score, 2), round(confidence, 1),
        {"static_ofi": round(static_ofi, 3), "normalized_flow": round(normalized_of, 3),
         "acceleration": round(accel_norm, 3), "beta": round(beta, 8),
         "spread_quality": round(spread_quality, 2)},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 9: Smart Money vs Retail Divergence
# ---------------------------------------------------------------------------

def algo_smart_retail(raw: dict) -> AlgoBiasResult:
    """Contrarian divergence between top traders and retail."""
    top = raw.get("top_trader_ratio", [])
    glob = raw.get("global_ratio", [])

    if len(top) < 6 or len(glob) < 6:
        return AlgoBiasResult("smart_retail", "Smart vs Retail", "neutral", 0, 0,
                              explanation="Insufficient positioning data")

    # Step 1: Current positioning
    # Binance API returns long_pct as decimal ratio (0.58 = 58%), convert to 0-100
    smart_long_raw = top[-1].get("long_pct", 50)
    retail_long_raw = glob[-1].get("long_pct", 50)
    smart_long = smart_long_raw * 100 if smart_long_raw <= 1 else smart_long_raw
    retail_long = retail_long_raw * 100 if retail_long_raw <= 1 else retail_long_raw
    smart_bias = smart_long - 50
    retail_bias = retail_long - 50

    # Step 2: Z-scores vs 48h history (convert all to 0-100 scale)
    smart_longs = [e.get("long_pct", 50) * 100 if e.get("long_pct", 50) <= 1 else e.get("long_pct", 50) for e in top]
    retail_longs = [e.get("long_pct", 50) * 100 if e.get("long_pct", 50) <= 1 else e.get("long_pct", 50) for e in glob]

    z_smart = (smart_long - _safe_mean(smart_longs)) / _safe_std(smart_longs) if _safe_std(smart_longs) > 0.5 else 0
    z_retail = (retail_long - _safe_mean(retail_longs)) / _safe_std(retail_longs) if _safe_std(retail_longs) > 0.5 else 0

    # Step 3: Divergence
    raw_div = smart_bias - retail_bias
    z_div = z_smart - z_retail

    # Step 4: Trend in divergence
    idx_24h = min(24, len(top) - 1, len(glob) - 1)
    if idx_24h > 0:
        top_24h = top[-idx_24h].get("long_pct", 50)
        glob_24h = glob[-idx_24h].get("long_pct", 50)
        top_24h_pct = top_24h * 100 if top_24h <= 1 else top_24h
        glob_24h_pct = glob_24h * 100 if glob_24h <= 1 else glob_24h
        div_24h_ago = top_24h_pct - glob_24h_pct
        div_current = smart_long - retail_long
        div_trend = div_current - div_24h_ago
    else:
        div_trend = 0

    # Step 6: Score
    score = math.tanh(z_div / 2) * 70 + math.tanh(div_trend / 5) * 20
    score = _clamp(score, -100, 100)

    # Step 7: Confidence
    confidence = _clamp(abs(z_div) * 25 + abs(div_trend) / 10 * 10
                        + (15 if abs(raw_div) > 8 else 0), 0, 85)

    direction = _direction_from_score(score)

    if raw_div > 8:
        label = "Smart LONG / Retail SHORT — accumulation"
    elif raw_div < -8:
        label = "Smart SHORT / Retail LONG — distribution"
    else:
        label = f"Divergence={raw_div:+.1f}pp"
    explanation = f"{label}, z(div)={z_div:.2f}, trend={div_trend:+.1f}pp/24h"

    return AlgoBiasResult(
        "smart_retail", "Smart vs Retail", direction,
        round(score, 2), round(confidence, 1),
        {"smart_long_pct": round(smart_long, 1), "retail_long_pct": round(retail_long, 1),
         "raw_divergence": round(raw_div, 1), "z_divergence": round(z_div, 3),
         "div_trend_24h": round(div_trend, 1)},
        explanation,
    )


# ---------------------------------------------------------------------------
# Algorithm 10: Bayesian Sentiment Fusion
# ---------------------------------------------------------------------------

def algo_bayesian_sentiment(raw: dict) -> AlgoBiasResult:
    """Bayesian posterior P(bullish | multi-signal evidence)."""
    fng = raw.get("fear_greed", {})
    funding = raw.get("funding_current", 0)
    glob = raw.get("global_ratio", [])
    opts = raw.get("options", {})
    premium = raw.get("premium_pct", 0)
    cot = raw.get("cot", {})

    # Step 1: Prior from COT (contrarian)
    cot_pct = cot.get("percentile") if cot else None
    if cot_pct is not None:
        p_bull = _clamp(1 - cot_pct / 100, 0.1, 0.9)
    else:
        p_bull = 0.5
    p_bear = 1 - p_bull

    # Step 2: Likelihood ratios

    # Signal A: Fear & Greed
    fng_val = fng.get("value", 50) if fng else 50
    if fng_val < 20:
        lr_a = 2.5
    elif fng_val < 35:
        lr_a = 1.5
    elif fng_val < 65:
        lr_a = 1.0
    elif fng_val < 80:
        lr_a = 0.7
    else:
        lr_a = 0.4

    # Signal B: Funding rate
    if funding < -0.0003:
        lr_b = 2.2
    elif funding < 0:
        lr_b = 1.3
    elif funding < 0.0003:
        lr_b = 0.8
    else:
        lr_b = 0.45

    # Signal C: Retail positioning (convert decimal ratio to 0-100)
    retail_long_raw = glob[-1].get("long_pct", 50) if glob else 50
    retail_long = retail_long_raw * 100 if retail_long_raw <= 1 else retail_long_raw
    if retail_long > 65:
        lr_c = 0.5
    elif retail_long > 55:
        lr_c = 0.75
    elif retail_long > 45:
        lr_c = 1.0
    elif retail_long > 35:
        lr_c = 1.3
    else:
        lr_c = 2.0

    # Signal D: P/C ratio
    pc = opts.get("pc_ratio", 0.85) if opts else 0.85
    if pc > 1.3:
        lr_d = 2.0
    elif pc > 1.0:
        lr_d = 1.3
    elif pc > 0.7:
        lr_d = 0.8
    else:
        lr_d = 0.5

    # Signal E: Futures premium
    if premium > 0.3:
        lr_e = 1.5
    elif premium > 0:
        lr_e = 1.1
    elif premium > -0.3:
        lr_e = 0.9
    else:
        lr_e = 0.6

    # Step 3: Bayesian update
    combined_lr = lr_a * lr_b * lr_c * lr_d * lr_e
    posterior_odds = (p_bull / max(p_bear, 0.01)) * combined_lr
    p_posterior = posterior_odds / (1 + posterior_odds)
    p_posterior = _clamp(p_posterior, 0.01, 0.99)

    # Step 5: Score
    score = _clamp((p_posterior - 0.5) * 200, -100, 100)

    # Step 6: Confidence
    confidence = _clamp(abs(p_posterior - 0.5) * 180, 0, 90)

    # Direction
    if p_posterior > 0.65:
        direction = "bullish"
    elif p_posterior < 0.35:
        direction = "bearish"
    else:
        direction = "neutral"

    explanation = (f"P(bull)={p_posterior:.0%}, "
                   f"FnG={fng_val}, Funding={'+'if funding>0 else ''}{funding:.5f}, "
                   f"Retail={retail_long:.0f}%L, P/C={pc:.2f}")

    return AlgoBiasResult(
        "bayesian_sentiment", "Bayesian Sentiment", direction,
        round(score, 2), round(confidence, 1),
        {"p_posterior_bull": round(p_posterior, 4),
         "prior_bull": round(p_bull, 3),
         "lr_fng": lr_a, "lr_funding": lr_b, "lr_retail": lr_c,
         "lr_pc": lr_d, "lr_premium": lr_e,
         "combined_lr": round(combined_lr, 4),
         "fng_value": fng_val},
        explanation,
    )


# ---------------------------------------------------------------------------
# META-ALGORITHM: Adaptive Ensemble
# ---------------------------------------------------------------------------

BASE_WEIGHTS = {
    "vpin": 0.14,
    "funding_ou": 0.12,
    "options_greeks": 0.12,
    "kyle_amihud": 0.10,
    "hurst": 0.05,
    "liquidation": 0.13,
    "vol_regime": 0.05,
    "ofi": 0.12,
    "smart_retail": 0.09,
    "bayesian_sentiment": 0.08,
}

# Algos categorized for Hurst-adaptive weighting
MOMENTUM_ALGOS = {"vpin", "ofi", "liquidation", "kyle_amihud"}
REVERSION_ALGOS = {"funding_ou", "bayesian_sentiment", "smart_retail"}


def compute_composite_bias(results: list[AlgoBiasResult]) -> CompositeBias:
    """Combine all algo results into a single composite bias."""

    # Filter out algos with zero confidence (no data)
    valid = [r for r in results if r.confidence > 0]
    if not valid:
        return CompositeBias("neutral", 0, 0, 0, 0, "random", "normal", 0,
                             [r.to_dict() for r in results], time.time())

    # Extract special algo outputs
    hurst_result = next((r for r in results if r.algo_id == "hurst"), None)
    vol_result = next((r for r in results if r.algo_id == "vol_regime"), None)

    hurst_val = hurst_result.components.get("hurst", 0.5) if hurst_result else 0.5
    vol_regime = vol_result.components.get("vol_regime", "normal") if vol_result else "normal"

    # Capture base weights for meta output
    base_weights_snapshot = dict(BASE_WEIGHTS)

    # Step 3: Hurst-adaptive weight modification
    weights = dict(BASE_WEIGHTS)
    hurst_effect = "none"

    if hurst_val > 0.55:
        boost = 1 + (hurst_val - 0.5) * 2
        reduce = 1 - (hurst_val - 0.5)
        for algo_id in MOMENTUM_ALGOS:
            if algo_id in weights:
                weights[algo_id] *= boost
        for algo_id in REVERSION_ALGOS:
            if algo_id in weights:
                weights[algo_id] *= max(reduce, 0.3)
        hurst_effect = f"trending (H={hurst_val:.3f}): momentum algos x{boost:.2f}, reversion algos x{max(reduce, 0.3):.2f}"
    elif hurst_val < 0.45:
        boost = 1 + (0.5 - hurst_val) * 2
        reduce = 1 - (0.5 - hurst_val)
        for algo_id in REVERSION_ALGOS:
            if algo_id in weights:
                weights[algo_id] *= boost
        for algo_id in MOMENTUM_ALGOS:
            if algo_id in weights:
                weights[algo_id] *= max(reduce, 0.3)
        hurst_effect = f"mean-reverting (H={hurst_val:.3f}): reversion algos x{boost:.2f}, momentum algos x{max(reduce, 0.3):.2f}"
    else:
        hurst_effect = f"random walk (H={hurst_val:.3f}): no weight adjustment"

    # Capture weights after Hurst, before vol
    weights_after_hurst = dict(weights)

    # Step 4: Vol regime modifier
    vol_effect = "none"
    if vol_regime == "low":
        for a in REVERSION_ALGOS:
            weights[a] = weights.get(a, 0) * 1.10
        for a in MOMENTUM_ALGOS:
            weights[a] = weights.get(a, 0) * 0.90
        vol_effect = "low vol: reversion +10%, momentum -10%"
    elif vol_regime == "high":
        for a in MOMENTUM_ALGOS:
            weights[a] = weights.get(a, 0) * 1.15
        for a in REVERSION_ALGOS:
            weights[a] = weights.get(a, 0) * 0.85
        vol_effect = "high vol: momentum +15%, reversion -15%"
    elif vol_regime == "extreme":
        vol_effect = "extreme vol: all confidences -25%"
    else:
        vol_effect = "normal vol: no adjustment"

    # Renormalize weights
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    # Step 5: Composite score (confidence-weighted)
    # Track per-algo contribution for meta output
    algo_contributions: dict[str, dict] = {}
    raw_score = 0.0
    for r in valid:
        w = weights.get(r.algo_id, 0.05)
        contribution = r.score * w * (r.confidence / 100)
        raw_score += contribution
        algo_contributions[r.algo_id] = {
            "base_weight": round(base_weights_snapshot.get(r.algo_id, 0.05) * 100, 1),
            "adjusted_weight": round(w * 100, 1),
            "weight_change": round((w - base_weights_snapshot.get(r.algo_id, 0.05)) * 100, 1),
            "contribution": round(contribution, 2),
            "category": "momentum" if r.algo_id in MOMENTUM_ALGOS else ("reversion" if r.algo_id in REVERSION_ALGOS else "regime"),
        }

    composite_score = _clamp(raw_score, -100, 100)

    # Step 6: Shannon entropy (agreement measure)
    n_bull = sum(1 for r in valid if r.direction == "bullish")
    n_bear = sum(1 for r in valid if r.direction == "bearish")
    n_neut = sum(1 for r in valid if r.direction == "neutral")
    total = len(valid)

    entropy = 0.0
    for count in [n_bull, n_bear, n_neut]:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    agreement_count = max(n_bull, n_bear, n_neut)

    # Step 7: Direction mapping
    if composite_score >= 60:
        direction = "strong_bullish"
    elif composite_score >= 25:
        direction = "bullish"
    elif composite_score <= -60:
        direction = "strong_bearish"
    elif composite_score <= -25:
        direction = "bearish"
    else:
        direction = "neutral"

    # Step 8: Composite confidence
    base_conf = sum(r.confidence * weights.get(r.algo_id, 0.05) for r in valid)
    if entropy < 0.8:
        entropy_adj = 15
    elif entropy > 1.3:
        entropy_adj = -20
    else:
        entropy_adj = 0

    vol_penalty = 0.75 if vol_regime == "extreme" else 1.0
    final_confidence = _clamp((base_conf + entropy_adj) * vol_penalty, 0, 95)

    # Regime label
    if hurst_val > 0.55:
        regime = "trending"
    elif hurst_val < 0.45:
        regime = "mean_reverting"
    else:
        regime = "random"

    # Build meta dict with full transparency
    meta = {
        "hurst_value": round(hurst_val, 4),
        "hurst_effect": hurst_effect,
        "vol_effect": vol_effect,
        "entropy_adj": entropy_adj,
        "vol_confidence_penalty": vol_penalty,
        "base_confidence": round(base_conf, 1),
        "algo_weights": algo_contributions,
        "momentum_algos": sorted(MOMENTUM_ALGOS),
        "reversion_algos": sorted(REVERSION_ALGOS),
    }

    return CompositeBias(
        direction=direction,
        score=composite_score,
        confidence=final_confidence,
        algo_count=len(valid),
        agreement_count=agreement_count,
        regime=regime,
        vol_regime=vol_regime,
        entropy=entropy,
        algos=[r.to_dict() for r in results],
        timestamp=time.time(),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ALL_ALGOS = [
    algo_vpin,
    algo_funding_ou,
    algo_options_greeks,
    algo_kyle_amihud,
    algo_hurst,
    algo_liquidation,
    algo_vol_regime,
    algo_ofi,
    algo_smart_retail,
    algo_bayesian_sentiment,
]


def run_algo_bias(raw_data: dict) -> CompositeBias:
    """Run all 10 algorithms and compute composite bias.

    Args:
        raw_data: dict assembled from API responses (see plan for schema)

    Returns:
        CompositeBias with full breakdown
    """
    results: list[AlgoBiasResult] = []
    for algo_fn in ALL_ALGOS:
        try:
            result = algo_fn(raw_data)
            results.append(result)
        except Exception as e:
            # Graceful degradation: algo fails → neutral with zero confidence
            algo_name = algo_fn.__name__.replace("algo_", "")
            results.append(AlgoBiasResult(
                algo_name, algo_name, "neutral", 0, 0,
                {"error": str(e)}, f"Algorithm error: {e}",
            ))

    return compute_composite_bias(results)
