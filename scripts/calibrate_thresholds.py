"""Threshold Calibration for Quant Algo Bias Alerts.

Runs all 9 active algorithms over 3 years of historical data, collects every
component metric, computes distribution statistics, and sweeps candidate
thresholds to find optimal values that maximize directional hit rate.

Usage:
    python3 scripts/calibrate_thresholds.py
    python3 scripts/calibrate_thresholds.py --start 2023-01-01 --end 2026-02-16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.quant.algo_bias import run_algo_bias, CompositeBias
from app.quant.alert_detector import _get_components, _get_direction, _n, _s, _b
from app.quant.quant_backtester import (
    fetch_all_historical,
    _date_to_ms,
    _slice_by_time,
    _get_price_at_ts,
    _assemble_raw_data,
    QUANT_BACKTEST_DIR,
)

HOUR = 3600_000


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(composite: CompositeBias) -> dict:
    """Extract all threshold-relevant metrics from a composite result."""
    vpin = _get_components(composite, "vpin")
    funding = _get_components(composite, "funding_ou")
    options = _get_components(composite, "options_greeks")
    kyle = _get_components(composite, "kyle_amihud")
    liq = _get_components(composite, "liquidation")
    vol = _get_components(composite, "vol_regime")
    smart = _get_components(composite, "smart_retail")
    bayes = _get_components(composite, "bayesian_sentiment")

    funding_z = _n(funding.get("z_score"))
    smart_div = _n(smart.get("raw_divergence"))

    # Individual algo directions (for per-algo direction derivation)
    def _algo_dir(algo_id: str) -> str:
        d = _get_direction(composite, algo_id)
        if "bullish" in d:
            return "bullish"
        if "bearish" in d:
            return "bearish"
        return "neutral"

    return {
        "vpin": _n(vpin.get("vpin")),
        "funding_z": funding_z,
        "abs_funding_z": abs(funding_z),
        "p_cascade": _n(liq.get("p_cascade")),
        "cascade_active": 1.0 if _b(liq.get("cascade_active")) else 0.0,
        "recent_liq_usd": _n(liq.get("recent_volume_usd")),
        "smart_div": smart_div,
        "abs_smart_div": abs(smart_div),
        "atr_ratio": _n(vol.get("atr_ratio")),
        "vol_regime": _s(vol.get("vol_regime")),
        "entropy": composite.entropy,
        "agreement_ratio": composite.agreement_count / max(composite.algo_count, 1),
        "composite_score": composite.score,
        "abs_composite_score": abs(composite.score),
        "composite_confidence": composite.confidence,
        "net_gex": _n(options.get("net_gex")),
        "pc_ratio": _n(options.get("pc_ratio")),
        "skew_25d": _n(options.get("skew_25d")),
        "p_bull": _n(bayes.get("p_posterior_bull")),
        "fng_value": _n(bayes.get("fng_value")),
        "z_lambda": _n(kyle.get("z_lambda")),
        "hurst": _n(_get_components(composite, "hurst").get("hurst")),
        # Composite direction (thresholded — often "neutral" when score < 25)
        "direction": composite.direction,
        # Individual algo directions (for accurate per-alert direction)
        "vpin_direction": _algo_dir("vpin"),
        "kyle_direction": _algo_dir("kyle_amihud"),
        "hurst_direction": _algo_dir("hurst"),
        "options_direction": _algo_dir("options_greeks"),
        "liq_direction": _algo_dir("liquidation"),
    }


# ---------------------------------------------------------------------------
# Threshold sweep config
# ---------------------------------------------------------------------------

THRESHOLD_SWEEPS = [
    {
        "metric_key": "vpin",
        "alert_name": "VPIN Toxic Flow",
        "current_threshold": 0.7,
        "comparison": "gt",
        "direction_source": "vpin_algo",  # VPIN's own direction (signed order imbalance)
        "percentiles": [50, 60, 70, 75, 80, 85, 90, 95],
    },
    {
        "metric_key": "abs_funding_z",
        "alert_name": "Funding Extreme",
        "current_threshold": 2.0,
        "comparison": "gt",
        "direction_source": "funding_z_sign",
        "percentiles": [70, 75, 80, 85, 90, 95],
    },
    {
        "metric_key": "p_cascade",
        "alert_name": "Cascade Forming",
        "current_threshold": 0.6,
        "comparison": "gt",
        "direction_source": "liq_algo",  # Liquidation algo's own direction (exhaustion logic)
        "percentiles": [70, 75, 80, 85, 90, 95],
    },
    {
        "metric_key": "abs_smart_div",
        "alert_name": "Smart/Retail Split",
        "current_threshold": 8.0,
        "comparison": "gt",
        "direction_source": "smart_div_sign",
        "percentiles": [70, 75, 80, 85, 90, 95],
    },
    {
        "metric_key": "atr_ratio",
        "alert_name": "Vol Compression",
        "current_threshold": 0.65,
        "comparison": "lt",
        "direction_source": "fixed_bullish",
        "percentiles": [5, 10, 15, 20, 25, 30],
    },
    {
        "metric_key": "entropy",
        "alert_name": "Strong Consensus",
        "current_threshold": 0.8,
        "comparison": "lt",
        "direction_source": "composite_score_sign",  # Sign of score (works even when |score| < 25)
        "percentiles": [5, 10, 15, 20, 25, 30],
    },
    {
        "metric_key": "abs_composite_score",
        "alert_name": "Extreme Composite",
        "current_threshold": 70.0,
        "comparison": "gt",
        "direction_source": "composite_score_sign",  # Sign of score
        "percentiles": [75, 80, 85, 90, 95, 99],
    },
    {
        "metric_key": "composite_confidence",
        "alert_name": "Extreme Confidence",
        "current_threshold": 75.0,
        "comparison": "gt",
        "direction_source": "composite_score_sign",  # Sign of score
        "percentiles": [75, 80, 85, 90, 95, 99],
    },
    {
        "metric_key": "net_gex",
        "alert_name": "Negative Gamma",
        "current_threshold": -100.0,
        "comparison": "lt",
        "direction_source": "options_algo",  # Options Greeks algo direction
        "percentiles": [1, 5, 10, 15, 20, 25],
    },
    {
        "metric_key": "p_bull",
        "alert_name": "Sentiment Overbull",
        "current_threshold": 0.8,
        "comparison": "gt",
        "direction_source": "contrarian_bearish",
        "percentiles": [80, 85, 90, 92, 95, 97],
    },
    {
        "metric_key": "p_bull",
        "alert_name": "Sentiment Overbear",
        "current_threshold": 0.2,
        "comparison": "lt",
        "direction_source": "contrarian_bullish",
        "percentiles": [3, 5, 8, 10, 15, 20],
    },
    {
        "metric_key": "z_lambda",
        "alert_name": "Illiquidity Spike",
        "current_threshold": 2.0,
        "comparison": "gt",
        "direction_source": "kyle_algo",  # Kyle-Amihud's own direction (order flow)
        "percentiles": [70, 75, 80, 85, 90, 95],
    },
]


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------

def compute_distribution(values: list[float]) -> dict:
    """Compute min, max, mean, std, and percentiles."""
    if not values or len(set(values)) < 2:
        return {
            "count": len(values),
            "unique": len(set(values)) if values else 0,
            "insufficient": True,
        }

    vs = sorted(values)
    n = len(vs)

    def pct(p):
        k = (p / 100) * (n - 1)
        f, c = math.floor(k), math.ceil(k)
        return vs[f] * (c - k) + vs[c] * (k - f) if f != c else vs[int(k)]

    return {
        "count": n,
        "unique": len(set(vs)),
        "min": vs[0],
        "max": vs[-1],
        "mean": round(mean(vs), 6),
        "std": round(stdev(vs), 6) if n >= 2 else 0,
        "P25": round(pct(25), 6),
        "P50": round(pct(50), 6),
        "P75": round(pct(75), 6),
        "P90": round(pct(90), 6),
        "P95": round(pct(95), 6),
        "P99": round(pct(99), 6),
        "insufficient": False,
    }


# ---------------------------------------------------------------------------
# Direction derivation
# ---------------------------------------------------------------------------

def _derive_direction(m: dict, config: dict) -> str:
    source = config["direction_source"]
    if source == "composite":
        # Use thresholded composite direction (strong_bullish/bullish/neutral/etc.)
        d = m.get("direction", "neutral")
        if "bullish" in d:
            return "bullish"
        if "bearish" in d:
            return "bearish"
        return "neutral"
    elif source == "composite_score_sign":
        # Use sign of composite score (works even when |score| < 25)
        score = m.get("composite_score", 0)
        if score > 0:
            return "bullish"
        elif score < 0:
            return "bearish"
        return "neutral"
    elif source == "vpin_algo":
        return m.get("vpin_direction", "neutral")
    elif source == "kyle_algo":
        return m.get("kyle_direction", "neutral")
    elif source == "liq_algo":
        return m.get("liq_direction", "neutral")
    elif source == "options_algo":
        return m.get("options_direction", "neutral")
    elif source == "funding_z_sign":
        return "bearish" if m.get("funding_z", 0) > 0 else "bullish"
    elif source == "smart_div_sign":
        return "bullish" if m.get("smart_div", 0) > 0 else "bearish"
    elif source == "contrarian_bearish":
        return "bearish"
    elif source == "contrarian_bullish":
        return "bullish"
    elif source == "fixed_bullish":
        return "bullish"
    return "neutral"


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_threshold(
    timeseries: list[dict],
    h1_candles: list[dict],
    config: dict,
    cooldown_ms: int = 4 * HOUR,
) -> list[dict]:
    """Sweep candidate thresholds for one metric and evaluate hit rates."""
    metric_key = config["metric_key"]
    comparison = config["comparison"]

    # Collect values for percentile computation
    values = sorted([
        m[metric_key] for m in timeseries
        if isinstance(m.get(metric_key), (int, float))
    ])

    if not values or len(set(values)) < 5:
        return [{"status": "insufficient_data"}]

    n = len(values)

    # Build candidates from percentiles + current threshold
    candidates = []
    seen = set()
    for p in config["percentiles"]:
        idx = int(p / 100 * (n - 1))
        val = round(values[idx], 6)
        if val not in seen:
            candidates.append({"percentile": f"P{p}", "value": val})
            seen.add(val)

    cur = config["current_threshold"]
    if round(cur, 6) not in seen:
        candidates.append({"percentile": "current", "value": round(cur, 6)})

    total_days = len(timeseries) / 24
    results = []

    for cand in candidates:
        threshold = cand["value"]

        # Find crossings
        crossings = []
        for m in timeseries:
            val = m.get(metric_key)
            if not isinstance(val, (int, float)):
                continue
            crossed = False
            if comparison == "gt":
                crossed = val > threshold
            elif comparison == "lt":
                crossed = val < threshold
            if crossed:
                crossings.append(m)

        # Cooldown dedup
        deduped = []
        last_ts = -float("inf")
        for m in crossings:
            if m["timestamp"] - last_ts >= cooldown_ms:
                deduped.append(m)
                last_ts = m["timestamp"]

        # Evaluate outcomes
        hits_4h = 0
        hits_24h = 0
        moves_24h = []
        directional = 0

        for m in deduped:
            ts = m["timestamp"]
            trigger = m["price"]
            if trigger <= 0:
                continue

            direction = _derive_direction(m, config)
            if direction == "neutral":
                continue

            directional += 1
            p4h = _get_price_at_ts(h1_candles, ts + 4 * HOUR)
            p24h = _get_price_at_ts(h1_candles, ts + 24 * HOUR)
            if p4h <= 0 or p24h <= 0:
                continue

            mv4h = (p4h - trigger) / trigger * 100
            mv24h = (p24h - trigger) / trigger * 100
            moves_24h.append(abs(mv24h))

            if direction == "bullish":
                if mv4h >= 0.5:
                    hits_4h += 1
                if mv24h >= 0.5:
                    hits_24h += 1
            else:
                if mv4h <= -0.5:
                    hits_4h += 1
                if mv24h <= -0.5:
                    hits_24h += 1

        alert_count = len(deduped)
        alerts_per_day = alert_count / max(total_days, 1)
        hr4 = (hits_4h / directional * 100) if directional > 0 else 0
        hr24 = (hits_24h / directional * 100) if directional > 0 else 0
        avg_mv = mean(moves_24h) if moves_24h else 0

        # Frequency penalty
        if alerts_per_day > 1:
            freq_pen = min(100, (alerts_per_day - 1) * 30)
        elif alerts_per_day < 1 / 7:
            freq_pen = min(100, (1 / 7 - alerts_per_day) * 700)
        else:
            freq_pen = 0

        cal_score = hr24 * 0.6 + hr4 * 0.2 - freq_pen * 0.2

        results.append({
            "percentile": cand["percentile"],
            "threshold": round(threshold, 6),
            "alerts": alert_count,
            "directional": directional,
            "alerts_per_day": round(alerts_per_day, 2),
            "hit_rate_4h": round(hr4, 1),
            "hit_rate_24h": round(hr24, 1),
            "avg_move_24h": round(avg_mv, 3),
            "freq_penalty": round(freq_pen, 1),
            "cal_score": round(cal_score, 1),
        })

    return results


def select_optimal(sweep_results: list[dict], config: dict) -> dict:
    """Select the best threshold from sweep results."""
    valid = [r for r in sweep_results if r.get("alerts", 0) >= 3 and "status" not in r]
    if not valid:
        return {"status": "no_valid_candidates", "current": config["current_threshold"]}

    valid.sort(key=lambda r: r["cal_score"], reverse=True)
    best = valid[0]

    current_match = [r for r in sweep_results if r.get("percentile") == "current"]
    current_score = current_match[0]["cal_score"] if current_match else 0

    return {
        "recommended": best["threshold"],
        "percentile": best["percentile"],
        "cal_score": best["cal_score"],
        "hit_rate_24h": best["hit_rate_24h"],
        "hit_rate_4h": best["hit_rate_4h"],
        "alerts_per_day": best["alerts_per_day"],
        "current": config["current_threshold"],
        "current_score": current_score,
        "delta": round(best["cal_score"] - current_score, 1),
    }


# ---------------------------------------------------------------------------
# Data collection loop
# ---------------------------------------------------------------------------

async def collect_distributions(
    symbol: str,
    start_date: str,
    end_date: str,
    progress_callback=None,
) -> tuple[list[dict], list[dict]]:
    """Run through all hourly steps and collect metric values."""
    historical = await fetch_all_historical(symbol, start_date, end_date, progress_callback)

    h1 = historical["candles_h1"]
    h4 = historical["candles_h4"]
    d1 = historical["candles_d1"]
    taker = historical.get("taker_volume", [])
    funding = historical.get("funding_rates", [])
    oi = historical.get("oi_history", [])
    top_trader = historical.get("top_trader", [])
    global_ratio = historical.get("global_ratio", [])
    fng = historical.get("fng_history", [])
    coinalyze_oi = historical.get("coinalyze_oi")
    coinalyze_liq = historical.get("coinalyze_liq")
    coinalyze_ls = historical.get("coinalyze_ls")
    deribit = historical.get("deribit_options")

    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date)
    step_start = start_ms + 48 * HOUR
    step_end = end_ms - 24 * HOUR
    total_steps = (step_end - step_start) // HOUR

    timeseries = []
    step_idx = 0

    for current_ts in range(step_start, step_end, HOUR):
        raw = _assemble_raw_data(
            current_ts, h1, h4, d1, taker, funding, oi,
            top_trader, global_ratio, fng,
            coinalyze_oi=coinalyze_oi, coinalyze_liq=coinalyze_liq,
            coinalyze_ls=coinalyze_ls, deribit_options=deribit,
        )

        composite = run_algo_bias(raw)
        metrics = extract_metrics(composite)
        metrics["timestamp"] = current_ts
        metrics["price"] = _get_price_at_ts(h1, current_ts)
        timeseries.append(metrics)

        step_idx += 1
        if progress_callback and step_idx % 500 == 0:
            try:
                progress_callback(step_idx, total_steps)
            except TypeError:
                progress_callback(f"Processing step {step_idx}/{total_steps}...")

    return timeseries, h1


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

DIST_METRICS = [
    ("vpin", "VPIN", 0.7),
    ("abs_funding_z", "|Funding z|", 2.0),
    ("p_cascade", "P(cascade)", 0.6),
    ("abs_smart_div", "|Smart Div|", 8.0),
    ("atr_ratio", "ATR Ratio", 0.65),
    ("entropy", "Entropy", 0.8),
    ("abs_composite_score", "|Comp Score|", 70.0),
    ("composite_confidence", "Comp Conf", 75.0),
    ("net_gex", "Net GEX", -100.0),
    ("p_bull", "P(bull)", 0.8),
    ("z_lambda", "z(Lambda)", 2.0),
    ("hurst", "Hurst", None),
]


def print_distribution_table(distributions: dict):
    print()
    print("=" * 110)
    print("  METRIC DISTRIBUTION ANALYSIS")
    print("=" * 110)
    header = f"  {'Metric':<16} {'Count':>7} {'Min':>9} {'Max':>9} {'Mean':>9} {'Std':>9} {'P50':>9} {'P90':>9} {'P95':>9} {'Current':>9}"
    print(header)
    print("  " + "-" * 106)

    for key, name, current in DIST_METRICS:
        d = distributions.get(key, {})
        if d.get("insufficient"):
            cur_str = f"{current}" if current is not None else "N/A"
            print(f"  {name:<16} {d.get('count', 0):>7} {'--- insufficient data ---':>60}  {cur_str:>9}")
            continue
        cur_str = f"{current:.4f}" if current is not None else "N/A"
        print(f"  {name:<16} {d['count']:>7} {d['min']:>9.4f} {d['max']:>9.4f} "
              f"{d['mean']:>9.4f} {d['std']:>9.4f} {d['P50']:>9.4f} "
              f"{d['P90']:>9.4f} {d['P95']:>9.4f} {cur_str:>9}")
    print()


def print_sweep_details(sweep_results: dict):
    print("=" * 110)
    print("  THRESHOLD SWEEP DETAILS")
    print("=" * 110)

    for config in THRESHOLD_SWEEPS:
        name = config["alert_name"]
        results = sweep_results.get(name, [])
        if not results or results[0].get("status") == "insufficient_data":
            print(f"\n  {name}: SKIPPED (insufficient data variance)")
            continue

        print(f"\n  {name} (current: {config['current_threshold']}, metric: {config['metric_key']}, comparison: {config['comparison']})")
        print(f"  {'Percentile':<12} {'Threshold':>12} {'Alerts':>8} {'A/Day':>7} {'4h HR':>8} {'24h HR':>8} {'AvgMv':>8} {'Score':>8}")
        print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for r in sorted(results, key=lambda x: x.get("cal_score", -999), reverse=True):
            tag = " <-- BEST" if r == sorted(results, key=lambda x: x.get("cal_score", -999), reverse=True)[0] and r.get("alerts", 0) >= 3 else ""
            tag = " <-- CURRENT" if r.get("percentile") == "current" else tag
            print(f"  {r.get('percentile', '?'):<12} {r.get('threshold', 0):>12.6f} "
                  f"{r.get('alerts', 0):>8} {r.get('alerts_per_day', 0):>7.2f} "
                  f"{r.get('hit_rate_4h', 0):>7.1f}% {r.get('hit_rate_24h', 0):>7.1f}% "
                  f"{r.get('avg_move_24h', 0):>7.3f}% {r.get('cal_score', 0):>7.1f}{tag}")
    print()


def print_recommendations(recommendations: dict):
    print("=" * 110)
    print("  THRESHOLD RECOMMENDATIONS")
    print("=" * 110)
    print(f"  {'Alert':<22} {'Current':>10} {'Reco':>10} {'Pctile':>8} {'A/Day':>7} {'4h HR':>8} {'24h HR':>8} {'Score':>8} {'Delta':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for config in THRESHOLD_SWEEPS:
        name = config["alert_name"]
        rec = recommendations.get(name, {})

        if rec.get("status"):
            print(f"  {name:<22} {config['current_threshold']:>10.4f} {'N/A':>10} {'':>8} {'':>7} {'':>8} {'':>8} {'':>8} {'SKIP':>8}")
            continue

        print(f"  {name:<22} {rec['current']:>10.4f} {rec['recommended']:>10.4f} "
              f"{rec['percentile']:>8} {rec['alerts_per_day']:>7.2f} "
              f"{rec['hit_rate_4h']:>7.1f}% {rec['hit_rate_24h']:>7.1f}% "
              f"{rec['cal_score']:>7.1f} {rec['delta']:>+7.1f}")
    print()
    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def progress(step_or_msg, total=None):
    if isinstance(step_or_msg, str):
        print(f"\r  {step_or_msg:<70}", end="", flush=True)
        return
    step = step_or_msg
    if total and total > 0:
        pct = (step + 1) / total * 100
        bar_len = 40
        filled = int(bar_len * (step + 1) / total)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:.0f}% ({step + 1}/{total} steps)", end="", flush=True)


async def main():
    parser = argparse.ArgumentParser(description="Calibrate Quant Alert Thresholds")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-16", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--cooldown", type=int, default=4, help="Alert cooldown hours")
    args = parser.parse_args()

    print("\n  Quant Threshold Calibration")
    print(f"  {args.symbol} | {args.start} → {args.end} | cooldown={args.cooldown}h")
    print()

    t0 = time.time()

    # Step 1: Collect distributions
    print("  Step 1: Collecting metric distributions...")
    timeseries, h1_candles = await collect_distributions(
        args.symbol, args.start, args.end, progress_callback=progress,
    )
    print(f"\r  Collected {len(timeseries)} hourly steps in {time.time() - t0:.1f}s{' ' * 30}")

    # Compute distributions
    distributions = {}
    for key, _, _ in DIST_METRICS:
        values = [m[key] for m in timeseries if isinstance(m.get(key), (int, float))]
        distributions[key] = compute_distribution(values)

    print_distribution_table(distributions)

    # Step 2: Sweep thresholds
    print("  Step 2: Sweeping thresholds with outcome evaluation...")
    sweep_results = {}
    for config in THRESHOLD_SWEEPS:
        metric_key = config["metric_key"]
        dist = distributions.get(metric_key, {})
        if dist.get("insufficient"):
            sweep_results[config["alert_name"]] = [{"status": "insufficient_data"}]
            continue
        sweep_results[config["alert_name"]] = sweep_threshold(
            timeseries, h1_candles, config, args.cooldown * HOUR,
        )

    print_sweep_details(sweep_results)

    # Step 3: Select optimal
    recommendations = {}
    for config in THRESHOLD_SWEEPS:
        name = config["alert_name"]
        results = sweep_results.get(name, [])
        recommendations[name] = select_optimal(results, config)

    print_recommendations(recommendations)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s\n")

    # Save JSON
    QUANT_BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = QUANT_BACKTEST_DIR / f"threshold_calibration_{ts_str}.json"

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "start_date": args.start,
        "end_date": args.end,
        "total_steps": len(timeseries),
        "elapsed_seconds": round(elapsed, 1),
        "distributions": distributions,
        "sweeps": {k: v for k, v in sweep_results.items()},
        "recommendations": recommendations,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
