"""Phase 2: Historical Validation — Detection Rate Analysis & Multi-TF Alignment.

Fetches 6+ months of historical BTCUSDT candles, runs all detectors,
and produces a quantitative report on detection rates, filter effectiveness,
and multi-TF alignment.

Usage:
    cd backend
    python -m scripts.run_validation [--start 2024-06-01] [--end 2025-01-01] [--symbol BTCUSDT]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# Add backend to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import SYMBOL
from app.core.engine import analyze_timeframe, run_multi_tf_analysis, AnalysisResult
from app.models import (
    DetectorMetrics, ValidationReport, TrendState,
    IDMStatus, Direction,
)
from app.services.data_fetcher import fetch_or_cache_historical

# Validation output directory
VALIDATION_DIR = Path(__file__).resolve().parents[1] / "data" / "validation"

# Sanity check ranges (per 1000 candles)
SANITY_CHECKS = {
    "bos_rate": {"min": 5, "max": 50, "label": "BOS rate per 1K candles"},
    "choch_rate": {"min": 2, "max": 20, "label": "CHoCH rate per 1K candles"},
    "choch_less_than_bos": {"label": "CHoCH rarer than BOS"},
    "fake_choch_pct": {"min": 10, "max": 70, "label": "Fake CHoCH filter %"},
    "valid_ob_pct": {"min": 5, "max": 50, "label": "Valid OB % (double filter)"},
    "fvg_extreme_pct": {"min": 15, "max": 80, "label": "FVG from extreme candle %"},
}


def _rate(count: int, total_candles: int) -> float:
    """Compute detections per 1000 candles."""
    if total_candles == 0:
        return 0.0
    return round(count / total_candles * 1000, 1)


def _pct(part: int, whole: int) -> float:
    """Compute percentage, safe for zero denominator."""
    if whole == 0:
        return 0.0
    return round(part / whole * 100, 1)


def collect_swing_metrics(result: AnalysisResult, n_candles: int) -> DetectorMetrics:
    """Collect swing detection metrics."""
    swings = result.swings
    total = len(swings)
    valid_smc = sum(1 for s in swings if s.is_valid_smc)
    strong = sum(1 for s in swings if s.is_strong)
    hh = sum(1 for s in swings if s.classification.value == "HH")
    hl = sum(1 for s in swings if s.classification.value == "HL")
    lh = sum(1 for s in swings if s.classification.value == "LH")
    ll = sum(1 for s in swings if s.classification.value == "LL")
    unclassified = sum(1 for s in swings if s.classification.value == "unclassified")

    return DetectorMetrics(
        detector_name="Swings",
        timeframe=result.timeframe,
        total_detected=total,
        per_1000_candles=_rate(total, n_candles),
        valid_pct=_pct(valid_smc, total),
        breakdown={
            "HH": hh, "HL": hl, "LH": lh, "LL": ll,
            "unclassified": unclassified,
            "valid_smc": valid_smc,
            "valid_smc_pct": _pct(valid_smc, total),
            "strong": strong,
            "strong_pct": _pct(strong, total),
        },
    )


def collect_idm_metrics(result: AnalysisResult, n_candles: int) -> DetectorMetrics:
    """Collect inducement detection metrics."""
    idms = result.inducements
    total = len(idms)
    taken = sum(1 for i in idms if i.status == IDMStatus.TAKEN)
    body_closed = sum(1 for i in idms if i.body_closed)
    major = sum(1 for i in idms if i.is_major)

    return DetectorMetrics(
        detector_name="IDM",
        timeframe=result.timeframe,
        total_detected=total,
        per_1000_candles=_rate(total, n_candles),
        valid_pct=_pct(taken, total),
        breakdown={
            "taken": taken,
            "taken_pct": _pct(taken, total),
            "body_closed": body_closed,
            "body_closed_of_taken_pct": _pct(body_closed, taken),
            "major": major,
            "major_pct": _pct(major, total),
        },
    )


def collect_bos_metrics(result: AnalysisResult, n_candles: int) -> DetectorMetrics:
    """Collect BOS detection metrics."""
    bos = result.bos_events
    total = len(bos)
    valid = sum(1 for b in bos if b.valid)
    bullish = sum(1 for b in bos if b.direction == Direction.BULLISH)
    bearish = sum(1 for b in bos if b.direction == Direction.BEARISH)
    idm_body = sum(1 for b in bos if b.idm_body_closed)

    return DetectorMetrics(
        detector_name="BOS",
        timeframe=result.timeframe,
        total_detected=total,
        per_1000_candles=_rate(total, n_candles),
        valid_pct=_pct(valid, total),
        breakdown={
            "valid": valid,
            "invalid": total - valid,
            "bullish": bullish,
            "bearish": bearish,
            "idm_body_closed": idm_body,
            "idm_body_closed_pct": _pct(idm_body, valid),
        },
    )


def collect_choch_metrics(result: AnalysisResult, n_candles: int) -> DetectorMetrics:
    """Collect CHoCH detection metrics."""
    choch = result.choch_events
    total = len(choch)
    fake = sum(1 for c in choch if c.is_fake)
    confirmed = sum(1 for c in choch if c.confirmed)
    confidences = [c.confidence for c in choch if not c.is_fake]
    avg_conf = round(mean(confidences), 2) if confidences else 0.0
    swing_model = sum(1 for c in choch if c.model == "swing")
    sweep_model = sum(1 for c in choch if c.model == "sweep")

    return DetectorMetrics(
        detector_name="CHoCH",
        timeframe=result.timeframe,
        total_detected=total,
        per_1000_candles=_rate(total, n_candles),
        valid_pct=_pct(total - fake, total),
        breakdown={
            "fake": fake,
            "fake_pct": _pct(fake, total),
            "confirmed": confirmed,
            "confirmed_pct": _pct(confirmed, total),
            "avg_confidence": avg_conf,
            "model_swing": swing_model,
            "model_sweep": sweep_model,
        },
    )


def collect_fvg_metrics(result: AnalysisResult, n_candles: int) -> DetectorMetrics:
    """Collect FVG detection metrics."""
    fvgs = result.fvgs
    total = len(fvgs)
    valid = sum(1 for f in fvgs if f.valid)
    extreme = sum(1 for f in fvgs if f.from_extreme_candle)
    mitigated = sum(1 for f in fvgs if f.mitigated)
    bullish = sum(1 for f in fvgs if f.direction == Direction.BULLISH)
    bearish = sum(1 for f in fvgs if f.direction == Direction.BEARISH)

    return DetectorMetrics(
        detector_name="FVG",
        timeframe=result.timeframe,
        total_detected=total,
        per_1000_candles=_rate(total, n_candles),
        valid_pct=_pct(valid, total),
        breakdown={
            "valid": valid,
            "from_extreme_candle": extreme,
            "extreme_of_valid_pct": _pct(extreme, valid),
            "mitigated": mitigated,
            "mitigated_pct": _pct(mitigated, total),
            "bullish": bullish,
            "bearish": bearish,
        },
    )


def collect_ob_metrics(result: AnalysisResult, n_candles: int) -> DetectorMetrics:
    """Collect Order Block detection metrics."""
    obs = result.order_blocks
    total = len(obs)
    valid = sum(1 for o in obs if o.valid)
    has_fvg = sum(1 for o in obs if o.has_fvg)
    swept = sum(1 for o in obs if o.swept_liquidity)
    trap = sum(1 for o in obs if o.is_trap)
    mitigated = sum(1 for o in obs if o.mitigated)
    bullish = sum(1 for o in obs if o.direction == Direction.BULLISH)
    bearish = sum(1 for o in obs if o.direction == Direction.BEARISH)

    return DetectorMetrics(
        detector_name="OB",
        timeframe=result.timeframe,
        total_detected=total,
        per_1000_candles=_rate(total, n_candles),
        valid_pct=_pct(valid, total),
        breakdown={
            "valid": valid,
            "has_fvg": has_fvg,
            "has_fvg_pct": _pct(has_fvg, total),
            "swept_liquidity": swept,
            "swept_pct": _pct(swept, total),
            "trap": trap,
            "trap_pct": _pct(trap, total),
            "mitigated": mitigated,
            "mitigated_pct": _pct(mitigated, total),
            "bullish": bullish,
            "bearish": bearish,
        },
    )


def collect_all_metrics(result: AnalysisResult, n_candles: int) -> list[DetectorMetrics]:
    """Collect metrics for all detectors on a single timeframe."""
    return [
        collect_swing_metrics(result, n_candles),
        collect_idm_metrics(result, n_candles),
        collect_bos_metrics(result, n_candles),
        collect_choch_metrics(result, n_candles),
        collect_fvg_metrics(result, n_candles),
        collect_ob_metrics(result, n_candles),
    ]


def compute_multi_tf_alignment(
    results: dict[str, AnalysisResult],
) -> dict:
    """Compute multi-TF alignment metrics."""
    trends = {}
    for tf in ["W1", "D1", "H4", "M15"]:
        r = results.get(tf)
        trends[tf] = r.trend.value if r else "ranging"

    # Pairwise agreement
    pairs = [("W1", "D1"), ("D1", "H4"), ("H4", "M15")]
    agreements = {}
    for a, b in pairs:
        ta = trends.get(a, "ranging")
        tb = trends.get(b, "ranging")
        agreements[f"{a}_{b}"] = ta == tb

    # Full alignment
    unique_trends = set(trends.values()) - {"ranging"}
    full_align = len(unique_trends) <= 1

    # Cross-TF fake CHoCH count
    cross_tf_fake = 0
    for tf in ["H4", "M15"]:
        r = results.get(tf)
        if r:
            cross_tf_fake += sum(1 for c in r.choch_events if c.is_fake)

    # D1 BOS in W1 trend direction
    w1_trend = trends.get("W1", "ranging")
    d1_r = results.get("D1")
    d1_bos_aligned = 0
    d1_bos_total = 0
    if d1_r and w1_trend != "ranging":
        for b in d1_r.bos_events:
            if b.valid:
                d1_bos_total += 1
                if b.direction.value == w1_trend:
                    d1_bos_aligned += 1

    return {
        "trends": trends,
        "pairwise_agreement": agreements,
        "full_alignment": full_align,
        "cross_tf_fake_choch": cross_tf_fake,
        "d1_bos_in_w1_direction": {
            "aligned": d1_bos_aligned,
            "total": d1_bos_total,
            "pct": _pct(d1_bos_aligned, d1_bos_total),
        },
    }


def run_sanity_checks(
    all_metrics: list[DetectorMetrics],
) -> dict[str, str]:
    """Run sanity checks on the collected metrics and return PASS/WARN/FAIL."""
    checks: dict[str, str] = {}

    # Get H4 metrics as the primary reference TF
    h4_bos = next((m for m in all_metrics if m.detector_name == "BOS" and m.timeframe == "H4"), None)
    h4_choch = next((m for m in all_metrics if m.detector_name == "CHoCH" and m.timeframe == "H4"), None)
    h4_fvg = next((m for m in all_metrics if m.detector_name == "FVG" and m.timeframe == "H4"), None)
    h4_ob = next((m for m in all_metrics if m.detector_name == "OB" and m.timeframe == "H4"), None)

    # BOS rate check
    if h4_bos:
        rate = h4_bos.per_1000_candles
        sc = SANITY_CHECKS["bos_rate"]
        if sc["min"] <= rate <= sc["max"]:
            checks["bos_rate"] = "PASS"
        elif rate < sc["min"] * 0.5 or rate > sc["max"] * 2:
            checks["bos_rate"] = f"FAIL ({rate}/1K)"
        else:
            checks["bos_rate"] = f"WARN ({rate}/1K)"

    # CHoCH rate check
    if h4_choch:
        rate = h4_choch.per_1000_candles
        sc = SANITY_CHECKS["choch_rate"]
        if sc["min"] <= rate <= sc["max"]:
            checks["choch_rate"] = "PASS"
        elif rate < sc["min"] * 0.5 or rate > sc["max"] * 2:
            checks["choch_rate"] = f"FAIL ({rate}/1K)"
        else:
            checks["choch_rate"] = f"WARN ({rate}/1K)"

    # CHoCH should be rarer than BOS
    if h4_bos and h4_choch:
        if h4_choch.per_1000_candles < h4_bos.per_1000_candles:
            checks["choch_less_than_bos"] = "PASS"
        else:
            checks["choch_less_than_bos"] = "WARN"

    # Fake CHoCH filter %
    if h4_choch:
        fake_pct = h4_choch.breakdown.get("fake_pct", 0)
        sc = SANITY_CHECKS["fake_choch_pct"]
        if sc["min"] <= fake_pct <= sc["max"]:
            checks["fake_choch_pct"] = f"PASS ({fake_pct}%)"
        elif fake_pct < sc["min"]:
            checks["fake_choch_pct"] = f"WARN ({fake_pct}% — filter may be too lenient)"
        else:
            checks["fake_choch_pct"] = f"WARN ({fake_pct}% — filter may be too aggressive)"

    # Valid OB %
    if h4_ob:
        valid_pct = h4_ob.valid_pct
        sc = SANITY_CHECKS["valid_ob_pct"]
        if sc["min"] <= valid_pct <= sc["max"]:
            checks["valid_ob_pct"] = f"PASS ({valid_pct}%)"
        elif valid_pct > sc["max"]:
            checks["valid_ob_pct"] = f"WARN ({valid_pct}% — double filter may be too lenient)"
        else:
            checks["valid_ob_pct"] = f"WARN ({valid_pct}% — double filter may be too aggressive)"

    # FVG from extreme candle %
    if h4_fvg:
        extreme_pct = h4_fvg.breakdown.get("extreme_of_valid_pct", 0)
        sc = SANITY_CHECKS["fvg_extreme_pct"]
        if sc["min"] <= extreme_pct <= sc["max"]:
            checks["fvg_extreme_pct"] = f"PASS ({extreme_pct}%)"
        elif extreme_pct < sc["min"]:
            checks["fvg_extreme_pct"] = f"WARN ({extreme_pct}% — may need tuning)"
        else:
            checks["fvg_extreme_pct"] = f"PASS ({extreme_pct}%)"

    return checks


def print_report(
    report: ValidationReport,
    all_metrics: list[DetectorMetrics],
) -> None:
    """Print human-readable validation report to console."""
    print()
    print("=" * 60)
    print("  TradingMamba Phase 2 — Historical Validation Report")
    print(f"  {report.symbol} | {report.start_date} -> {report.end_date}")
    print("=" * 60)
    print()

    # Candle counts
    print("CANDLE COUNTS:")
    for tf in ["W1", "D1", "H4", "M15"]:
        count = report.total_candles.get(tf, 0)
        print(f"  {tf}: {count:,} candles")
    print()

    # Detection rates table
    print("DETECTION RATES (per 1000 candles):")
    print(f"  {'Detector':<10} {'W1':>8} {'D1':>8} {'H4':>8} {'M15':>8}")
    print(f"  {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for detector_name in ["Swings", "IDM", "BOS", "CHoCH", "FVG", "OB"]:
        row = f"  {detector_name:<10}"
        for tf in ["W1", "D1", "H4", "M15"]:
            m = next((x for x in all_metrics if x.detector_name == detector_name and x.timeframe == tf), None)
            if m:
                row += f" {m.per_1000_candles:>7.1f}"
            else:
                row += f" {'—':>7}"
        print(row)
    print()

    # Total counts table
    print("TOTAL COUNTS:")
    print(f"  {'Detector':<10} {'W1':>8} {'D1':>8} {'H4':>8} {'M15':>8}")
    print(f"  {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for detector_name in ["Swings", "IDM", "BOS", "CHoCH", "FVG", "OB"]:
        row = f"  {detector_name:<10}"
        for tf in ["W1", "D1", "H4", "M15"]:
            m = next((x for x in all_metrics if x.detector_name == detector_name and x.timeframe == tf), None)
            if m:
                row += f" {m.total_detected:>8}"
            else:
                row += f" {'—':>8}"
        print(row)
    print()

    # Filter effectiveness (H4 as primary reference)
    print("FILTER EFFECTIVENESS (H4):")
    for detector_name in ["BOS", "CHoCH", "FVG", "OB"]:
        m = next((x for x in all_metrics if x.detector_name == detector_name and x.timeframe == "H4"), None)
        if not m:
            continue

        if detector_name == "BOS":
            valid = m.breakdown.get("valid", 0)
            invalid = m.breakdown.get("invalid", 0)
            idm_pct = m.breakdown.get("idm_body_closed_pct", 0)
            print(f"  BOS:   {m.valid_pct}% valid ({invalid} invalidated), IDM body close: {idm_pct}%")
        elif detector_name == "CHoCH":
            fake_pct = m.breakdown.get("fake_pct", 0)
            conf_pct = m.breakdown.get("confirmed_pct", 0)
            avg_conf = m.breakdown.get("avg_confidence", 0)
            print(f"  CHoCH: {fake_pct}% fake filtered, {conf_pct}% confirmed, avg confidence: {avg_conf}")
        elif detector_name == "FVG":
            extreme_pct = m.breakdown.get("extreme_of_valid_pct", 0)
            mit_pct = m.breakdown.get("mitigated_pct", 0)
            print(f"  FVG:   {m.valid_pct}% valid, {extreme_pct}% from extreme candle, {mit_pct}% mitigated")
        elif detector_name == "OB":
            fvg_pct = m.breakdown.get("has_fvg_pct", 0)
            swept_pct = m.breakdown.get("swept_pct", 0)
            trap_pct = m.breakdown.get("trap_pct", 0)
            print(f"  OB:    {m.valid_pct}% valid, FVG: {fvg_pct}%, swept liq: {swept_pct}%, traps: {trap_pct}%")
    print()

    # Multi-TF alignment
    alignment = report.multi_tf_alignment
    trends = alignment.get("trends", {})
    pairs = alignment.get("pairwise_agreement", {})
    print("MULTI-TF ALIGNMENT:")
    print(f"  Current trends: W1={trends.get('W1', '?')} | D1={trends.get('D1', '?')} | H4={trends.get('H4', '?')} | M15={trends.get('M15', '?')}")
    for pair_key, agrees in pairs.items():
        a, b = pair_key.split("_")
        status = "YES" if agrees else "NO"
        print(f"  {a}<->{b} agreement: {status}")
    full = alignment.get("full_alignment", False)
    print(f"  Full 4-TF alignment: {'YES' if full else 'NO'}")
    d1_bos = alignment.get("d1_bos_in_w1_direction", {})
    print(f"  D1 BOS in W1 direction: {d1_bos.get('aligned', 0)}/{d1_bos.get('total', 0)} ({d1_bos.get('pct', 0)}%)")
    print(f"  Cross-TF fake CHoCH: {alignment.get('cross_tf_fake_choch', 0)}")
    print()

    # Sanity checks
    print("SANITY CHECKS:")
    has_fail = False
    has_warn = False
    for check_key, result_str in report.sanity_check_results.items():
        label = SANITY_CHECKS.get(check_key, {}).get("label", check_key)
        if result_str.startswith("PASS"):
            icon = "OK"
        elif result_str.startswith("WARN"):
            icon = "!!"
            has_warn = True
        else:
            icon = "XX"
            has_fail = True
        print(f"  [{icon}] {label}: {result_str}")
    print()

    # Verdict
    if has_fail:
        print("VERDICT: FAIL — Detectors need tuning before Phase 3")
    elif has_warn:
        print("VERDICT: PASS (with warnings) — Review warnings, but can proceed to Phase 3")
    else:
        print("VERDICT: PASS — Ready for Phase 3 (Backtesting)")
    print()


async def main(start_date: str, end_date: str, symbol: str) -> None:
    """Run the full Phase 2 validation pipeline."""
    print(f"\nPhase 2 Validation: {symbol} | {start_date} -> {end_date}")
    print("=" * 60)

    # Step 1: Fetch historical data for all timeframes
    timeframes = ["W1", "D1", "H4", "M15"]
    candles_by_tf: dict[str, list] = {}

    for tf in timeframes:
        print(f"  Fetching {tf}...", end=" ", flush=True)
        candles = await fetch_or_cache_historical(symbol, tf, start_date, end_date)
        candles_by_tf[tf] = candles
        print(f"{len(candles):,} candles")

    print()

    # Step 2: Run engine on each timeframe individually
    print("Running detection engine...")
    results_single: dict[str, AnalysisResult] = {}
    for tf in timeframes:
        candles = candles_by_tf[tf]
        if candles:
            print(f"  Analyzing {tf} ({len(candles):,} candles)...", end=" ", flush=True)
            result = analyze_timeframe(candles, tf)
            results_single[tf] = result
            n_swings = len(result.swings)
            n_bos = len(result.bos_events)
            n_choch = len(result.choch_events)
            print(f"swings={n_swings}, BOS={n_bos}, CHoCH={n_choch}")

    print()

    # Step 3: Run multi-TF analysis
    print("Running multi-TF analysis (W1->D1->H4->M15)...")
    results_multi = run_multi_tf_analysis(candles_by_tf)
    print("  Done. Cross-TF fake CHoCH filter applied.")
    print()

    # Step 4: Collect all metrics (use multi-TF results which include cross-TF filtering)
    print("Collecting metrics...")
    all_metrics: list[DetectorMetrics] = []
    total_candles: dict[str, int] = {}

    for tf in timeframes:
        r = results_multi.get(tf)
        if r:
            n = len(candles_by_tf[tf])
            total_candles[tf] = n
            metrics = collect_all_metrics(r, n)
            all_metrics.extend(metrics)

    # Step 5: Multi-TF alignment
    alignment = compute_multi_tf_alignment(results_multi)

    # Step 6: Sanity checks
    sanity = run_sanity_checks(all_metrics)

    # Step 7: Build report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = ValidationReport(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        total_candles=total_candles,
        detector_metrics=all_metrics,
        multi_tf_alignment=alignment,
        sanity_check_results=sanity,
        timestamp=timestamp,
    )

    # Step 8: Save report JSON
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    report_path = VALIDATION_DIR / f"report_{timestamp}.json"

    report_dict = {
        "symbol": report.symbol,
        "start_date": report.start_date,
        "end_date": report.end_date,
        "total_candles": report.total_candles,
        "timestamp": report.timestamp,
        "detector_metrics": [asdict(m) for m in report.detector_metrics],
        "multi_tf_alignment": report.multi_tf_alignment,
        "sanity_check_results": report.sanity_check_results,
    }

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Report saved to: {report_path}")

    # Step 9: Print human-readable report
    print_report(report, all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingMamba Phase 2 Historical Validation")
    parser.add_argument("--start", default="2024-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default=SYMBOL, help="Trading pair (default: BTCUSDT)")
    args = parser.parse_args()

    asyncio.run(main(args.start, args.end, args.symbol))
