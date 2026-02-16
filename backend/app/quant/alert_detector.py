"""Quant Alert Detector — Python port of detectAlerts() from AlgoBiasTab.tsx.

Same 12 individual threshold conditions + 4 Perfect Storm combos.
Same direction derivation logic. Used by the quant backtester to
detect alert conditions on historical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from app.quant.algo_bias import CompositeBias, AlgoBiasResult


@dataclass
class QuantAlert:
    """A detected alert condition from quant algo outputs."""
    id: str                         # e.g. "vpin-toxic", "combo-liquidation-waterfall"
    severity: str                   # "critical" | "warning" | "info"
    title: str
    direction: str                  # "bullish" | "bearish" | "neutral"
    direction_reason: str
    description: str
    combo: Optional[str] = None     # combo ID if Perfect Storm
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_algo(composite: CompositeBias, algo_id: str) -> Optional[AlgoBiasResult]:
    """Find an algo result by ID."""
    for a in composite.algos:
        if isinstance(a, AlgoBiasResult):
            if a.algo_id == algo_id:
                return a
        elif isinstance(a, dict):
            if a.get("algo_id") == algo_id:
                return a
    return None


def _get_components(composite: CompositeBias, algo_id: str) -> dict:
    """Get components dict for an algo, or empty dict."""
    a = _get_algo(composite, algo_id)
    if a is None:
        return {}
    if isinstance(a, AlgoBiasResult):
        return a.components or {}
    return a.get("components", {})


def _get_direction(composite: CompositeBias, algo_id: str) -> str:
    """Get direction string for an algo."""
    a = _get_algo(composite, algo_id)
    if a is None:
        return "neutral"
    if isinstance(a, AlgoBiasResult):
        return a.direction or "neutral"
    return a.get("direction", "neutral")


def _n(v) -> float:
    """Safe number accessor."""
    return float(v) if isinstance(v, (int, float)) else 0.0


def _b(v) -> bool:
    """Safe boolean accessor."""
    return v is True


def _s(v) -> str:
    """Safe string accessor."""
    return str(v) if isinstance(v, str) else ""


def _algo_dir(composite: CompositeBias, algo_id: str) -> str:
    """Get algo direction as 'bullish'/'bearish'/'neutral'."""
    d = _get_direction(composite, algo_id)
    if "bullish" in d:
        return "bullish"
    if "bearish" in d:
        return "bearish"
    return "neutral"


def _composite_dir(composite: CompositeBias) -> str:
    """Get composite direction as 'bullish'/'bearish'/'neutral'."""
    if "bullish" in composite.direction:
        return "bullish"
    if "bearish" in composite.direction:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Main detection function — exact port of TypeScript detectAlerts()
# ---------------------------------------------------------------------------

def detect_alerts(composite: CompositeBias) -> list[QuantAlert]:
    """Detect significant move alert conditions from composite bias output.

    Same thresholds and logic as the frontend detectAlerts() in AlgoBiasTab.tsx.
    Returns list of QuantAlert sorted by severity (critical first).
    """
    alerts: list[QuantAlert] = []
    c_dir = _composite_dir(composite)

    # Extract components for each algo
    vpin = _get_components(composite, "vpin")
    funding = _get_components(composite, "funding_ou")
    options = _get_components(composite, "options_greeks")
    kyle = _get_components(composite, "kyle_amihud")
    liq = _get_components(composite, "liquidation")
    vol = _get_components(composite, "vol_regime")
    smart = _get_components(composite, "smart_retail")
    bayes = _get_components(composite, "bayesian_sentiment")

    # ── Boolean conditions (thresholds calibrated on 3y BTCUSDT data) ──
    vpin_high = _n(vpin.get("vpin")) > 0.064
    funding_z = _n(funding.get("z_score"))
    funding_extreme = abs(funding_z) > 0.551
    cascade_active = _b(liq.get("cascade_active"))
    cascade_forming = _n(liq.get("p_cascade")) > 0.78
    smart_div = _n(smart.get("raw_divergence"))
    smart_retail_split = abs(smart_div) > 3.8
    vol_compression = _s(vol.get("vol_regime")) == "low" and _n(vol.get("atr_ratio")) < 0.769
    vol_extreme = _s(vol.get("vol_regime")) == "extreme"
    strong_consensus = composite.entropy < 0.985 and (composite.agreement_count / max(composite.algo_count, 1)) > 0.7
    extreme_bias = abs(composite.score) > 2.874 and composite.confidence > 16.144
    negative_gamma = _n(options.get("net_gex")) < -100
    p_bull = _n(bayes.get("p_posterior_bull"))
    sentiment_extreme = p_bull > 0.638 or (p_bull > 0 and p_bull < 0.15)
    illiquidity_spike = _n(kyle.get("z_lambda")) > 0.006
    fng_value = _n(bayes.get("fng_value"))

    # Direction derivations
    sell_liq = _n(liq.get("sell_liq_usd"))
    buy_liq = _n(liq.get("buy_liq_usd"))
    cascade_dir = "bullish" if sell_liq > buy_liq else ("bearish" if buy_liq > sell_liq else c_dir)
    cascade_side = "Longs flushed → expect bounce UP" if sell_liq > buy_liq else "Shorts squeezed → expect drop DOWN"

    funding_dir = "bearish" if funding_z > 0 else "bullish"
    funding_side = ("Longs overleveraged → expect price to drop" if funding_z > 0
                    else "Shorts overleveraged → expect price to rise")

    smart_dir = "bullish" if smart_div > 0 else "bearish"
    smart_side = (f"Smart money net LONG vs retail SHORT → expect move UP" if smart_div > 0
                  else f"Smart money net SHORT vs retail LONG → expect move DOWN")

    sent_dir = "bearish" if p_bull > 0.638 else "bullish"
    sent_side = ("Crowd extremely bullish → contrarian: expect pullback DOWN" if p_bull > 0.638
                 else "Crowd extremely bearish → contrarian: expect reversal UP")

    # ── Perfect Storm Combos (CRITICAL) ──

    if cascade_active and vpin_high and funding_extreme:
        alerts.append(QuantAlert(
            id="combo-liquidation-waterfall",
            severity="critical",
            title="Liquidation Waterfall",
            combo="liquidation_waterfall",
            direction=cascade_dir,
            direction_reason=f"{cascade_side}. Informed flow (VPIN {_n(vpin.get('vpin')):.2f}) confirms direction",
            description=f"Active cascade (${_n(liq.get('recent_volume_usd')) / 1e6:.0f}M) + toxic flow + extreme leverage",
            metrics={"vpin": _n(vpin.get("vpin")), "funding_z": funding_z,
                     "liq_vol": _n(liq.get("recent_volume_usd"))},
        ))

    if vol_compression and negative_gamma and composite.entropy < 0.8:
        d = c_dir if c_dir != "neutral" else "bullish"
        alerts.append(QuantAlert(
            id="combo-gamma-squeeze",
            severity="critical",
            title="Gamma Squeeze Setup",
            combo="gamma_squeeze",
            direction=d,
            direction_reason=("Compression + negative gamma → explosive breakout UP likely"
                              if d == "bullish"
                              else "Compression + negative gamma → explosive breakdown likely"),
            description=f"Vol compression (ATR {_n(vol.get('atr_ratio')):.2f}) + negative GEX ({_n(options.get('net_gex')):.0f}) + algo consensus",
            metrics={"atr_ratio": _n(vol.get("atr_ratio")), "net_gex": _n(options.get("net_gex")),
                     "entropy": composite.entropy},
        ))

    if fng_value > 0 and fng_value < 25 and funding_extreme and (cascade_active or cascade_forming):
        d = "bullish" if funding_z > 0 else "bearish"
        alerts.append(QuantAlert(
            id="combo-capitulation",
            severity="critical",
            title="Capitulation Signal",
            combo="capitulation",
            direction=d,
            direction_reason=(f"Fear & Greed at {fng_value:.0f} (extreme fear) + longs liquidating → selling exhaustion, expect bounce UP"
                              if d == "bullish"
                              else f"Fear & Greed at {fng_value:.0f} + shorts squeezed → expect further DOWN"),
            description=f"Extreme fear (FnG {fng_value:.0f}) + funding z={funding_z:.1f} + cascade pressure",
            metrics={"fng": fng_value, "funding_z": funding_z, "p_cascade": _n(liq.get("p_cascade"))},
        ))

    if vpin_high and smart_retail_split and illiquidity_spike:
        alerts.append(QuantAlert(
            id="combo-structural-imbalance",
            severity="critical",
            title="Structural Imbalance",
            combo="structural_imbalance",
            direction=smart_dir,
            direction_reason=f"{smart_side}. Illiquid book (z={_n(kyle.get('z_lambda')):.1f}) means small flow will move price fast",
            description=f"Toxic flow (VPIN {_n(vpin.get('vpin')):.2f}) + smart/retail split ({smart_div:.1f}pp) + thin book",
            metrics={"vpin": _n(vpin.get("vpin")), "divergence": smart_div,
                     "z_lambda": _n(kyle.get("z_lambda"))},
        ))

    # ── Individual Alerts (skip if already in a combo) ──
    combo_ids = {a.combo for a in alerts if a.combo}

    if cascade_active and "liquidation_waterfall" not in combo_ids:
        alerts.append(QuantAlert(
            id="cascade-active",
            severity="critical",
            title="Liquidation Cascade Active",
            direction=cascade_dir,
            direction_reason=cascade_side,
            description=f"${_n(liq.get('recent_volume_usd')) / 1e6:.0f}M liquidated in 30 min — forced selling exhausting",
            metrics={"volume": _n(liq.get("recent_volume_usd")),
                     "sell_liq": sell_liq, "buy_liq": buy_liq},
        ))

    # Composite score sign direction (more reliable than thresholded direction)
    cs_dir = "bullish" if composite.score > 0 else ("bearish" if composite.score < 0 else "neutral")

    if extreme_bias:
        alerts.append(QuantAlert(
            id="extreme-composite",
            severity="warning",
            title="Extreme Composite Bias",
            direction=cs_dir,
            direction_reason=f"{composite.algo_count} algos collectively point {'UP' if cs_dir == 'bullish' else 'DOWN'} with {composite.confidence:.0f}% confidence",
            description=f"Composite score {'+'  if composite.score > 0 else ''}{composite.score:.1f} at {composite.confidence:.0f}% confidence",
            metrics={"score": composite.score, "confidence": composite.confidence},
        ))

    if vpin_high and "liquidation_waterfall" not in combo_ids and "structural_imbalance" not in combo_ids:
        d = _algo_dir(composite, "vpin")
        alerts.append(QuantAlert(
            id="vpin-toxic",
            severity="warning",
            title="Toxic Flow Detected",
            direction=d,
            direction_reason=f"Informed traders aggressively {'BUYING → expect price to push UP' if d == 'bullish' else 'SELLING → expect price to drop DOWN' if d == 'bearish' else 'active, direction unclear'}",
            description=f"VPIN={_n(vpin.get('vpin')):.3f} — institutional flow above toxicity threshold",
            metrics={"vpin": _n(vpin.get("vpin"))},
        ))

    if funding_extreme and "liquidation_waterfall" not in combo_ids and "capitulation" not in combo_ids:
        alerts.append(QuantAlert(
            id="funding-extreme",
            severity="warning",
            title="Funding Extreme",
            direction=funding_dir,
            direction_reason=funding_side,
            description=f"Funding z={funding_z:.2f} — {'longs' if funding_z > 0 else 'shorts'} {abs(funding_z):.1f}sigma above equilibrium",
            metrics={"z_score": funding_z},
        ))

    if cascade_forming and not cascade_active and "capitulation" not in combo_ids:
        d = "bearish" if funding_z > 0 else "bullish"
        alerts.append(QuantAlert(
            id="cascade-forming",
            severity="warning",
            title="Cascade Forming",
            direction=d,
            direction_reason=("Overleveraged longs at risk → if cascade triggers, expect sharp drop DOWN then reversal"
                              if funding_z > 0
                              else "Overleveraged shorts at risk → if cascade triggers, expect squeeze UP then reversal"),
            description=f"P(cascade)={_n(liq.get('p_cascade')) * 100:.0f}% — liquidation cascade probability elevated",
            metrics={"p_cascade": _n(liq.get("p_cascade"))},
        ))

    if smart_retail_split and "structural_imbalance" not in combo_ids:
        alerts.append(QuantAlert(
            id="smart-retail-split",
            severity="warning",
            title="Smart/Retail Split",
            direction=smart_dir,
            direction_reason=smart_side,
            description=f"Divergence={smart_div:.1f}pp — smart money and retail sharply disagree",
            metrics={"divergence": smart_div, "smart_long_pct": _n(smart.get("smart_long_pct")),
                     "retail_long_pct": _n(smart.get("retail_long_pct"))},
        ))

    if vol_extreme:
        alerts.append(QuantAlert(
            id="vol-extreme",
            severity="warning",
            title="Extreme Volatility",
            direction="neutral",
            direction_reason="Market too volatile for directional conviction — reduce size",
            description=f"ATR ratio={_n(vol.get('atr_ratio')):.2f} — chaotic conditions",
            metrics={"atr_ratio": _n(vol.get("atr_ratio"))},
        ))

    if sentiment_extreme:
        alerts.append(QuantAlert(
            id="sentiment-extreme",
            severity="warning",
            title="Sentiment Extreme",
            direction=sent_dir,
            direction_reason=sent_side,
            description=f"Bayesian posterior P(bull)={p_bull * 100:.0f}% — contrarian signal",
            metrics={"p_bull": p_bull, "fng": fng_value},
        ))

    if illiquidity_spike and "structural_imbalance" not in combo_ids:
        d = _algo_dir(composite, "kyle_amihud")
        alerts.append(QuantAlert(
            id="illiquidity-spike",
            severity="warning",
            title="Illiquidity Spike",
            direction=d,
            direction_reason=f"Order book thin (z={_n(kyle.get('z_lambda')):.1f}) — small orders will move price sharply",
            description=f"Kyle's Lambda z={_n(kyle.get('z_lambda')):.2f} — price impact elevated",
            metrics={"z_lambda": _n(kyle.get("z_lambda")),
                     "depth_imbalance": _n(kyle.get("depth_imbalance"))},
        ))

    # INFO-level
    if vol_compression and "gamma_squeeze" not in combo_ids:
        alerts.append(QuantAlert(
            id="vol-compression",
            severity="info",
            title="Vol Compression",
            direction="bullish",
            direction_reason="BTC historically breaks UP 60% of the time after vol compression — breakout imminent",
            description=f"ATR ratio={_n(vol.get('atr_ratio')):.3f} — volatility compressed, coiling for breakout",
            metrics={"atr_ratio": _n(vol.get("atr_ratio"))},
        ))

    if strong_consensus:
        alerts.append(QuantAlert(
            id="strong-consensus",
            severity="info",
            title="Strong Consensus",
            direction=cs_dir,
            direction_reason=f"{composite.agreement_count} of {composite.algo_count} algos agree: price likely to move {'UP' if cs_dir == 'bullish' else 'DOWN' if cs_dir == 'bearish' else 'sideways'}",
            description=f"Algo agreement {composite.agreement_count}/{composite.algo_count}, entropy={composite.entropy:.2f}",
            metrics={"agreement": composite.agreement_count, "algo_count": composite.algo_count,
                     "entropy": composite.entropy},
        ))

    if negative_gamma and "gamma_squeeze" not in combo_ids:
        alerts.append(QuantAlert(
            id="negative-gamma",
            severity="info",
            title="Negative Gamma",
            direction=c_dir,
            direction_reason=f"Dealers short gamma → any move will be amplified as dealers hedge in same direction",
            description=f"Net GEX={_n(options.get('net_gex')):.0f} — dealers must chase price",
            metrics={"net_gex": _n(options.get("net_gex"))},
        ))

    # Sort: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 9))

    return alerts
