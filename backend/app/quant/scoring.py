"""Quant Combined Scorer — merges 4 layers, applies grade promotion/demotion.

Takes scores from Alpha, Microstructure, Risk, and Execution models,
produces a combined QuantScore with grade modification recommendations.
"""

from __future__ import annotations

from app.models import QuantScore, SignalGrade
from app.config import QUANT_WEIGHTS


def compute_quant_score(
    alpha_score: float,
    micro_score: float,
    risk_tradability: float,
    execution_score: float,
    alpha_components: dict | None = None,
    micro_components: dict | None = None,
    exec_components: dict | None = None,
) -> QuantScore:
    """Compute combined quant score from 4 layers.

    Weights (from config):
    - Alpha: 55% (directional conviction)
    - Microstructure: 20% (market structure quality)
    - Risk: 15% (tradability conditions)
    - Execution: 10% (timing quality)

    Grade modification:
    - combined > +40 → 3 confirmations → promote grade
    - combined > +20 → 2 confirmations
    - combined > +5  → 1 confirmation
    - combined -5..+5 → neutral
    - combined < -5  → 1 contradiction
    - combined < -20 → 2 contradictions
    - combined < -40 → 3 contradictions → demote grade
    """
    w = QUANT_WEIGHTS

    # Normalize risk_tradability from 0-100 to -100..+100 scale
    # 100 tradability = +100, 50 = 0, 0 = -100
    risk_normalized = (risk_tradability - 50) * 2

    combined = (
        alpha_score * w["alpha"]
        + micro_score * w["micro"]
        + risk_normalized * w["risk"]
        + execution_score * w["exec"]
    )

    combined = max(-100.0, min(100.0, combined))

    # Map to confirmations/contradictions
    confirmations = 0
    contradictions = 0

    if combined > 40:
        confirmations = 3
    elif combined > 20:
        confirmations = 2
    elif combined > 5:
        confirmations = 1
    elif combined < -40:
        contradictions = 3
    elif combined < -20:
        contradictions = 2
    elif combined < -5:
        contradictions = 1

    # Grade change
    grade_change = 0
    if confirmations >= 3:
        grade_change = 1   # Promote one tier
    elif confirmations >= 2:
        grade_change = 0   # No grade change, confidence boost only
    elif contradictions >= 3:
        grade_change = -1  # Demote one tier
    elif contradictions >= 2:
        grade_change = 0   # No grade change, confidence reduction only

    # Merge component breakdowns
    all_components = {}
    if alpha_components:
        all_components.update({f"alpha.{k}": v for k, v in alpha_components.items()})
    if micro_components:
        all_components.update({f"micro.{k}": v for k, v in micro_components.items()})
    if exec_components:
        all_components.update({f"exec.{k}": v for k, v in exec_components.items()})

    return QuantScore(
        alpha_score=round(alpha_score, 2),
        micro_score=round(micro_score, 2),
        risk_tradability=round(risk_tradability, 2),
        execution_score=round(execution_score, 2),
        combined_score=round(combined, 2),
        confirmations=confirmations,
        contradictions=contradictions,
        grade_change=grade_change,
        components=all_components,
    )


def apply_grade_modification(
    current_grade: SignalGrade,
    quant_score: QuantScore,
) -> tuple[SignalGrade, float]:
    """Apply quant-based grade promotion/demotion + confidence adjustment.

    Returns (new_grade, confidence_delta).

    Grade promotion: C→B, B→A (A stays A)
    Grade demotion: A→B, B→C (D stays D)
    Confidence: ±5 to ±10 per confirmation/contradiction (in addition to grade change)
    """
    grade_order = [SignalGrade.D, SignalGrade.C, SignalGrade.B, SignalGrade.A]
    current_idx = grade_order.index(current_grade)

    new_idx = current_idx + quant_score.grade_change
    new_idx = max(0, min(len(grade_order) - 1, new_idx))
    new_grade = grade_order[new_idx]

    # Confidence adjustment (separate from grade)
    confidence_delta = 0.0
    if quant_score.confirmations > 0:
        confidence_delta = quant_score.confirmations * 5.0  # +5 per confirmation
    elif quant_score.contradictions > 0:
        confidence_delta = -quant_score.contradictions * 5.0  # -5 per contradiction

    # Additional boost/penalty from combined score magnitude
    if abs(quant_score.combined_score) > 60:
        confidence_delta += 5.0 if quant_score.combined_score > 0 else -5.0

    return new_grade, confidence_delta
