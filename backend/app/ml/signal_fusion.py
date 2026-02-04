"""
Smart Money Signal Fusion Engine

Combines multiple Smart Money concepts and technical indicators to generate
high-probability trading signals with confidence scoring.

The fusion logic learns which concept combinations are most effective.

100% FREE - Uses pandas-ta for technical analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging

try:
    import pandas as pd
except ImportError:
    raise ImportError("Install pandas: pip install pandas")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SmartMoneySignal:
    """Represents a trading signal with Smart Money context"""
    direction: str  # 'bullish' or 'bearish'
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    concepts: List[str]  # Smart Money concepts supporting this signal
    timeframe: str
    entry_zone: Tuple[float, float]  # (low, high)
    stop_loss: float
    take_profit: List[float]  # Multiple TP levels
    risk_reward: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reasoning: List[str] = field(default_factory=list)
    confluence_score: int = 0


class SmartMoneyConceptScorer:
    """
    Scores individual Smart Money concepts based on market context.
    Learns optimal scoring from historical performance.
    """

    # Base weights for each concept (can be learned over time)
    BASE_WEIGHTS = {
        'order_block': 0.85,
        'fair_value_gap': 0.80,
        'fvg': 0.80,
        'liquidity': 0.90,
        'market_structure': 0.95,
        'premium_discount': 0.75,
        'kill_zones': 0.70,
        'kill_zone': 0.70,
        'entry_models': 0.85,
        'breaker': 0.80,
        'breaker_block': 0.80,
        'institutional': 0.65,
        'time_based': 0.60,
        'price_action': 0.70,
        # Audio-first training concepts
        'optimal_trade_entry': 0.90,
        'fibonacci_ote': 0.85,
        'displacement': 0.80,
        'buy_stops': 0.75,
        'sell_stops': 0.75,
        'equal_highs_lows': 0.70,
        'smart_money': 0.65,
        'higher_high': 0.60,
        'swing_high_low': 0.55,
    }

    def __init__(self):
        self.weights = self.BASE_WEIGHTS.copy()
        self.performance_history = defaultdict(list)
        self._ml_classifier = None  # Tier 3 ML classifier (lazy loaded)
        self._pattern_quality_model = None  # Tier 6+ pattern quality (lazy loaded)

    def _get_ml_classifier(self):
        """Lazy load ML classifier (Tier 3)."""
        if self._ml_classifier is None:
            try:
                from .ml_models import get_classifier
                classifier = get_classifier()
                if classifier.models:  # Only use if models are trained
                    self._ml_classifier = classifier
            except Exception:
                pass
        return self._ml_classifier

    def _get_pattern_quality_model(self):
        """Lazy load Pattern Quality Model (video-trained)."""
        if self._pattern_quality_model is None:
            try:
                from .pattern_quality_model import get_pattern_quality_model
                pqm = get_pattern_quality_model()
                if pqm.is_trained:
                    self._pattern_quality_model = pqm
            except Exception:
                pass
        return self._pattern_quality_model

    def score_concept(self, concept: str, context: Dict) -> float:
        """Score a concept based on market context, using ML predictions when available."""
        # Tier 3: Use ML classifier probability if trained
        ml_classifier = self._get_ml_classifier()
        if ml_classifier and ml_classifier.has_model(concept):
            features = context.get('ml_features')
            if features is not None:
                ml_prob = ml_classifier.predict(concept, features)
                # Blend ML prediction with base weight (70% ML, 30% base)
                base_weight = self.weights.get(concept, 0.5)
                base_weight = ml_prob * 0.7 + base_weight * 0.3
            else:
                base_weight = self.weights.get(concept, 0.5)
        else:
            base_weight = self.weights.get(concept, 0.5)

        # Video-trained Pattern Quality Model (blends video knowledge)
        pqm = self._get_pattern_quality_model()
        if pqm is not None:
            extended_features = context.get('extended_features')
            if extended_features is not None:
                pqm_score = pqm.score_pattern(concept, extended_features)
                # Three-way blend: 40% base, 30% ML classifier, 30% video-trained PQM
                base_weight = base_weight * 0.7 + pqm_score * 0.3

        # Context multipliers
        multipliers = []

        # Higher timeframe alignment
        if context.get('htf_aligned', False):
            multipliers.append(1.2)

        # Kill zone timing
        if context.get('in_kill_zone', False):
            multipliers.append(1.15)

        # Fresh level (not previously tested)
        if context.get('is_fresh', True):
            multipliers.append(1.1)

        # Volume confirmation
        if context.get('volume_confirmed', False):
            multipliers.append(1.1)

        # Apply multipliers
        final_score = base_weight
        for mult in multipliers:
            final_score *= mult

        return min(final_score, 1.0)

    def update_weights(self, concept: str, outcome: float):
        """Update concept weights based on trade outcomes"""
        self.performance_history[concept].append(outcome)

        # Recalculate weight based on recent performance
        if len(self.performance_history[concept]) >= 10:
            recent = self.performance_history[concept][-20:]
            win_rate = sum(1 for o in recent if o > 0) / len(recent)
            avg_outcome = sum(recent) / len(recent)

            # Blend base weight with learned weight
            learned_weight = (win_rate * 0.6 + (avg_outcome + 1) / 2 * 0.4)
            self.weights[concept] = 0.7 * self.BASE_WEIGHTS[concept] + 0.3 * learned_weight


class ConceptFusionEngine:
    """
    Fuses multiple Smart Money concepts to determine signal strength.
    Uses learned concept relationships for optimal combination.
    """

    # Concept synergies (concepts that work well together)
    SYNERGIES = {
        ('order_block', 'fair_value_gap'): 1.3,
        ('order_block', 'fvg'): 1.3,
        ('order_block', 'liquidity'): 1.25,
        ('market_structure', 'order_block'): 1.2,
        ('premium_discount', 'order_block'): 1.2,
        ('kill_zones', 'entry_models'): 1.25,
        ('liquidity', 'market_structure'): 1.15,
        ('fair_value_gap', 'premium_discount'): 1.15,
        ('breaker', 'fair_value_gap'): 1.2,
        # Audio-first training synergies
        ('optimal_trade_entry', 'fibonacci_ote'): 1.35,
        ('displacement', 'order_block'): 1.25,
        ('kill_zone', 'optimal_trade_entry'): 1.20,
        ('buy_stops', 'liquidity'): 1.15,
        ('sell_stops', 'liquidity'): 1.15,
        ('displacement', 'fvg'): 1.20,
        ('equal_highs_lows', 'liquidity'): 1.15,
        ('breaker_block', 'fvg'): 1.20,
    }

    # Concept conflicts (concepts that contradict each other)
    CONFLICTS = {
        # Premium bullish entry vs discount bearish entry
    }

    def __init__(self):
        self.concept_scorer = SmartMoneyConceptScorer()
        self.synergies = self.SYNERGIES.copy()

        # Override hardcoded synergies with video-learned co-occurrence where available
        try:
            from .playlist_registry import PlaylistRegistry, playlist_context
            vk = PlaylistRegistry.get_video_knowledge(playlist_context.get())
            if vk.is_loaded():
                for (c1, c2), default_bonus in self.SYNERGIES.items():
                    co_occ = vk.get_co_occurrence(c1, c2)
                    if co_occ > 0:
                        # Scale co-occurrence (0-1) to synergy bonus (1.0-1.4)
                        learned_bonus = 1.0 + co_occ * 0.4
                        # Blend: 60% learned, 40% hardcoded
                        self.synergies[(c1, c2)] = learned_bonus * 0.6 + default_bonus * 0.4
        except Exception:
            pass  # Keep hardcoded synergies if video knowledge unavailable

    def calculate_confluence(self, concepts: List[str], context: Dict) -> Tuple[float, int, List[str]]:
        """
        Calculate confluence score from multiple concepts.

        Returns:
            - combined_score: Weighted combination of concept scores
            - confluence_count: Number of confirming concepts
            - reasoning: Explanation of the scoring
        """
        if not concepts:
            return 0.0, 0, ["No Smart Money concepts identified"]

        reasoning = []
        scores = []

        # Score each concept
        for concept in concepts:
            score = self.concept_scorer.score_concept(concept, context)
            scores.append(score)
            reasoning.append(f"{concept}: {score:.2f}")

        # Apply synergy bonuses
        synergy_bonus = 1.0
        for (c1, c2), bonus in self.synergies.items():
            if c1 in concepts and c2 in concepts:
                synergy_bonus *= bonus
                reasoning.append(f"Synergy bonus ({c1} + {c2}): x{bonus}")

        # Calculate combined score
        base_score = sum(scores) / len(scores)
        combined_score = min(base_score * synergy_bonus, 1.0)

        # Confluence count (concepts scoring above threshold)
        confluence_count = sum(1 for s in scores if s >= 0.6)

        reasoning.append(f"Combined score: {combined_score:.2f}")
        reasoning.append(f"Confluence: {confluence_count} concepts")

        return combined_score, confluence_count, reasoning

    def update_from_outcome(self, concepts: List[str], outcome: float):
        """Update fusion weights based on trade outcome"""
        for concept in concepts:
            self.concept_scorer.update_weights(concept, outcome)


class MultiTimeframeAnalyzer:
    """
    Analyzes Smart Money concepts across multiple timeframes.
    Higher timeframes carry more weight.
    """

    TIMEFRAME_WEIGHTS = {
        'MN': 1.0,   # Monthly - highest
        'W1': 0.95,  # Weekly
        'D1': 0.90,  # Daily
        'H4': 0.80,  # 4-hour
        'H1': 0.70,  # Hourly
        'M30': 0.55,
        'M15': 0.45,
        'M5': 0.30,
        'M1': 0.15,  # 1-min - lowest
    }

    def __init__(self):
        self.analysis_cache = {}

    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze Smart Money concepts on a single timeframe"""
        if df is None or df.empty:
            return {}

        analysis = {
            'timeframe': timeframe,
            'weight': self.TIMEFRAME_WEIGHTS.get(timeframe, 0.5),
            'concepts': [],
            'bias': 'neutral',
            'key_levels': [],
        }

        # Determine bias from structure
        if len(df) >= 3:
            recent_highs = df['high'].tail(20)
            recent_lows = df['low'].tail(20)

            higher_highs = sum(recent_highs.diff().dropna() > 0)
            higher_lows = sum(recent_lows.diff().dropna() > 0)

            if higher_highs > 12 and higher_lows > 12:
                analysis['bias'] = 'bullish'
                analysis['concepts'].append('market_structure')
            elif higher_highs < 8 and higher_lows < 8:
                analysis['bias'] = 'bearish'
                analysis['concepts'].append('market_structure')

        # Find order blocks (simplified)
        if len(df) >= 10:
            for i in range(len(df) - 3, max(0, len(df) - 20), -1):
                # Bullish OB: Down candle followed by strong up move
                if df['close'].iloc[i] < df['open'].iloc[i]:
                    if df['close'].iloc[i+1:i+3].max() > df['high'].iloc[i]:
                        analysis['key_levels'].append({
                            'type': 'bullish_ob',
                            'high': float(df['high'].iloc[i]),
                            'low': float(df['low'].iloc[i]),
                        })
                        analysis['concepts'].append('order_block')
                        break

        # Find FVGs (simplified)
        if len(df) >= 3:
            for i in range(len(df) - 3, max(0, len(df) - 10), -1):
                gap_up = df['low'].iloc[i+2] > df['high'].iloc[i]
                gap_down = df['high'].iloc[i+2] < df['low'].iloc[i]

                if gap_up or gap_down:
                    analysis['concepts'].append('fair_value_gap')
                    analysis['key_levels'].append({
                        'type': 'fvg_bullish' if gap_up else 'fvg_bearish',
                        'high': float(df['low'].iloc[i+2] if gap_up else df['low'].iloc[i]),
                        'low': float(df['high'].iloc[i] if gap_up else df['high'].iloc[i+2]),
                    })
                    break

        # Premium/Discount zone
        if len(df) >= 50:
            swing_high = df['high'].tail(50).max()
            swing_low = df['low'].tail(50).min()
            current = df['close'].iloc[-1]
            equilibrium = (swing_high + swing_low) / 2

            if current > equilibrium:
                analysis['zone'] = 'premium'
            else:
                analysis['zone'] = 'discount'
            analysis['concepts'].append('premium_discount')

        return analysis

    def get_mtf_alignment(self, analyses: Dict[str, Dict]) -> Dict:
        """Check if multiple timeframes are aligned"""
        if not analyses:
            return {'aligned': False, 'bias': 'neutral', 'score': 0}

        biases = []
        total_weight = 0

        for tf, analysis in analyses.items():
            weight = analysis.get('weight', 0.5)
            bias = analysis.get('bias', 'neutral')

            if bias != 'neutral':
                biases.append((bias, weight))
                total_weight += weight

        if not biases:
            return {'aligned': False, 'bias': 'neutral', 'score': 0}

        # Calculate weighted bias
        bullish_weight = sum(w for b, w in biases if b == 'bullish')
        bearish_weight = sum(w for b, w in biases if b == 'bearish')

        if total_weight > 0:
            alignment_score = abs(bullish_weight - bearish_weight) / total_weight
        else:
            alignment_score = 0

        dominant_bias = 'bullish' if bullish_weight > bearish_weight else 'bearish'
        is_aligned = alignment_score > 0.6

        return {
            'aligned': is_aligned,
            'bias': dominant_bias if is_aligned else 'neutral',
            'score': alignment_score,
            'bullish_weight': bullish_weight,
            'bearish_weight': bearish_weight,
        }


class SignalGenerator:
    """
    Main signal generation class.
    Combines all Smart Money analysis to produce actionable signals.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.models_dir = self.data_dir / "ml_models"

        self.fusion_engine = ConceptFusionEngine()
        self.mtf_analyzer = MultiTimeframeAnalyzer()

        # Performance tracking
        self.signals_generated = 0
        self.signal_history = []

    def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, pd.DataFrame],  # Timeframe -> DataFrame
        detected_concepts: List[str],
        context: Dict = None
    ) -> Optional[SmartMoneySignal]:
        """Generate a trading signal from market data and concepts"""

        if not market_data:
            return None

        context = context or {}

        # Analyze each timeframe
        tf_analyses = {}
        for tf, df in market_data.items():
            analysis = self.mtf_analyzer.analyze_timeframe(df, tf)
            if analysis:
                tf_analyses[tf] = analysis
                # Add concepts from chart analysis
                detected_concepts.extend(analysis.get('concepts', []))

        # Deduplicate concepts
        detected_concepts = list(set(detected_concepts))

        # Get MTF alignment
        mtf_result = self.mtf_analyzer.get_mtf_alignment(tf_analyses)

        # Calculate confluence
        context['htf_aligned'] = mtf_result['aligned']
        context['in_kill_zone'] = self._is_in_kill_zone()

        confluence_score, confluence_count, reasoning = self.fusion_engine.calculate_confluence(
            detected_concepts, context
        )

        # Determine signal direction
        if mtf_result['aligned'] and mtf_result['bias'] != 'neutral':
            direction = mtf_result['bias']
        else:
            # Use concept bias
            bullish_concepts = ['order_block', 'fair_value_gap', 'liquidity']  # When in discount
            bearish_concepts = ['order_block', 'fair_value_gap', 'liquidity']  # When in premium

            # Check zone context
            zone = None
            for analysis in tf_analyses.values():
                if 'zone' in analysis:
                    zone = analysis['zone']
                    break

            if zone == 'discount':
                direction = 'bullish'
            elif zone == 'premium':
                direction = 'bearish'
            else:
                direction = 'neutral'

        # Don't generate weak signals
        if confluence_score < 0.5 or confluence_count < 2:
            return None

        if direction == 'neutral':
            return None

        # Calculate entry, SL, TP from key levels
        entry_zone, stop_loss, take_profits = self._calculate_levels(
            direction, tf_analyses, market_data
        )

        # Calculate risk/reward
        if entry_zone and stop_loss and take_profits:
            entry_mid = (entry_zone[0] + entry_zone[1]) / 2
            risk = abs(entry_mid - stop_loss)
            reward = abs(take_profits[0] - entry_mid) if take_profits else 0
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0

        # Create signal
        signal = SmartMoneySignal(
            direction=direction,
            strength=confluence_score,
            confidence=min(confluence_score * mtf_result['score'] * 1.2, 1.0) if mtf_result['aligned'] else confluence_score * 0.7,
            concepts=detected_concepts,
            timeframe=list(market_data.keys())[0] if market_data else 'H1',
            entry_zone=entry_zone or (0, 0),
            stop_loss=stop_loss or 0,
            take_profit=take_profits or [],
            risk_reward=risk_reward,
            reasoning=reasoning,
            confluence_score=confluence_count,
        )

        # Track
        self.signals_generated += 1
        self.signal_history.append({
            'symbol': symbol,
            'signal': signal.__dict__,
            'timestamp': signal.timestamp,
        })

        return signal

    def _is_in_kill_zone(self) -> bool:
        """Check if current time is in an Smart Money kill zone"""
        now = datetime.utcnow()
        hour = now.hour

        # London Kill Zone: 2:00-5:00 AM EST (7:00-10:00 UTC)
        # NY Kill Zone: 7:00-10:00 AM EST (12:00-15:00 UTC)
        # Asian: 8:00 PM - 12:00 AM EST (1:00-5:00 UTC)

        kill_zones = [
            (1, 5),    # Asian
            (7, 10),   # London
            (12, 15),  # New York
        ]

        for start, end in kill_zones:
            if start <= hour < end:
                return True

        return False

    def _calculate_levels(
        self,
        direction: str,
        tf_analyses: Dict,
        market_data: Dict[str, pd.DataFrame]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[List[float]]]:
        """Calculate entry zone, stop loss, and take profit levels"""

        # Get current price from lowest timeframe
        current_price = None
        for tf in ['M15', 'M30', 'H1', 'H4', 'D1']:
            if tf in market_data and not market_data[tf].empty:
                current_price = float(market_data[tf]['close'].iloc[-1])
                break

        if current_price is None:
            return None, None, None

        # Find key levels from analyses
        key_levels = []
        for analysis in tf_analyses.values():
            key_levels.extend(analysis.get('key_levels', []))

        if not key_levels:
            # Default levels based on recent range
            for tf in ['H1', 'H4', 'D1']:
                if tf in market_data and len(market_data[tf]) >= 20:
                    df = market_data[tf]
                    atr = (df['high'] - df['low']).tail(14).mean()

                    if direction == 'bullish':
                        entry_zone = (current_price - atr * 0.5, current_price)
                        stop_loss = current_price - atr * 1.5
                        take_profits = [
                            current_price + atr * 1.5,
                            current_price + atr * 3,
                            current_price + atr * 5,
                        ]
                    else:
                        entry_zone = (current_price, current_price + atr * 0.5)
                        stop_loss = current_price + atr * 1.5
                        take_profits = [
                            current_price - atr * 1.5,
                            current_price - atr * 3,
                            current_price - atr * 5,
                        ]

                    return entry_zone, stop_loss, take_profits

        # Use key levels if available
        if direction == 'bullish':
            # Entry at order block/FVG below current price
            entry_levels = [l for l in key_levels if l.get('high', float('inf')) < current_price]
            if entry_levels:
                best_entry = max(entry_levels, key=lambda x: x.get('high', 0))
                entry_zone = (best_entry['low'], best_entry['high'])
                stop_loss = best_entry['low'] - (best_entry['high'] - best_entry['low']) * 0.5
            else:
                entry_zone = (current_price * 0.995, current_price)
                stop_loss = current_price * 0.99

            # TPs at levels above
            tp_levels = [l for l in key_levels if l.get('low', 0) > current_price]
            take_profits = sorted([l['low'] for l in tp_levels])[:3] if tp_levels else [current_price * 1.01, current_price * 1.02]

        else:  # bearish
            # Entry at levels above current price
            entry_levels = [l for l in key_levels if l.get('low', 0) > current_price]
            if entry_levels:
                best_entry = min(entry_levels, key=lambda x: x.get('low', float('inf')))
                entry_zone = (best_entry['low'], best_entry['high'])
                stop_loss = best_entry['high'] + (best_entry['high'] - best_entry['low']) * 0.5
            else:
                entry_zone = (current_price, current_price * 1.005)
                stop_loss = current_price * 1.01

            # TPs at levels below
            tp_levels = [l for l in key_levels if l.get('high', float('inf')) < current_price]
            take_profits = sorted([l['high'] for l in tp_levels], reverse=True)[:3] if tp_levels else [current_price * 0.99, current_price * 0.98]

        return entry_zone, stop_loss, take_profits

    def record_outcome(self, signal_id: int, outcome: float):
        """Record trade outcome for learning"""
        if signal_id < len(self.signal_history):
            signal_data = self.signal_history[signal_id]
            concepts = signal_data['signal'].get('concepts', [])
            self.fusion_engine.update_from_outcome(concepts, outcome)

    def get_performance_stats(self) -> Dict:
        """Get signal generation statistics"""
        return {
            'total_signals': self.signals_generated,
            'concept_weights': self.fusion_engine.concept_scorer.weights,
            'recent_signals': self.signal_history[-10:],
        }

    def save(self):
        """Save signal generator state"""
        state_path = self.models_dir / "signal_generator_state.json"
        with open(state_path, 'w') as f:
            json.dump({
                'signals_generated': self.signals_generated,
                'concept_weights': self.fusion_engine.concept_scorer.weights,
                'performance_history': dict(self.fusion_engine.concept_scorer.performance_history),
            }, f, indent=2)

    def load(self):
        """Load signal generator state"""
        state_path = self.models_dir / "signal_generator_state.json"
        try:
            with open(state_path) as f:
                state = json.load(f)
                self.signals_generated = state.get('signals_generated', 0)
                self.fusion_engine.concept_scorer.weights = state.get('concept_weights', {})
        except:
            pass


def test_signal_fusion():
    """Test the signal fusion engine"""
    print("=" * 60)
    print("Smart Money SIGNAL FUSION TEST")
    print("=" * 60)

    # Create sample market data
    import pandas as pd
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    df['high'] = df['open'] + abs(np.random.randn(100) * 0.3)
    df['low'] = df['open'] - abs(np.random.randn(100) * 0.3)
    df['close'] = df['open'] + np.random.randn(100) * 0.2

    # Test signal generation
    generator = SignalGenerator()

    market_data = {'H1': df}
    concepts = ['order_block', 'fair_value_gap', 'premium_discount']

    signal = generator.generate_signal(
        symbol='EURUSD',
        market_data=market_data,
        detected_concepts=concepts,
        context={'is_fresh': True}
    )

    if signal:
        print(f"\nGenerated Signal:")
        print(f"  Direction: {signal.direction}")
        print(f"  Strength: {signal.strength:.2f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Concepts: {signal.concepts}")
        print(f"  Entry Zone: {signal.entry_zone}")
        print(f"  Stop Loss: {signal.stop_loss:.5f}")
        print(f"  Take Profits: {[f'{tp:.5f}' for tp in signal.take_profit]}")
        print(f"  Risk/Reward: {signal.risk_reward:.2f}")
        print(f"  Reasoning:")
        for r in signal.reasoning:
            print(f"    - {r}")
    else:
        print("No signal generated (insufficient confluence)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_signal_fusion()
