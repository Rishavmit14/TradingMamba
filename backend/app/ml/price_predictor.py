"""
ICT-Based Price Prediction Module
Uses statistical methods and ICT principles - NO deep learning required!
100% FREE - runs on CPU
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PredictionDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class PricePrediction:
    """Price prediction result"""
    direction: PredictionDirection
    probability: float
    expected_move_percent: float
    confidence_interval: Tuple[float, float]
    timeframe: str
    key_factors: List[str]
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            'direction': self.direction.value,
            'probability': self.probability,
            'expected_move_percent': self.expected_move_percent,
            'confidence_interval': self.confidence_interval,
            'timeframe': self.timeframe,
            'key_factors': self.key_factors,
            'timestamp': self.timestamp.isoformat()
        }


class ICTPricePredictor:
    """
    Price prediction using ICT principles and statistical methods

    Features used:
    - Market structure (trend direction)
    - Premium/Discount zone position
    - Order block proximity
    - FVG presence
    - Kill zone timing
    - Liquidity levels
    - Historical patterns
    """

    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_trained = False
        self.feature_names = []

    def extract_features(self, data: pd.DataFrame, analysis: Dict = None) -> np.ndarray:
        """
        Extract ICT-based features from price data

        Parameters:
        - data: OHLCV DataFrame
        - analysis: Optional ICT analysis results

        Returns:
        - Feature array
        """
        features = []
        feature_names = []

        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values if 'volume' in data.columns else np.ones(len(data))

        # ============ Price Action Features ============

        # 1. Trend direction (simple moving average comparison)
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else sma_10
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20

        trend_short = 1 if closes[-1] > sma_10 else -1
        trend_medium = 1 if sma_10 > sma_20 else -1
        trend_long = 1 if sma_20 > sma_50 else -1

        features.extend([trend_short, trend_medium, trend_long])
        feature_names.extend(['trend_short', 'trend_medium', 'trend_long'])

        # 2. Price position in range (Premium/Discount)
        range_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
        range_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)
        range_size = range_high - range_low

        if range_size > 0:
            position_in_range = (closes[-1] - range_low) / range_size
        else:
            position_in_range = 0.5

        features.append(position_in_range)
        feature_names.append('position_in_range')

        # 3. Volatility (ATR-like)
        ranges = highs - lows
        atr = np.mean(ranges[-14:]) if len(ranges) >= 14 else np.mean(ranges)
        normalized_atr = atr / closes[-1] if closes[-1] > 0 else 0

        features.append(normalized_atr)
        feature_names.append('normalized_atr')

        # 4. Momentum (Rate of Change)
        roc_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        roc_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0

        features.extend([roc_5, roc_10])
        feature_names.extend(['roc_5', 'roc_10'])

        # 5. RSI-like momentum
        gains = []
        losses = []
        for i in range(1, min(15, len(closes))):
            change = closes[-i] - closes[-i - 1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.0001

        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        normalized_rsi = (rsi - 50) / 50  # Normalize to -1 to 1

        features.append(normalized_rsi)
        feature_names.append('rsi_normalized')

        # 6. Volume trend
        if len(volumes) >= 10:
            vol_sma = np.mean(volumes[-10:])
            vol_ratio = volumes[-1] / vol_sma if vol_sma > 0 else 1
        else:
            vol_ratio = 1

        features.append(vol_ratio)
        feature_names.append('volume_ratio')

        # 7. Candle patterns
        body = closes[-1] - data['open'].values[-1]
        upper_wick = highs[-1] - max(closes[-1], data['open'].values[-1])
        lower_wick = min(closes[-1], data['open'].values[-1]) - lows[-1]
        candle_range = highs[-1] - lows[-1]

        body_ratio = body / candle_range if candle_range > 0 else 0
        upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0

        features.extend([body_ratio, upper_wick_ratio, lower_wick_ratio])
        feature_names.extend(['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio'])

        # ============ ICT-Specific Features ============

        # 8. Break of Structure detection
        swing_high_broken = 0
        swing_low_broken = 0

        if len(highs) >= 10:
            prev_swing_high = max(highs[-10:-1])
            prev_swing_low = min(lows[-10:-1])

            if highs[-1] > prev_swing_high:
                swing_high_broken = 1
            if lows[-1] < prev_swing_low:
                swing_low_broken = 1

        features.extend([swing_high_broken, swing_low_broken])
        feature_names.extend(['swing_high_broken', 'swing_low_broken'])

        # 9. FVG detection (simplified)
        bullish_fvg = 0
        bearish_fvg = 0

        if len(data) >= 3:
            if lows[-1] > highs[-3]:
                bullish_fvg = 1
            if highs[-1] < lows[-3]:
                bearish_fvg = 1

        features.extend([bullish_fvg, bearish_fvg])
        feature_names.extend(['bullish_fvg', 'bearish_fvg'])

        # 10. Order Block proximity (simplified)
        # Check if recent bearish candle followed by bullish break
        ob_bullish = 0
        ob_bearish = 0

        if len(data) >= 3:
            # Recent bullish momentum with previous bearish candle
            if closes[-2] < data['open'].values[-2] and closes[-1] > highs[-2]:
                ob_bullish = 1
            # Recent bearish momentum with previous bullish candle
            if closes[-2] > data['open'].values[-2] and closes[-1] < lows[-2]:
                ob_bearish = 1

        features.extend([ob_bullish, ob_bearish])
        feature_names.extend(['ob_bullish', 'ob_bearish'])

        # 11. Higher timeframe bias (if analysis provided)
        htf_bias = 0
        if analysis and 'bias' in analysis:
            bias = analysis['bias']
            if isinstance(bias, dict):
                if bias.get('bias') == 'bullish':
                    htf_bias = 1
                elif bias.get('bias') == 'bearish':
                    htf_bias = -1

        features.append(htf_bias)
        feature_names.append('htf_bias')

        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)

    def predict(self, data: pd.DataFrame, analysis: Dict = None,
                timeframe: str = 'H1') -> PricePrediction:
        """
        Predict price direction using ICT principles

        Parameters:
        - data: OHLCV DataFrame
        - analysis: ICT analysis results
        - timeframe: Trading timeframe

        Returns:
        - PricePrediction object
        """
        features = self.extract_features(data, analysis)

        # Calculate prediction based on features
        factors = []
        bullish_score = 0
        bearish_score = 0

        # Weight each factor
        weights = {
            'trend': 0.25,
            'position': 0.15,
            'momentum': 0.20,
            'structure': 0.20,
            'patterns': 0.20
        }

        # Trend analysis
        trend_sum = features[0, 0] + features[0, 1] + features[0, 2]
        if trend_sum > 1:
            bullish_score += weights['trend']
            factors.append("Bullish trend alignment")
        elif trend_sum < -1:
            bearish_score += weights['trend']
            factors.append("Bearish trend alignment")

        # Position in range
        position = features[0, 3]
        if position < 0.3:
            bullish_score += weights['position']
            factors.append("Price in discount zone")
        elif position > 0.7:
            bearish_score += weights['position']
            factors.append("Price in premium zone")

        # Momentum
        rsi = features[0, 7]
        if rsi < -0.3:
            bullish_score += weights['momentum'] * 0.5
            factors.append("Oversold conditions")
        elif rsi > 0.3:
            bearish_score += weights['momentum'] * 0.5
            factors.append("Overbought conditions")

        roc = features[0, 5] + features[0, 6]
        if roc > 0.01:
            bullish_score += weights['momentum'] * 0.5
            factors.append("Positive momentum")
        elif roc < -0.01:
            bearish_score += weights['momentum'] * 0.5
            factors.append("Negative momentum")

        # Structure breaks
        if features[0, 11] == 1:  # Swing high broken
            bullish_score += weights['structure']
            factors.append("Break of structure (bullish)")
        if features[0, 12] == 1:  # Swing low broken
            bearish_score += weights['structure']
            factors.append("Break of structure (bearish)")

        # ICT patterns
        if features[0, 13] == 1:  # Bullish FVG
            bullish_score += weights['patterns'] * 0.5
            factors.append("Bullish FVG present")
        if features[0, 14] == 1:  # Bearish FVG
            bearish_score += weights['patterns'] * 0.5
            factors.append("Bearish FVG present")

        if features[0, 15] == 1:  # Bullish OB
            bullish_score += weights['patterns'] * 0.5
            factors.append("Bullish order block")
        if features[0, 16] == 1:  # Bearish OB
            bearish_score += weights['patterns'] * 0.5
            factors.append("Bearish order block")

        # HTF bias
        htf = features[0, 17]
        if htf > 0:
            bullish_score += 0.1
            factors.append("Higher timeframe bullish")
        elif htf < 0:
            bearish_score += 0.1
            factors.append("Higher timeframe bearish")

        # Calculate final prediction
        total_score = bullish_score + bearish_score
        if total_score == 0:
            total_score = 1

        bullish_prob = bullish_score / total_score
        bearish_prob = bearish_score / total_score

        if bullish_prob > 0.55:
            direction = PredictionDirection.BULLISH
            probability = bullish_prob
        elif bearish_prob > 0.55:
            direction = PredictionDirection.BEARISH
            probability = bearish_prob
        else:
            direction = PredictionDirection.NEUTRAL
            probability = max(bullish_prob, bearish_prob)

        # Estimate expected move based on ATR
        atr = features[0, 4] * data['close'].values[-1]
        expected_move = atr * 2  # Target 2 ATR move
        expected_move_percent = (expected_move / data['close'].values[-1]) * 100

        # Confidence interval
        ci_low = expected_move_percent * 0.5
        ci_high = expected_move_percent * 1.5

        return PricePrediction(
            direction=direction,
            probability=round(probability, 3),
            expected_move_percent=round(expected_move_percent, 2),
            confidence_interval=(round(ci_low, 2), round(ci_high, 2)),
            timeframe=timeframe,
            key_factors=factors[:5],  # Top 5 factors
            timestamp=datetime.utcnow()
        )

    def train_from_history(self, historical_data: List[Dict]) -> Dict:
        """
        Train model from historical signals and outcomes

        Parameters:
        - historical_data: List of {features, outcome} dicts

        Returns:
        - Training metrics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available', 'trained': False}

        if len(historical_data) < 50:
            return {'error': 'Need at least 50 samples', 'trained': False}

        X = np.array([d['features'] for d in historical_data])
        y = np.array([1 if d['outcome'] == 'win' else 0 for d in historical_data])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train ensemble model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)

        # Fit final model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        return {
            'trained': True,
            'samples': len(historical_data),
            'cv_accuracy': round(np.mean(cv_scores), 3),
            'cv_std': round(np.std(cv_scores), 3)
        }

    def get_prediction_explanation(self, prediction: PricePrediction) -> str:
        """Generate human-readable explanation of prediction"""

        direction_emoji = "🟢" if prediction.direction == PredictionDirection.BULLISH else \
                         "🔴" if prediction.direction == PredictionDirection.BEARISH else "⚪"

        explanation = f"""
{direction_emoji} **Price Prediction: {prediction.direction.value.upper()}**

**Probability:** {prediction.probability:.1%}
**Expected Move:** {prediction.expected_move_percent:.2f}%
**Confidence Range:** {prediction.confidence_interval[0]:.2f}% - {prediction.confidence_interval[1]:.2f}%
**Timeframe:** {prediction.timeframe}

**Key Factors:**
"""
        for i, factor in enumerate(prediction.key_factors, 1):
            explanation += f"  {i}. {factor}\n"

        return explanation


# Singleton instance
price_predictor = ICTPricePredictor()
