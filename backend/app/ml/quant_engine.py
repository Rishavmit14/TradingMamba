"""
Quant Engine - Full Quantitative Trading Stack (Tier 5)

Brings together all tiers into a hedge-fund-grade signal system:
- Regime Detection (HMM-based)
- Multi-Asset Correlation Analysis
- Risk Management (Kelly Criterion, position sizing)
- Final Signal Generation (all tiers combined)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_cache import get_data_cache

logger = logging.getLogger(__name__)

# Market regime definitions
REGIMES = {
    0: 'trending_up',
    1: 'trending_down',
    2: 'range_bound',
    3: 'high_volatility',
}

# Pattern effectiveness by regime (learned from backtesting)
REGIME_PATTERN_WEIGHTS = {
    'trending_up': {
        'order_block': 1.2, 'fvg': 0.9, 'displacement': 1.3,
        'optimal_trade_entry': 1.4, 'breaker_block': 0.8,
        'buy_stops': 1.1, 'sell_stops': 0.7,
    },
    'trending_down': {
        'order_block': 1.2, 'fvg': 0.9, 'displacement': 1.3,
        'optimal_trade_entry': 1.4, 'breaker_block': 0.8,
        'buy_stops': 0.7, 'sell_stops': 1.1,
    },
    'range_bound': {
        'order_block': 0.8, 'fvg': 1.3, 'displacement': 0.6,
        'optimal_trade_entry': 0.9, 'breaker_block': 1.2,
        'buy_stops': 1.0, 'sell_stops': 1.0,
    },
    'high_volatility': {
        'order_block': 0.7, 'fvg': 0.6, 'displacement': 0.5,
        'optimal_trade_entry': 0.8, 'breaker_block': 0.6,
        'buy_stops': 1.3, 'sell_stops': 1.3,
    },
}


class RegimeDetector:
    """
    Market regime detection using statistical methods.

    Classifies market into one of 4 regimes:
    - Trending Up: positive returns, low-to-moderate volatility
    - Trending Down: negative returns, moderate-to-high volatility
    - Range Bound: near-zero returns, low volatility
    - High Volatility: extreme moves, high volatility
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self._hmm = None
        self._use_hmm = False

        # Try to load hmmlearn
        try:
            from hmmlearn.hmm import GaussianHMM
            self._hmm_class = GaussianHMM
            self._use_hmm = True
            logger.info("HMM-based regime detection available")
        except ImportError:
            logger.info("hmmlearn not installed, using statistical regime detection")

    def detect(self, data: pd.DataFrame) -> Dict:
        """
        Detect current market regime.

        Args:
            data: OHLCV DataFrame

        Returns:
            Dict with regime, confidence, and statistics
        """
        if len(data) < self.lookback:
            return {'regime': 'unknown', 'confidence': 0}

        closes = data['close'].values.astype(float)
        returns = np.diff(np.log(closes))

        if len(returns) < self.lookback:
            return {'regime': 'unknown', 'confidence': 0}

        recent_returns = returns[-self.lookback:]
        recent_vol = np.std(recent_returns) * np.sqrt(252)  # Annualized

        if self._use_hmm:
            return self._hmm_detect(returns, recent_returns, recent_vol)
        else:
            return self._statistical_detect(recent_returns, recent_vol, closes)

    def _statistical_detect(
        self,
        returns: np.ndarray,
        vol: float,
        closes: np.ndarray,
    ) -> Dict:
        """Statistical regime detection (no HMM dependency)."""
        avg_return = np.mean(returns)
        trend_strength = avg_return / (np.std(returns) + 1e-10)

        # Classify based on return and volatility characteristics
        vol_threshold_high = 0.4  # 40% annualized
        vol_threshold_low = 0.15  # 15% annualized

        if vol > vol_threshold_high:
            regime = 'high_volatility'
            confidence = min(vol / vol_threshold_high, 1.0)
        elif abs(trend_strength) > 0.15:
            regime = 'trending_up' if trend_strength > 0 else 'trending_down'
            confidence = min(abs(trend_strength) / 0.3, 1.0)
        else:
            regime = 'range_bound'
            confidence = 1.0 - abs(trend_strength) / 0.15

        # Additional statistics
        recent_20 = closes[-20:]
        ma_20 = np.mean(recent_20)
        ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else ma_20

        return {
            'regime': regime,
            'confidence': round(confidence, 3),
            'statistics': {
                'avg_daily_return': round(float(avg_return * 100), 4),
                'annualized_vol': round(float(vol * 100), 2),
                'trend_strength': round(float(trend_strength), 4),
                'price_vs_ma20': round(float((closes[-1] - ma_20) / ma_20 * 100), 2),
                'price_vs_ma50': round(float((closes[-1] - ma_50) / ma_50 * 100), 2),
                'ma20_vs_ma50': round(float((ma_20 - ma_50) / ma_50 * 100), 2),
            },
            'method': 'statistical',
        }

    def _hmm_detect(
        self,
        all_returns: np.ndarray,
        recent_returns: np.ndarray,
        vol: float,
    ) -> Dict:
        """HMM-based regime detection."""
        try:
            # Prepare features: returns + rolling volatility
            features = np.column_stack([
                all_returns,
                pd.Series(all_returns).rolling(10).std().fillna(method='bfill').values,
            ])

            model = self._hmm_class(
                n_components=4, covariance_type='full',
                n_iter=100, random_state=42
            )
            model.fit(features)

            # Predict current state
            states = model.predict(features)
            current_state = int(states[-1])

            # Map HMM states to regime names based on characteristics
            state_returns = {}
            state_vols = {}
            for s in range(4):
                mask = states == s
                if np.sum(mask) > 0:
                    state_returns[s] = np.mean(all_returns[mask])
                    state_vols[s] = np.std(all_returns[mask])

            # Sort states by volatility to assign regime labels
            sorted_states = sorted(state_vols.keys(), key=lambda s: state_vols.get(s, 0))
            regime_map = {}
            for i, s in enumerate(sorted_states):
                if i == 3:  # Highest vol
                    regime_map[s] = 'high_volatility'
                elif state_returns.get(s, 0) > 0.0005:
                    regime_map[s] = 'trending_up'
                elif state_returns.get(s, 0) < -0.0005:
                    regime_map[s] = 'trending_down'
                else:
                    regime_map[s] = 'range_bound'

            regime = regime_map.get(current_state, 'unknown')

            # Confidence from state probabilities
            state_probs = model.predict_proba(features[-1:])
            confidence = float(np.max(state_probs))

            return {
                'regime': regime,
                'confidence': round(confidence, 3),
                'state': current_state,
                'state_probabilities': {
                    regime_map.get(i, f'state_{i}'): round(float(p), 3)
                    for i, p in enumerate(state_probs[0])
                },
                'method': 'hmm',
            }

        except Exception as e:
            logger.warning(f"HMM detection failed, falling back to statistical: {e}")
            return self._statistical_detect(
                recent_returns, vol,
                np.exp(np.cumsum(all_returns)) * 100  # Approximate close prices
            )

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get pattern weight adjustments for a regime."""
        return REGIME_PATTERN_WEIGHTS.get(regime, {})


class CorrelationAnalyzer:
    """
    Multi-asset correlation analysis.

    Tracks correlations between crypto, equities, USD, and gold
    to identify divergences and confirm/deny signals.
    """

    DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SPY', 'GOLD', 'DXY']

    def __init__(self):
        self.data_cache = get_data_cache()

    def get_correlation_matrix(
        self,
        symbols: List[str] = None,
        timeframe: str = 'D1',
        lookback_days: int = 90,
    ) -> Dict:
        """
        Compute correlation matrix between multiple assets.

        Returns:
            Dict with correlation matrix, divergences, and analysis
        """
        symbols = symbols or self.DEFAULT_SYMBOLS

        # Fetch data for all symbols
        all_returns = {}
        for symbol in symbols:
            try:
                data = self.data_cache.get_ohlcv(symbol, timeframe, lookback_days)
                if len(data) >= 20:
                    closes = data['close'].values.astype(float)
                    returns = np.diff(np.log(closes))
                    all_returns[symbol] = returns
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        if len(all_returns) < 2:
            return {"error": "Need at least 2 symbols with data"}

        # Align lengths
        min_len = min(len(r) for r in all_returns.values())
        aligned = {s: r[-min_len:] for s, r in all_returns.items()}

        # Build correlation matrix
        symbols_with_data = list(aligned.keys())
        n = len(symbols_with_data)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    corr = np.corrcoef(
                        aligned[symbols_with_data[i]],
                        aligned[symbols_with_data[j]]
                    )[0, 1]
                    matrix[i][j] = round(float(corr), 4)

        # Find divergences (unusual correlation changes)
        divergences = []
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = symbols_with_data[i], symbols_with_data[j]
                corr = matrix[i][j]

                # Check recent 10-day vs full period correlation
                recent = np.corrcoef(
                    aligned[s1][-10:],
                    aligned[s2][-10:]
                )[0, 1] if min_len >= 10 else corr

                if abs(recent - corr) > 0.3:
                    divergences.append({
                        'pair': f"{s1}/{s2}",
                        'full_period_corr': round(float(corr), 4),
                        'recent_10d_corr': round(float(recent), 4),
                        'divergence': round(float(recent - corr), 4),
                        'interpretation': self._interpret_divergence(s1, s2, corr, recent),
                    })

        return {
            'symbols': symbols_with_data,
            'matrix': matrix.tolist(),
            'period_days': lookback_days,
            'data_points': min_len,
            'divergences': divergences,
        }

    def _interpret_divergence(self, s1: str, s2: str, full_corr: float, recent_corr: float) -> str:
        """Interpret a correlation divergence."""
        shift = recent_corr - full_corr
        if shift > 0.3:
            return f"{s1} and {s2} becoming more correlated (risk-on alignment)"
        elif shift < -0.3:
            return f"{s1} and {s2} decoupling (potential regime shift)"
        return "Minor divergence"

    def check_signal_confirmation(
        self,
        symbol: str,
        direction: str,
        related_symbols: Dict[str, str] = None,
    ) -> Dict:
        """
        Check if correlated assets confirm a signal direction.

        Args:
            symbol: Main symbol
            direction: 'bullish' or 'bearish'
            related_symbols: Dict of related symbol -> expected direction
        """
        if not related_symbols:
            # Default relationships
            if symbol in ('BTCUSDT', 'ETHUSDT'):
                related_symbols = {
                    'SPY': 'same',   # Crypto tends to follow risk assets
                    'DXY': 'inverse',  # Inverse to USD strength
                    'GOLD': 'same',    # Both risk-off hedges
                }
            else:
                return {"confirmed": True, "reason": "no_correlation_data"}

        confirmations = 0
        contradictions = 0
        details = []

        for related, relationship in related_symbols.items():
            try:
                data = self.data_cache.get_ohlcv(related, 'D1', 5)
                if len(data) >= 2:
                    recent_return = (float(data['close'].iloc[-1]) - float(data['close'].iloc[-2])) / float(data['close'].iloc[-2])

                    related_direction = 'bullish' if recent_return > 0 else 'bearish'

                    if relationship == 'same':
                        expected = direction
                    elif relationship == 'inverse':
                        expected = 'bearish' if direction == 'bullish' else 'bullish'
                    else:
                        continue

                    if related_direction == expected:
                        confirmations += 1
                        details.append(f"{related}: confirms ({related_direction})")
                    else:
                        contradictions += 1
                        details.append(f"{related}: contradicts ({related_direction})")
            except Exception:
                pass

        total = confirmations + contradictions
        confirmed = confirmations > contradictions if total > 0 else True

        return {
            'confirmed': confirmed,
            'confirmation_ratio': confirmations / total if total > 0 else 1.0,
            'confirmations': confirmations,
            'contradictions': contradictions,
            'details': details,
        }


class RiskManager:
    """
    Position sizing and risk management.

    Uses Kelly Criterion, ATR-based stops, and portfolio limits.
    """

    def __init__(self, max_risk_pct: float = 2.0, max_positions: int = 5):
        self.max_risk_pct = max_risk_pct
        self.max_positions = max_positions

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float = 0.5,
        avg_rr: float = 1.5,
    ) -> Dict:
        """
        Calculate optimal position size.

        Uses Kelly Criterion with half-Kelly for safety.
        """
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return {"error": "Stop loss equals entry price"}

        # Kelly Criterion: f* = (bp - q) / b
        # b = avg win / avg loss (R:R ratio)
        # p = win probability, q = 1 - p
        b = avg_rr
        p = win_rate
        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0

        # Half Kelly for safety
        half_kelly = max(kelly / 2, 0)

        # Cap at max risk
        risk_fraction = min(half_kelly, self.max_risk_pct / 100)

        # Dollar risk
        dollar_risk = account_balance * risk_fraction

        # Position size
        position_size = dollar_risk / risk_per_unit if risk_per_unit > 0 else 0

        return {
            'position_size': round(position_size, 6),
            'dollar_risk': round(dollar_risk, 2),
            'risk_pct': round(risk_fraction * 100, 2),
            'kelly_fraction': round(kelly, 4),
            'half_kelly': round(half_kelly, 4),
            'win_rate_used': round(win_rate, 4),
            'rr_used': round(avg_rr, 2),
        }

    def dynamic_stop_loss(
        self,
        data: pd.DataFrame,
        direction: str,
        atr_multiplier: float = 1.5,
    ) -> Dict:
        """
        Calculate dynamic stop loss based on ATR and market structure.
        """
        if len(data) < 20:
            return {"error": "Insufficient data"}

        closes = data['close'].values.astype(float)
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)

        # ATR calculation
        tr = np.maximum(
            highs[-14:] - lows[-14:],
            np.maximum(
                np.abs(highs[-14:] - closes[-15:-1]),
                np.abs(lows[-14:] - closes[-15:-1])
            )
        )
        atr = np.mean(tr)

        current_price = closes[-1]

        if direction == 'bullish':
            # Stop below current price
            atr_stop = current_price - (atr * atr_multiplier)
            # Also check recent swing low
            recent_low = np.min(lows[-20:])
            stop_loss = min(atr_stop, recent_low)
            # Take profits at 1.5x, 2x, 3x risk
            risk = current_price - stop_loss
            take_profits = [
                current_price + risk * 1.5,
                current_price + risk * 2.0,
                current_price + risk * 3.0,
            ]
        else:
            # Stop above current price
            atr_stop = current_price + (atr * atr_multiplier)
            recent_high = np.max(highs[-20:])
            stop_loss = max(atr_stop, recent_high)
            risk = stop_loss - current_price
            take_profits = [
                current_price - risk * 1.5,
                current_price - risk * 2.0,
                current_price - risk * 3.0,
            ]

        return {
            'stop_loss': round(float(stop_loss), 2),
            'atr': round(float(atr), 2),
            'atr_multiplier': atr_multiplier,
            'take_profits': [round(float(tp), 2) for tp in take_profits],
            'risk_amount': round(float(abs(current_price - stop_loss)), 2),
            'risk_pct': round(float(abs(current_price - stop_loss) / current_price * 100), 2),
        }


class QuantEngine:
    """
    Full quantitative trading engine combining all tiers.

    Flow:
    1. Detect market regime (this module)
    2. Get pattern detections (SmartMoneyAnalyzer)
    3. Get ML predictions (Tier 3+4)
    4. Apply regime-adjusted weights
    5. Check correlation confirmation
    6. Calculate position size and risk
    7. Generate final signal
    """

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.risk_manager = RiskManager()
        self.data_cache = get_data_cache()

    def detect_regime(self, symbol: str, timeframe: str = 'D1') -> Dict:
        """Detect current market regime for a symbol."""
        data = self.data_cache.get_ohlcv(symbol, timeframe, lookback_days=120)
        regime = self.regime_detector.detect(data)
        regime['symbol'] = symbol
        regime['timeframe'] = timeframe
        regime['pattern_weight_adjustments'] = self.regime_detector.get_regime_weights(
            regime.get('regime', 'range_bound')
        )
        return regime

    def get_correlation_matrix(
        self,
        symbols: List[str],
        timeframe: str = 'D1',
        lookback_days: int = 90,
    ) -> Dict:
        """Get multi-asset correlation matrix."""
        return self.correlation_analyzer.get_correlation_matrix(symbols, timeframe, lookback_days)

    def generate_signal(self, symbol: str, timeframe: str = 'D1') -> Dict:
        """
        Generate full quant signal combining all tiers.

        Returns comprehensive signal with regime context,
        ML predictions, correlation confirmation, and risk management.
        """
        logger.info(f"Generating quant signal: {symbol} {timeframe}")

        # Step 1: Get market data
        data = self.data_cache.get_ohlcv(symbol, timeframe, lookback_days=365)
        if len(data) < 100:
            return {"error": f"Insufficient data: {len(data)} bars"}

        # Step 2: Detect regime
        regime = self.regime_detector.detect(data)

        # Step 3: Run pattern analysis
        from ..services.smart_money_analyzer import SmartMoneyAnalyzer
        analyzer = SmartMoneyAnalyzer(use_ml=True)
        analysis = analyzer.analyze(data.tail(200))

        # Step 4: Get ML predictions (Tier 3+4)
        ml_predictions = {}
        try:
            from .feature_engineering import get_feature_engineer
            from .deep_models import get_ensemble_predictor

            engineer = get_feature_engineer()
            features = engineer.extract_features(data, len(data) - 1)

            # Build sequence for Tier 4
            if len(data) >= 60:
                feature_seq = engineer.extract_features_batch(data, len(data) - 10)
            else:
                feature_seq = None

            predictor = get_ensemble_predictor()
            for ptype in analysis.ml_patterns_used:
                pred = predictor.predict(ptype, features, feature_seq)
                ml_predictions[ptype] = pred
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")

        # Step 5: Apply regime weights
        regime_weights = self.regime_detector.get_regime_weights(
            regime.get('regime', 'range_bound')
        )
        adjusted_confidence = {}
        for ptype, score in analysis.ml_confidence_scores.items():
            weight_adj = regime_weights.get(ptype, 1.0)
            adjusted_confidence[ptype] = round(score * weight_adj, 4)

        # Step 6: Check correlation confirmation
        direction = analysis.bias.value
        correlation_check = self.correlation_analyzer.check_signal_confirmation(
            symbol, direction
        )

        # Step 7: Calculate risk parameters
        current_price = float(data['close'].iloc[-1])
        risk_params = self.risk_manager.dynamic_stop_loss(data, direction)

        # Step 8: Position sizing (using backtest win rate if available)
        win_rate = 0.5
        avg_rr = 1.5
        try:
            from .backtester import get_backtester
            perf = get_backtester().get_aggregate_performance()
            if perf:
                avg_wr = np.mean([p['avg_win_rate'] for p in perf.values() if p.get('avg_win_rate', 0) > 0])
                avg_r = np.mean([p['avg_rr'] for p in perf.values() if p.get('avg_rr', 0) > 0])
                if avg_wr > 0:
                    win_rate = avg_wr / 100
                if avg_r > 0:
                    avg_rr = avg_r
        except Exception:
            pass

        position_sizing = self.risk_manager.calculate_position_size(
            account_balance=10000,  # Default, should be configurable
            entry_price=current_price,
            stop_loss=risk_params.get('stop_loss', current_price * 0.98),
            win_rate=win_rate,
            avg_rr=avg_rr,
        )

        # Step 9: Final signal confidence
        base_confidence = analysis.bias_confidence

        # Boost/penalize based on regime
        regime_factor = 1.0
        if regime.get('regime') == 'high_volatility':
            regime_factor = 0.7  # Reduce confidence in high vol
        elif regime.get('regime') in ('trending_up', 'trending_down'):
            if (regime['regime'] == 'trending_up' and direction == 'bullish') or \
               (regime['regime'] == 'trending_down' and direction == 'bearish'):
                regime_factor = 1.2  # Boost when aligned with trend

        # Correlation factor
        corr_factor = 1.1 if correlation_check.get('confirmed') else 0.85

        # ML factor
        ml_factor = 1.0
        if ml_predictions:
            avg_ml = np.mean([p.get('ensemble_prob', 0.5) for p in ml_predictions.values()])
            ml_factor = 0.8 + avg_ml * 0.4  # Range: 0.8-1.2

        final_confidence = min(base_confidence * regime_factor * corr_factor * ml_factor, 1.0)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'confidence': round(final_confidence, 4),
            'current_price': current_price,
            'regime': regime,
            'analysis': {
                'bias': analysis.bias.value,
                'bias_reasoning': analysis.bias_reasoning,
                'patterns_detected': analysis.ml_patterns_used,
                'patterns_not_learned': analysis.ml_patterns_not_learned,
                'confidence_scores': analysis.ml_confidence_scores,
                'regime_adjusted_scores': adjusted_confidence,
            },
            'ml_predictions': ml_predictions,
            'correlation': correlation_check,
            'risk': risk_params,
            'position_sizing': position_sizing,
            'confidence_factors': {
                'base': round(base_confidence, 4),
                'regime_factor': round(regime_factor, 2),
                'correlation_factor': round(corr_factor, 2),
                'ml_factor': round(ml_factor, 2),
                'final': round(final_confidence, 4),
            },
            'generated_at': datetime.utcnow().isoformat(),
        }

    def get_dashboard(self) -> Dict:
        """Get full quant dashboard."""
        dashboard = {
            'timestamp': datetime.utcnow().isoformat(),
            'tiers': {},
        }

        # Tier 1: Backtest data
        try:
            from .backtester import get_backtester
            bt = get_backtester()
            dashboard['tiers']['backtest'] = bt.get_aggregate_performance()
        except Exception:
            dashboard['tiers']['backtest'] = {'status': 'not_available'}

        # Tier 2: Optimizer
        try:
            from .parameter_optimizer import get_optimizer
            dashboard['tiers']['optimizer'] = {'status': 'available'}
        except Exception:
            dashboard['tiers']['optimizer'] = {'status': 'not_available'}

        # Tier 3: ML models
        try:
            from .ml_models import get_classifier
            c = get_classifier()
            dashboard['tiers']['ml_classifier'] = c.get_status()
        except Exception:
            dashboard['tiers']['ml_classifier'] = {'status': 'not_available'}

        # Tier 4: Deep models
        try:
            from .deep_models import get_deep_model
            d = get_deep_model()
            dashboard['tiers']['deep_model'] = d.get_metrics()
        except Exception:
            dashboard['tiers']['deep_model'] = {'status': 'not_available'}

        return dashboard


# Singleton
_engine_instance = None

def get_quant_engine() -> QuantEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = QuantEngine()
    return _engine_instance
