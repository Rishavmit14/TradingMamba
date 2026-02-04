"""
Pattern Quality Model - Trained ML model that scores detected Smart Money patterns.

Uses video-learned knowledge as features alongside OHLCV features to predict
whether a detected pattern will be profitable. This is where video training
genuinely feeds into model.fit().

Training data: for each historical bar where a pattern was detected,
extract 64 extended features + 6 pattern-type one-hot features = 70 total.
Label: did price move in the predicted direction within N bars?

Model: HistGradientBoostingClassifier with walk-forward validation.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Pattern types that can be scored (one-hot encoded)
SCOREABLE_PATTERNS = [
    'order_block', 'fvg', 'displacement',
    'optimal_trade_entry', 'breaker', 'liquidity',
]

N_PATTERN_FEATURES = len(SCOREABLE_PATTERNS)  # 6
FORWARD_BARS = 10  # Bars to look ahead for profit/loss
PROFIT_THRESHOLD = 0.005  # 0.5% move = profitable


@dataclass
class PatternQualityResult:
    """Result of training the pattern quality model."""
    accuracy: float
    f1_score: float
    n_samples: int
    n_profitable: int
    n_unprofitable: int
    n_folds: int
    feature_importance: Dict[str, float]
    ohlcv_importance: float  # Sum of importance for OHLCV features
    video_importance: float  # Sum of importance for video features
    pattern_importance: float  # Sum of importance for pattern type features
    model_version: str


class PatternQualityModel:
    """
    ML model that predicts whether a detected pattern will be profitable.

    Uses 70 features: 42 OHLCV + 22 video-derived + 6 pattern-type one-hot.
    Trained with walk-forward validation on historical pattern outcomes.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent.parent / "data")
        self.data_dir = Path(data_dir)
        self.model_path = self.data_dir / "ml_models" / "pattern_quality_model.joblib"

        self.model = None
        self.is_trained = False
        self.model_version = None
        self.n_features = None
        self.feature_importance: Dict[str, float] = {}
        self.training_result: Optional[PatternQualityResult] = None

        # Try to load saved model
        self._load_model()

    def _load_model(self):
        """Load saved model if exists."""
        if self.model_path.exists():
            try:
                saved = joblib.load(self.model_path)
                self.model = saved['model']
                self.model_version = saved.get('version', 'unknown')
                self.n_features = saved.get('n_features', 70)
                self.feature_importance = saved.get('feature_importance', {})
                self.is_trained = True
                logger.info(f"Pattern quality model loaded: v{self.model_version}")
            except Exception as e:
                logger.warning(f"Failed to load pattern quality model: {e}")

    def train(self, symbol: str = 'BTCUSDT', timeframe: str = 'D1',
              lookback_days: int = 730) -> Dict:
        """
        Train the pattern quality model on historical data.

        For each bar in history:
        1. Run Smart Money pattern detection
        2. For each detected pattern, extract 70 features
        3. Label: was the pattern profitable (price moved in direction within N bars)?
        4. Train with walk-forward validation

        Returns training metrics dict.
        """
        from .data_cache import get_data_cache
        from .feature_engineering import get_feature_engineer, NUM_EXTENDED_FEATURES

        logger.info(f"Training pattern quality model on {symbol} {timeframe}")

        # Get OHLCV data
        cache = get_data_cache()
        try:
            data = cache.get_ohlcv(symbol, timeframe, lookback_days)
        except Exception as e:
            return {'error': f'Failed to fetch data: {e}'}

        if len(data) < 200:
            return {'error': f'Only {len(data)} bars, need 200+'}

        # Run Smart Money analysis to detect patterns across all bars
        analysis = self._run_pattern_detection(data)
        if not analysis:
            return {'error': 'Pattern detection failed'}

        # Build training dataset
        X, y = self._build_dataset(data, analysis)

        if len(X) < 50:
            return {'error': f'Only {len(X)} pattern instances found, need 50+'}

        logger.info(f"Training on {len(X)} pattern instances ({sum(y)} profitable, {len(y) - sum(y)} not)")

        # Walk-forward validation
        result = self._walk_forward_train(X, y)

        if result is None:
            return {'error': 'Walk-forward training failed'}

        # Save model
        self._save_model()

        self.training_result = result

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'accuracy': round(result.accuracy, 4),
            'f1_score': round(result.f1_score, 4),
            'n_samples': result.n_samples,
            'n_profitable': result.n_profitable,
            'n_unprofitable': result.n_unprofitable,
            'n_folds': result.n_folds,
            'ohlcv_feature_importance': round(result.ohlcv_importance, 4),
            'video_feature_importance': round(result.video_importance, 4),
            'pattern_feature_importance': round(result.pattern_importance, 4),
            'top_features': dict(sorted(result.feature_importance.items(),
                                        key=lambda x: x[1], reverse=True)[:15]),
            'model_version': result.model_version,
        }

    def _run_pattern_detection(self, data) -> Dict:
        """Run Smart Money analysis on the full dataset."""
        try:
            from ..services.smart_money_analyzer import SmartMoneyAnalyzer
            analyzer = SmartMoneyAnalyzer()
            analysis = analyzer.analyze(data)
            return analysis
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {}

    def _build_dataset(self, data, analysis: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (X, y) dataset from detected patterns.

        For each detected pattern instance:
        - X: 64 extended features + 6 pattern-type one-hot = 70
        - y: 1 if profitable, 0 if not
        """
        from .feature_engineering import get_feature_engineer, NUM_EXTENDED_FEATURES

        engineer = get_feature_engineer()
        closes = data['close'].values.astype(float)
        n_bars = len(data)

        X_list = []
        y_list = []

        # Process each pattern type
        # analysis is a SmartMoneyAnalysisResult dataclass, use attribute access
        pattern_data = {
            'order_block': getattr(analysis, 'order_blocks', []) or [],
            'fvg': getattr(analysis, 'fair_value_gaps', []) or [],
            'displacement': getattr(analysis, 'displacements', []) or [],
            'optimal_trade_entry': getattr(analysis, 'ote_zones', []) or [],
            'breaker': getattr(analysis, 'breaker_blocks', []) or [],
            'liquidity': self._flatten_liquidity(getattr(analysis, 'liquidity_levels', {}) or {}),
        }

        for pattern_type, patterns in pattern_data.items():
            if not patterns:
                continue

            type_idx = SCOREABLE_PATTERNS.index(pattern_type) if pattern_type in SCOREABLE_PATTERNS else -1
            if type_idx == -1:
                continue

            for pattern in patterns:
                # Find the bar index for this pattern
                bar_idx = self._find_pattern_bar(pattern, data)
                if bar_idx is None or bar_idx < 50 or bar_idx >= n_bars - FORWARD_BARS:
                    continue

                # Extract 64 extended features
                ext_features = engineer.extract_extended_features(data, bar_idx, analysis)

                # Add 6 pattern-type one-hot features
                pattern_features = np.zeros(N_PATTERN_FEATURES)
                pattern_features[type_idx] = 1.0

                # Combine: 64 + 6 = 70
                full_features = np.concatenate([ext_features, pattern_features])

                # Label: was this pattern profitable?
                current_price = closes[bar_idx]
                future_price = closes[bar_idx + FORWARD_BARS]
                forward_return = (future_price - current_price) / current_price

                # Determine expected direction from pattern
                direction = self._get_pattern_direction(pattern, pattern_type, analysis)

                if direction == 'bullish':
                    profitable = 1 if forward_return > PROFIT_THRESHOLD else 0
                elif direction == 'bearish':
                    profitable = 1 if forward_return < -PROFIT_THRESHOLD else 0
                else:
                    # Neutral direction - skip
                    continue

                X_list.append(full_features)
                y_list.append(profitable)

        if not X_list:
            return np.zeros((0, NUM_EXTENDED_FEATURES + N_PATTERN_FEATURES)), np.zeros(0)

        return np.array(X_list), np.array(y_list)

    def _flatten_liquidity(self, liquidity_levels: Dict) -> list:
        """Flatten liquidity_levels dict into a flat list of items."""
        flat = []
        for key, vals in liquidity_levels.items():
            if isinstance(vals, list):
                flat.extend(vals)
        return flat

    def _find_pattern_bar(self, pattern, data) -> Optional[int]:
        """Find the bar index where a pattern was detected."""
        # Helper to get attribute from dict or dataclass
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Try index directly
        for key in ['bar_index', 'index', 'detection_index', 'start_index']:
            idx = _get(pattern, key)
            if idx is not None:
                return int(idx)

        # Try matching by price
        price = _get(pattern, 'price') or _get(pattern, 'high') or _get(pattern, 'level')
        if price is not None:
            closes = data['close'].values.astype(float)
            diffs = np.abs(closes - float(price))
            idx = int(np.argmin(diffs))
            return idx

        return None

    def _get_pattern_direction(self, pattern, pattern_type: str,
                                analysis) -> str:
        """Determine the expected direction of a pattern."""
        # Helper to get attribute from dict or dataclass
        def _get(obj, key, default=''):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Check pattern direction/type
        p_type = str(_get(pattern, 'type', ''))
        direction = str(_get(pattern, 'direction', ''))

        for s in [p_type, direction]:
            if 'bull' in s.lower():
                return 'bullish'
            if 'bear' in s.lower():
                return 'bearish'

        # Fall back to analysis bias
        analysis_bias = _get(analysis, 'bias', '')
        bias_str = str(analysis_bias).lower() if analysis_bias else ''
        if 'bull' in bias_str:
            return 'bullish'
        if 'bear' in bias_str:
            return 'bearish'

        return 'neutral'

    def _walk_forward_train(self, X: np.ndarray, y: np.ndarray) -> Optional[PatternQualityResult]:
        """Train with walk-forward validation."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import accuracy_score, f1_score

        n_samples = len(X)
        min_train = max(50, n_samples // 3)
        step = max(20, n_samples // 10)

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        accuracies = []
        f1_scores = []
        fold = 0

        train_end = min_train
        while train_end + step <= n_samples:
            test_end = min(train_end + step, n_samples)

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                train_end += step
                continue

            try:
                model = HistGradientBoostingClassifier(
                    max_iter=200,
                    max_depth=5,
                    learning_rate=0.05,
                    min_samples_leaf=5,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

                accuracies.append(acc)
                f1_scores.append(f1)
                fold += 1
            except Exception as e:
                logger.warning(f"Fold {fold} failed: {e}")

            train_end += step

        if not accuracies:
            return None

        # Final model on all data
        try:
            final_model = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=5,
                random_state=42,
            )
            final_model.fit(X, y)

            # Calibrate
            cal_model = CalibratedClassifierCV(final_model, cv=3, method='isotonic')
            cal_model.fit(X, y)

            self.model = cal_model
            self.is_trained = True
            self.n_features = X.shape[1]

        except Exception as e:
            logger.error(f"Final model training failed: {e}")
            return None

        # Feature importance
        from .feature_engineering import EXTENDED_FEATURE_NAMES
        feature_names = list(EXTENDED_FEATURE_NAMES) + [f'is_{p}' for p in SCOREABLE_PATTERNS]
        # HistGradientBoostingClassifier may not have feature_importances_ in older sklearn
        if hasattr(final_model, 'feature_importances_'):
            importances = final_model.feature_importances_
        else:
            try:
                from sklearn.inspection import permutation_importance
                perm = permutation_importance(final_model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
                importances = perm.importances_mean
            except Exception:
                importances = np.zeros(X.shape[1])

        self.feature_importance = {}
        for name, imp in zip(feature_names[:len(importances)], importances):
            if imp > 0.001:
                self.feature_importance[name] = round(float(imp), 4)

        # Compute importance breakdown
        ohlcv_imp = sum(importances[:42])
        video_imp = sum(importances[42:64]) if len(importances) > 42 else 0.0
        pattern_imp = sum(importances[64:]) if len(importances) > 64 else 0.0
        total_imp = ohlcv_imp + video_imp + pattern_imp
        if total_imp > 0:
            ohlcv_imp /= total_imp
            video_imp /= total_imp
            pattern_imp /= total_imp

        version = hashlib.md5(
            f"pqm_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        self.model_version = version

        return PatternQualityResult(
            accuracy=float(np.mean(accuracies)),
            f1_score=float(np.mean(f1_scores)),
            n_samples=n_samples,
            n_profitable=int(np.sum(y)),
            n_unprofitable=int(n_samples - np.sum(y)),
            n_folds=fold,
            feature_importance=self.feature_importance,
            ohlcv_importance=float(ohlcv_imp),
            video_importance=float(video_imp),
            pattern_importance=float(pattern_imp),
            model_version=version,
        )

    def score_pattern(self, pattern_type: str, extended_features: np.ndarray = None,
                       fallback_score: float = 0.5) -> float:
        """
        Score a detected pattern: probability it will be profitable.

        Args:
            pattern_type: e.g., 'order_block', 'fvg'
            extended_features: 64-dim feature vector (42 OHLCV + 22 video)
            fallback_score: returned if model not trained

        Returns:
            Probability 0.0-1.0 that the pattern will be profitable.
        """
        if not self.is_trained or self.model is None:
            return fallback_score

        if extended_features is None:
            return fallback_score

        # Build 70-dim vector: 64 extended + 6 pattern one-hot
        pattern_onehot = np.zeros(N_PATTERN_FEATURES)
        if pattern_type in SCOREABLE_PATTERNS:
            pattern_onehot[SCOREABLE_PATTERNS.index(pattern_type)] = 1.0

        full_features = np.concatenate([extended_features, pattern_onehot]).reshape(1, -1)
        full_features = np.nan_to_num(full_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle feature dimension mismatch
        if full_features.shape[1] != self.n_features:
            logger.warning(f"Feature mismatch: expected {self.n_features}, got {full_features.shape[1]}")
            return fallback_score

        try:
            proba = self.model.predict_proba(full_features)
            # Return probability of class 1 (profitable)
            return float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        except Exception as e:
            logger.warning(f"Pattern scoring failed: {e}")
            return fallback_score

    def _save_model(self):
        """Save model to disk."""
        if self.model is None:
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'version': self.model_version,
            'n_features': self.n_features,
            'feature_importance': self.feature_importance,
            'trained_at': datetime.utcnow().isoformat(),
        }, self.model_path)
        logger.info(f"Pattern quality model saved: {self.model_path}")

    def get_status(self) -> Dict:
        """Get model status."""
        return {
            'trained': self.is_trained,
            'model_version': self.model_version,
            'n_features': self.n_features,
            'feature_importance_breakdown': {
                'ohlcv': round(self.training_result.ohlcv_importance, 4) if self.training_result else None,
                'video': round(self.training_result.video_importance, 4) if self.training_result else None,
                'pattern_type': round(self.training_result.pattern_importance, 4) if self.training_result else None,
            },
            'top_features': dict(sorted(self.feature_importance.items(),
                                        key=lambda x: x[1], reverse=True)[:10]) if self.feature_importance else {},
        }


# Singleton
_pqm_instance = None


def get_pattern_quality_model() -> PatternQualityModel:
    """Get the singleton PatternQualityModel instance."""
    global _pqm_instance
    if _pqm_instance is None:
        _pqm_instance = PatternQualityModel()
    return _pqm_instance
