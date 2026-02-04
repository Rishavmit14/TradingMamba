"""
Genuine Predictive ML System - Forward Price Direction Prediction

Trains on OHLCV data to predict forward returns using walk-forward validation.
Tracks live predictions vs actual outcomes. Auto-retrains when accuracy degrades.

NO look-ahead bias. NO circular training. Real forward-looking labels.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import warnings
import hashlib

warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        ExtraTreesClassifier,
    )
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "ml_models" / "predictors"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Prediction thresholds
BULLISH_THRESHOLD = 0.005   # +0.5% forward return = bullish
BEARISH_THRESHOLD = -0.005  # -0.5% forward return = bearish

# Default symbols and horizons
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'XAUUSD']
HORIZONS = [5, 10, 20]  # bars ahead

# Minimum data requirements
MIN_TRAIN_BARS = 500
MIN_WALK_FORWARD_BARS = 100
WALK_FORWARD_STEP = 50

# Retrain triggers
RETRAIN_ACCURACY_THRESHOLD = 0.40  # retrain if rolling accuracy < 40%
RETRAIN_MAX_AGE_DAYS = 30
RETRAIN_MIN_PREDICTIONS = 30


class Direction(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class Prediction:
    symbol: str
    timeframe: str
    horizon: int
    direction: str
    confidence: float
    prob_bullish: float
    prob_neutral: float
    prob_bearish: float
    price_at_prediction: float
    feature_snapshot: Dict
    model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'horizon': self.horizon,
            'direction': self.direction,
            'confidence': round(self.confidence, 4),
            'prob_bullish': round(self.prob_bullish, 4),
            'prob_neutral': round(self.prob_neutral, 4),
            'prob_bearish': round(self.prob_bearish, 4),
            'price_at_prediction': self.price_at_prediction,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class TrainResult:
    symbol: str
    timeframe: str
    horizon: int
    accuracy: float
    f1_macro: float
    directional_accuracy: float
    n_folds: int
    n_train_samples: int
    class_distribution: Dict
    feature_importance: Dict
    model_version: str
    trained_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'horizon': self.horizon,
            'accuracy': round(self.accuracy, 4),
            'f1_macro': round(self.f1_macro, 4),
            'directional_accuracy': round(self.directional_accuracy, 4),
            'n_folds': self.n_folds,
            'n_train_samples': self.n_train_samples,
            'class_distribution': self.class_distribution,
            'top_features': dict(sorted(
                self.feature_importance.items(),
                key=lambda x: abs(x[1]), reverse=True
            )[:10]),
            'model_version': self.model_version,
            'trained_at': self.trained_at.isoformat(),
        }


def create_forward_labels(
    closes: np.ndarray,
    horizon: int,
    bull_threshold: float = BULLISH_THRESHOLD,
    bear_threshold: float = BEARISH_THRESHOLD,
) -> np.ndarray:
    """
    Create forward-looking labels from close prices.

    For each bar i, compute: (close[i+horizon] - close[i]) / close[i]
    Classify as:
      0 = BEARISH  (return < bear_threshold)
      1 = NEUTRAL  (bear_threshold <= return <= bull_threshold)
      2 = BULLISH  (return > bull_threshold)
     -1 = UNRESOLVABLE (not enough forward data)

    Returns:
        np.ndarray of shape (n_bars,) with values in {-1, 0, 1, 2}
    """
    n = len(closes)
    labels = np.full(n, -1, dtype=np.int32)

    for i in range(n - horizon):
        fwd_return = (closes[i + horizon] - closes[i]) / closes[i]
        if fwd_return > bull_threshold:
            labels[i] = 2  # BULLISH
        elif fwd_return < bear_threshold:
            labels[i] = 0  # BEARISH
        else:
            labels[i] = 1  # NEUTRAL

    return labels


def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    min_train: int = MIN_TRAIN_BARS,
    test_size: int = MIN_WALK_FORWARD_BARS,
    step: int = WALK_FORWARD_STEP,
) -> List[Dict]:
    """
    Expanding-window walk-forward validation.

    Train on [0..T], test on [T+1..T+test_size], step forward.
    No data leakage: test data is always in the future relative to training.

    Returns:
        List of fold results with accuracy, f1, directional accuracy
    """
    n = len(X)
    folds = []

    t = min_train
    while t + test_size <= n:
        X_train, y_train = X[:t], y[:t]
        X_test, y_test = X[t:t + test_size], y[t:t + test_size]

        # Filter out unresolvable labels (-1) from train and test
        train_mask = y_train >= 0
        test_mask = y_test >= 0

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            t += step
            continue

        X_tr, y_tr = X_train[train_mask], y_train[train_mask]
        X_te, y_te = X_test[test_mask], y_test[test_mask]

        # Train a quick model for this fold
        model = HistGradientBoostingClassifier(
            max_iter=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        # Metrics
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)

        # Directional accuracy: exclude neutral predictions and neutral actuals
        dir_mask = (y_te != 1) & (y_pred != 1)
        dir_acc = accuracy_score(y_te[dir_mask], y_pred[dir_mask]) if dir_mask.sum() > 5 else 0.0

        folds.append({
            'train_size': int(train_mask.sum()),
            'test_size': int(test_mask.sum()),
            'accuracy': float(acc),
            'f1_macro': float(f1),
            'directional_accuracy': float(dir_acc),
        })

        t += step

    return folds


class PredictionModel:
    """
    Ensemble prediction model for a single (symbol, timeframe, horizon).

    Combines:
    - HistGradientBoostingClassifier (45% weight) - fast, handles missing values
    - ExtraTreesClassifier (35% weight) - decorrelates with HGB
    - RidgeClassifierCV (20% weight) - linear baseline

    All wrapped in CalibratedClassifierCV for reliable probabilities.
    """

    def __init__(self, symbol: str, timeframe: str, horizon: int):
        self.symbol = symbol
        self.timeframe = timeframe
        self.horizon = horizon
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.model_version = ""
        self.feature_importance = {}

    def _model_path(self) -> Path:
        return MODEL_DIR / f"predictor_{self.symbol}_{self.timeframe}_h{self.horizon}.joblib"

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the ensemble model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (values in {0, 1, 2})

        Returns:
            Training metrics dict
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not installed', 'trained': False}

        # Filter valid labels
        mask = y >= 0
        X_valid, y_valid = X[mask], y[mask]

        if len(X_valid) < 100:
            return {'error': f'Only {len(X_valid)} valid samples, need 100+', 'trained': False}

        # Scale features
        X_scaled = self.scaler.fit_transform(X_valid)

        # Build ensemble components
        hgb = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, random_state=42
        )
        et = ExtraTreesClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        ridge = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0])

        # Train each component
        hgb.fit(X_scaled, y_valid)
        et.fit(X_scaled, y_valid)
        ridge.fit(X_scaled, y_valid)

        # Calibrate HGB for reliable probabilities (HGB has predict_proba)
        try:
            cal_hgb = CalibratedClassifierCV(hgb, cv=3, method='isotonic')
            cal_hgb.fit(X_scaled, y_valid)
        except Exception:
            cal_hgb = hgb  # fallback if calibration fails

        # Store models
        self.model = {
            'hgb': cal_hgb,
            'et': et,
            'ridge': ridge,
            'classes': np.unique(y_valid),
        }

        # Feature importance from ExtraTrees (most reliable source)
        if hasattr(et, 'feature_importances_'):
            from .feature_engineering import EXTENDED_FEATURE_NAMES
            importances = et.feature_importances_
            self.feature_importance = {
                name: float(imp) for name, imp in zip(EXTENDED_FEATURE_NAMES[:len(importances)], importances)
            }

        # Generate model version hash
        self.model_version = hashlib.md5(
            f"{self.symbol}_{self.timeframe}_h{self.horizon}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]

        self.is_trained = True

        # Save model
        self._save()

        # Class distribution
        unique, counts = np.unique(y_valid, return_counts=True)
        label_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
        class_dist = {label_map.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)}

        return {
            'trained': True,
            'n_samples': len(X_valid),
            'class_distribution': class_dist,
            'model_version': self.model_version,
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class and probabilities using ensemble voting.

        Returns:
            (predictions, probabilities) where probabilities shape is (n, 3)
            Classes: 0=bearish, 1=neutral, 2=bullish
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(X)
        classes = self.model['classes']
        n_classes = len(classes)
        n_samples = X.shape[0]

        # Weights
        w_hgb, w_et, w_ridge = 0.45, 0.35, 0.20

        # HGB probabilities
        hgb = self.model['hgb']
        if hasattr(hgb, 'predict_proba'):
            proba_hgb = hgb.predict_proba(X_scaled)
        else:
            pred_hgb = hgb.predict(X_scaled)
            proba_hgb = np.zeros((n_samples, n_classes))
            for i, p in enumerate(pred_hgb):
                idx = np.where(classes == p)[0]
                if len(idx) > 0:
                    proba_hgb[i, idx[0]] = 1.0

        # ET probabilities
        et = self.model['et']
        proba_et = et.predict_proba(X_scaled)

        # Ridge: no predict_proba, use decision function
        ridge = self.model['ridge']
        try:
            decision = ridge.decision_function(X_scaled)
            if decision.ndim == 1:
                # Binary case - expand to 2 columns
                proba_ridge = np.column_stack([1 - _sigmoid(decision), _sigmoid(decision)])
                if n_classes == 3:
                    proba_ridge = np.column_stack([
                        proba_ridge[:, 0] * 0.5,
                        np.full(n_samples, 0.33),
                        proba_ridge[:, 1] * 0.5,
                    ])
            else:
                # Multi-class: apply softmax to decision values
                proba_ridge = _softmax(decision)
        except Exception:
            # Fallback: uniform
            proba_ridge = np.full((n_samples, n_classes), 1.0 / n_classes)

        # Ensure all probability arrays have same shape
        for arr_name, arr in [('hgb', proba_hgb), ('et', proba_et), ('ridge', proba_ridge)]:
            if arr.shape[1] != n_classes:
                # Pad or trim to match
                padded = np.full((n_samples, n_classes), 1.0 / n_classes)
                cols = min(arr.shape[1], n_classes)
                padded[:, :cols] = arr[:, :cols]
                if arr_name == 'hgb':
                    proba_hgb = padded
                elif arr_name == 'et':
                    proba_et = padded
                else:
                    proba_ridge = padded

        # Weighted ensemble
        ensemble_proba = w_hgb * proba_hgb + w_et * proba_et + w_ridge * proba_ridge

        # Normalize
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        ensemble_proba = ensemble_proba / row_sums

        # Predictions from highest probability
        predictions = classes[np.argmax(ensemble_proba, axis=1)]

        return predictions, ensemble_proba

    def _save(self):
        """Save model to disk."""
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_version': self.model_version,
                'feature_importance': self.feature_importance,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'horizon': self.horizon,
                'trained_at': datetime.utcnow().isoformat(),
            }, self._model_path())
            logger.info(f"Saved model: {self._model_path().name}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load(self) -> bool:
        """Load model from disk."""
        path = self._model_path()
        if not path.exists():
            return False
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_version = data['model_version']
            self.feature_importance = data.get('feature_importance', {})
            self.is_trained = True
            logger.info(f"Loaded model: {path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {path.name}: {e}")
            return False


class PricePredictor:
    """
    Main orchestrator for price predictions.

    Manages training, prediction, resolution, and performance tracking
    for all symbols and horizons.
    """

    def __init__(self):
        self.models: Dict[str, PredictionModel] = {}
        self._load_existing_models()

    def _model_key(self, symbol: str, timeframe: str, horizon: int) -> str:
        return f"{symbol}_{timeframe}_h{horizon}"

    def _load_existing_models(self):
        """Load all saved models from disk."""
        for path in MODEL_DIR.glob("predictor_*.joblib"):
            try:
                data = joblib.load(path)
                symbol = data['symbol']
                tf = data['timeframe']
                horizon = data['horizon']
                key = self._model_key(symbol, tf, horizon)
                model = PredictionModel(symbol, tf, horizon)
                if model.load():
                    self.models[key] = model
            except Exception as e:
                logger.warning(f"Could not load {path.name}: {e}")

    def train_symbol(self, symbol: str, timeframe: str = 'D1') -> Dict:
        """
        Train prediction models for a symbol across all horizons.

        1. Fetch OHLCV data
        2. Extract 42 features per bar
        3. Create forward labels per horizon
        4. Run walk-forward validation
        5. Train final model on all data
        6. Save models and metrics

        Returns:
            Dict with training results per horizon
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not installed'}

        from .data_cache import get_data_cache
        from .feature_engineering import get_feature_engineer

        cache = get_data_cache()
        engineer = get_feature_engineer()

        # Fetch OHLCV data
        try:
            data = cache.get_ohlcv(symbol, timeframe, lookback_days=730)
        except Exception as e:
            return {'error': f'Failed to fetch data: {e}'}

        if len(data) < MIN_TRAIN_BARS:
            return {'error': f'Only {len(data)} bars available, need {MIN_TRAIN_BARS}+'}

        logger.info(f"Training predictor for {symbol} {timeframe}: {len(data)} bars")

        # Run Smart Money analysis for video-derived features
        analysis = None
        try:
            from ..services.smart_money_analyzer import SmartMoneyAnalyzer
            analyzer = SmartMoneyAnalyzer()
            analysis = analyzer.analyze(data, timeframe=timeframe)
            logger.info("Smart Money analysis available for extended features")
        except Exception as e:
            logger.info(f"Smart Money analysis unavailable ({e}), using OHLCV features only")

        # Extract extended features (42 OHLCV + 22 video = 64) for all bars
        X = engineer.extract_extended_features_batch(data, start=50, analysis=analysis)
        closes = data['close'].values[50:]  # Align with feature extraction

        if len(X) != len(closes):
            min_len = min(len(X), len(closes))
            X = X[:min_len]
            closes = closes[:min_len]

        # Replace NaN/inf with 0 to prevent division errors
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        results = {}

        for horizon in HORIZONS:
            # Create forward labels
            labels = create_forward_labels(closes, horizon)

            # Walk-forward validation
            folds = walk_forward_validate(X, labels)

            if not folds:
                results[f'h{horizon}'] = {
                    'error': 'Not enough data for walk-forward validation',
                    'trained': False,
                }
                continue

            avg_acc = np.mean([f['accuracy'] for f in folds])
            avg_f1 = np.mean([f['f1_macro'] for f in folds])
            avg_dir = np.mean([f['directional_accuracy'] for f in folds])

            # Train final model on all data
            key = self._model_key(symbol, timeframe, horizon)
            model = PredictionModel(symbol, timeframe, horizon)
            train_result = model.train(X, labels)

            if train_result.get('trained'):
                self.models[key] = model

                # Build result
                unique, counts = np.unique(labels[labels >= 0], return_counts=True)
                label_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
                class_dist = {label_map.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)}

                result = TrainResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon=horizon,
                    accuracy=avg_acc,
                    f1_macro=avg_f1,
                    directional_accuracy=avg_dir,
                    n_folds=len(folds),
                    n_train_samples=int((labels >= 0).sum()),
                    class_distribution=class_dist,
                    feature_importance=model.feature_importance,
                    model_version=model.model_version,
                )
                results[f'h{horizon}'] = result.to_dict()

                # Save metrics to DB
                self._save_metrics(result)
            else:
                results[f'h{horizon}'] = train_result

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_bars': len(data),
            'feature_bars': len(X),
            'horizons': results,
        }

    def predict_symbol(self, symbol: str, timeframe: str = 'D1') -> Dict:
        """
        Generate forward predictions for a symbol across all horizons.

        Returns:
            Dict with predictions per horizon
        """
        from .data_cache import get_data_cache
        from .feature_engineering import get_feature_engineer

        cache = get_data_cache()
        engineer = get_feature_engineer()

        # Fetch latest OHLCV
        try:
            data = cache.get_ohlcv(symbol, timeframe, lookback_days=60)
        except Exception as e:
            return {'error': f'Failed to fetch data: {e}'}

        if len(data) < 51:
            return {'error': f'Only {len(data)} bars, need 51+'}

        # Run Smart Money analysis for video-derived features
        analysis = None
        try:
            from ..services.smart_money_analyzer import SmartMoneyAnalyzer
            analyzer = SmartMoneyAnalyzer()
            analysis = analyzer.analyze(data, timeframe=timeframe)
        except Exception:
            pass  # Fall back to OHLCV-only features

        # Extract extended features for the latest bar (42 OHLCV + 22 video = 64)
        X_latest = engineer.extract_extended_features(data, len(data) - 1, analysis).reshape(1, -1)
        X_latest = np.nan_to_num(X_latest, nan=0.0, posinf=0.0, neginf=0.0)
        current_price = float(data['close'].iloc[-1])

        predictions = {}

        for horizon in HORIZONS:
            key = self._model_key(symbol, timeframe, horizon)
            model = self.models.get(key)

            if model is None or not model.is_trained:
                predictions[f'h{horizon}'] = {'error': 'Model not trained', 'trained': False}
                continue

            try:
                pred_class, proba = model.predict(X_latest)

                label_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
                direction = label_map.get(int(pred_class[0]), 'neutral')
                confidence = float(np.max(proba[0]))

                # Map probabilities to class names
                classes = model.model.get('classes', np.array([0, 1, 2]))
                prob_dict = {0: 0.0, 1: 0.0, 2: 0.0}
                for i, cls in enumerate(classes):
                    if i < proba.shape[1]:
                        prob_dict[int(cls)] = float(proba[0, i])

                # Top feature influences
                top_features = dict(sorted(
                    model.feature_importance.items(),
                    key=lambda x: abs(x[1]), reverse=True
                )[:5])

                pred = Prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon=horizon,
                    direction=direction,
                    confidence=confidence,
                    prob_bullish=prob_dict.get(2, 0.0),
                    prob_neutral=prob_dict.get(1, 0.0),
                    prob_bearish=prob_dict.get(0, 0.0),
                    price_at_prediction=current_price,
                    feature_snapshot=top_features,
                    model_version=model.model_version,
                )

                predictions[f'h{horizon}'] = pred.to_dict()

                # Save prediction to DB
                self._save_prediction(pred)

            except Exception as e:
                logger.error(f"Prediction failed for {key}: {e}")
                predictions[f'h{horizon}'] = {'error': str(e)}

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'predictions': predictions,
        }

    def resolve_predictions(self) -> Dict:
        """
        Resolve pending predictions by checking actual prices.

        For each unresolved prediction, check if enough bars have passed
        and record actual direction.

        Returns:
            Summary of resolved predictions
        """
        from ..database import db

        try:
            pending = db.get_pending_predictions()
        except Exception as e:
            return {'error': f'DB error: {e}', 'resolved': 0}

        if not pending:
            return {'resolved': 0, 'message': 'No pending predictions'}

        from .data_cache import get_data_cache
        cache = get_data_cache()

        resolved_count = 0
        results = []

        for pred in pending:
            symbol = pred['symbol']
            timeframe = pred['timeframe']
            horizon = pred['horizon']
            price_at = pred['price_at_prediction']

            try:
                data = cache.get_ohlcv(symbol, timeframe, lookback_days=60)
                if len(data) < 2:
                    continue

                current_price = float(data['close'].iloc[-1])
                predicted_at = datetime.fromisoformat(pred['predicted_at'])
                now = datetime.utcnow()

                # Estimate bars passed based on timeframe
                tf_hours = {'M1': 1/60, 'M5': 5/60, 'M15': 0.25, 'M30': 0.5,
                            'H1': 1, 'H4': 4, 'D1': 24, 'W1': 168}
                hours_per_bar = tf_hours.get(timeframe, 24)
                hours_passed = (now - predicted_at).total_seconds() / 3600
                bars_passed = hours_passed / hours_per_bar

                if bars_passed < horizon:
                    continue  # Not enough time has passed

                # Calculate actual return
                actual_return = (current_price - price_at) / price_at

                if actual_return > BULLISH_THRESHOLD:
                    actual_direction = 'bullish'
                elif actual_return < BEARISH_THRESHOLD:
                    actual_direction = 'bearish'
                else:
                    actual_direction = 'neutral'

                was_correct = 1 if pred['direction'] == actual_direction else 0

                db.resolve_prediction(
                    pred['id'],
                    actual_return=actual_return,
                    actual_direction=actual_direction,
                    was_correct=was_correct,
                )

                resolved_count += 1
                results.append({
                    'symbol': symbol,
                    'horizon': horizon,
                    'predicted': pred['direction'],
                    'actual': actual_direction,
                    'correct': bool(was_correct),
                    'return': round(actual_return * 100, 3),
                })

            except Exception as e:
                logger.warning(f"Failed to resolve prediction {pred['id']}: {e}")

        return {
            'resolved': resolved_count,
            'results': results,
        }

    def get_performance(self, symbol: str = None, lookback_days: int = 90) -> Dict:
        """
        Get live prediction performance from resolved predictions.

        Returns:
            Performance metrics per symbol and horizon
        """
        from ..database import db

        try:
            perf = db.get_prediction_performance(symbol, lookback_days)
        except Exception as e:
            return {'error': str(e)}

        return perf

    def get_status(self) -> Dict:
        """
        Get status of all trained models and pending predictions.
        """
        from ..database import db

        models_info = []
        for key, model in self.models.items():
            models_info.append({
                'key': key,
                'symbol': model.symbol,
                'timeframe': model.timeframe,
                'horizon': model.horizon,
                'model_version': model.model_version,
                'is_trained': model.is_trained,
            })

        try:
            pending = db.get_pending_predictions()
            pending_count = len(pending) if pending else 0
        except Exception:
            pending_count = 0

        try:
            recent = db.get_recent_predictions(limit=10)
        except Exception:
            recent = []

        return {
            'models': models_info,
            'total_models': len(self.models),
            'pending_predictions': pending_count,
            'recent_predictions': recent,
        }

    def check_retrain_needed(self, symbol: str, timeframe: str = 'D1') -> Dict:
        """
        Check if any models for this symbol need retraining.

        Triggers:
        - Rolling accuracy < 40% over last 30 predictions
        - Model older than 30 days
        """
        needs_retrain = {}

        for horizon in HORIZONS:
            key = self._model_key(symbol, timeframe, horizon)
            model = self.models.get(key)

            if model is None:
                needs_retrain[f'h{horizon}'] = {'retrain': True, 'reason': 'no_model'}
                continue

            # Check model age
            model_path = model._model_path()
            if model_path.exists():
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                age_days = (datetime.now() - mtime).days
                if age_days > RETRAIN_MAX_AGE_DAYS:
                    needs_retrain[f'h{horizon}'] = {
                        'retrain': True,
                        'reason': f'model_age_{age_days}_days',
                    }
                    continue

            # Check rolling accuracy
            try:
                from ..database import db
                perf = db.get_prediction_performance(symbol, lookback_days=30)
                if perf and 'by_horizon' in perf:
                    h_perf = perf['by_horizon'].get(str(horizon), {})
                    total = h_perf.get('total', 0)
                    acc = h_perf.get('accuracy', 0.5)

                    if total >= RETRAIN_MIN_PREDICTIONS and acc < RETRAIN_ACCURACY_THRESHOLD:
                        needs_retrain[f'h{horizon}'] = {
                            'retrain': True,
                            'reason': f'low_accuracy_{acc:.1%}',
                        }
                        continue
            except Exception:
                pass

            needs_retrain[f'h{horizon}'] = {'retrain': False}

        return needs_retrain

    def _save_prediction(self, pred: Prediction):
        """Save a prediction to the database."""
        try:
            from ..database import db
            db.save_prediction({
                'symbol': pred.symbol,
                'timeframe': pred.timeframe,
                'horizon': pred.horizon,
                'direction': pred.direction,
                'confidence': pred.confidence,
                'prob_bullish': pred.prob_bullish,
                'prob_neutral': pred.prob_neutral,
                'prob_bearish': pred.prob_bearish,
                'price_at_prediction': pred.price_at_prediction,
                'predicted_at': pred.timestamp.isoformat(),
                'feature_snapshot': json.dumps(pred.feature_snapshot),
                'model_version': pred.model_version,
            })
        except Exception as e:
            logger.warning(f"Failed to save prediction to DB: {e}")

    def _save_metrics(self, result: TrainResult):
        """Save training metrics to the database."""
        try:
            from ..database import db
            db.save_predictor_metrics({
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'horizon': result.horizon,
                'accuracy': result.accuracy,
                'f1_macro': result.f1_macro,
                'directional_accuracy': result.directional_accuracy,
                'n_walk_forward_folds': result.n_folds,
                'n_train_samples': result.n_train_samples,
                'class_distribution': json.dumps(result.class_distribution),
                'feature_importance': json.dumps(
                    dict(sorted(result.feature_importance.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:15])
                ),
            })
        except Exception as e:
            logger.warning(f"Failed to save metrics to DB: {e}")


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


# Singleton
_predictor_instance = None

def get_price_predictor() -> PricePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PricePredictor()
    return _predictor_instance
