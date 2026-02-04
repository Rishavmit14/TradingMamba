"""
Deep Pattern Models - MLP Sequence Models (Tier 4)

Uses sklearn MLPClassifier for neural network pattern classification.
Key innovation: sequence features (10-bar lookback windows) instead of
single-bar features, capturing temporal patterns.

No PyTorch/TensorFlow dependency - pure scikit-learn.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from .feature_engineering import get_feature_engineer, NUM_FEATURES

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "ml_models" / "deep"
SEQUENCE_LENGTH = 10  # 10-bar lookback window
SEQUENCE_FEATURES = NUM_FEATURES * SEQUENCE_LENGTH  # 420 input features


class DeepPatternModel:
    """
    MLP-based pattern classifier using sequence features.

    Instead of looking at a single bar, looks at 10 consecutive bars
    (flattened into 420 features) to capture temporal patterns.
    """

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict] = {}
        self.metrics: Dict[str, Dict] = {}
        self._load_existing_models()

    def _load_existing_models(self):
        """Load previously trained deep models."""
        for path in MODEL_DIR.glob("*_deep.joblib"):
            try:
                parts = path.stem.replace("_deep", "").split("_")
                if len(parts) >= 3:
                    pattern_type = "_".join(parts[:-2])
                    data = joblib.load(path)
                    self.models[pattern_type] = data
                    logger.info(f"Loaded deep model: {pattern_type}")
            except Exception as e:
                logger.warning(f"Failed to load deep model {path}: {e}")

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Convert single-bar features into sequence features.

        Input: (n_bars, 42)
        Output: (n_bars - SEQUENCE_LENGTH + 1, 420)
        """
        n = X.shape[0]
        if n < SEQUENCE_LENGTH:
            return np.zeros((0, SEQUENCE_FEATURES))

        n_sequences = n - SEQUENCE_LENGTH + 1
        X_seq = np.zeros((n_sequences, SEQUENCE_FEATURES))

        for i in range(n_sequences):
            # Flatten 10 consecutive bars into one feature vector
            X_seq[i] = X[i:i + SEQUENCE_LENGTH].flatten()

        return X_seq

    def train(
        self,
        symbol: str,
        timeframe: str = 'D1',
        lookback_days: int = 730,
    ) -> Dict:
        """
        Train MLP deep models for all pattern types.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            lookback_days: Training data period

        Returns:
            Dict with training results
        """
        logger.info(f"Training deep models: {symbol} {timeframe}")

        engineer = get_feature_engineer()
        X_raw, y_dict = engineer.build_training_dataset(symbol, timeframe, lookback_days)

        if X_raw.shape[0] < SEQUENCE_LENGTH + 50:
            return {"error": f"Insufficient data: {X_raw.shape[0]} samples"}

        # Build sequences
        X_seq = self._build_sequences(X_raw)
        # Adjust labels to match sequence start positions
        offset = SEQUENCE_LENGTH - 1

        results = {}
        for pattern_type, y_raw in y_dict.items():
            y_aligned = y_raw[offset:]
            if len(y_aligned) != X_seq.shape[0]:
                y_aligned = y_aligned[:X_seq.shape[0]]

            # Filter to only bars with signals
            mask = y_aligned >= 0
            if np.sum(mask) < 50:
                results[pattern_type] = {
                    "status": "skipped",
                    "reason": "insufficient_samples",
                    "samples": int(np.sum(mask))
                }
                continue

            X_pattern = X_seq[mask]
            y_pattern = y_aligned[mask]

            try:
                metrics = self._train_pattern(pattern_type, X_pattern, y_pattern, symbol, timeframe)
                results[pattern_type] = metrics
            except Exception as e:
                logger.error(f"Failed to train deep {pattern_type}: {e}")
                results[pattern_type] = {"status": "error", "error": str(e)}

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "sequence_length": SEQUENCE_LENGTH,
            "total_sequence_features": SEQUENCE_FEATURES,
            "total_sequences": X_seq.shape[0],
            "pattern_results": results,
            "trained_at": datetime.utcnow().isoformat(),
        }

    def _train_pattern(
        self,
        pattern_type: str,
        X: np.ndarray,
        y: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> Dict:
        """Train MLP for a single pattern type."""
        logger.info(f"Training deep {pattern_type}: {X.shape[0]} sequences")

        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        all_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42,
                batch_size=min(32, len(X_train)),
            )

            mlp.fit(X_train_s, y_train)
            y_prob = mlp.predict_proba(X_test_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            fold_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
            }
            all_metrics.append(fold_metrics)

        if not all_metrics:
            return {"status": "error", "reason": "no_valid_folds"}

        # Train final model on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            batch_size=min(32, len(X)),
        )
        mlp.fit(X_scaled, y)

        # Save
        model_data = {
            'mlp': mlp,
            'scaler': scaler,
            'symbol': symbol,
            'timeframe': timeframe,
            'trained_at': datetime.utcnow().isoformat(),
            'n_samples': len(y),
        }
        self.models[pattern_type] = model_data

        model_path = MODEL_DIR / f"{pattern_type}_{symbol}_{timeframe}_deep.joblib"
        joblib.dump(model_data, model_path)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = round(np.mean([m[key] for m in all_metrics]), 4)

        avg_metrics['status'] = 'trained'
        avg_metrics['samples'] = len(y)
        avg_metrics['model'] = 'MLP(128,64,32)'
        avg_metrics['sequence_length'] = SEQUENCE_LENGTH

        self.metrics[pattern_type] = avg_metrics
        return avg_metrics

    def predict_sequence(self, pattern_type: str, feature_sequence: np.ndarray) -> float:
        """
        Predict using sequence features.

        Args:
            pattern_type: Pattern type
            feature_sequence: Either (10, 42) or pre-flattened (420,)

        Returns:
            Probability (0-1)
        """
        if pattern_type not in self.models:
            return 0.5

        model_data = self.models[pattern_type]

        if feature_sequence.ndim == 2:
            # (10, 42) → (420,)
            feature_sequence = feature_sequence.flatten()

        if feature_sequence.ndim == 1:
            feature_sequence = feature_sequence.reshape(1, -1)

        X_scaled = model_data['scaler'].transform(feature_sequence)
        prob = model_data['mlp'].predict_proba(X_scaled)[:, 1]

        return float(prob[0])

    def get_metrics(self) -> Dict:
        """Get training metrics for all deep models."""
        return {
            "models_trained": len(self.models),
            "pattern_types": list(self.models.keys()),
            "metrics": self.metrics,
        }

    def has_model(self, pattern_type: str) -> bool:
        return pattern_type in self.models


class EnsemblePredictor:
    """
    Combines Tier 3 (GBM/RF/LR) and Tier 4 (MLP) predictions.

    Weighted average with configurable blending.
    """

    def __init__(self):
        self._classifier = None
        self._deep_model = None

    def _load_models(self):
        if self._classifier is None:
            try:
                from .ml_models import get_classifier
                self._classifier = get_classifier()
            except Exception:
                pass
        if self._deep_model is None:
            try:
                self._deep_model = get_deep_model()
            except Exception:
                pass

    def predict(
        self,
        pattern_type: str,
        features: np.ndarray,
        feature_sequence: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Get ensemble prediction combining Tier 3 and Tier 4.

        Args:
            pattern_type: Pattern type
            features: Single-bar features (42,) for Tier 3
            feature_sequence: Sequence features (10, 42) for Tier 4

        Returns:
            Dict with individual and ensemble probabilities
        """
        self._load_models()

        result = {
            'pattern_type': pattern_type,
            'tier3_prob': 0.5,
            'tier4_prob': 0.5,
            'ensemble_prob': 0.5,
            'signal': 'neutral',
        }

        # Tier 3 prediction
        if self._classifier and self._classifier.has_model(pattern_type):
            result['tier3_prob'] = self._classifier.predict(pattern_type, features)

        # Tier 4 prediction
        if self._deep_model and self._deep_model.has_model(pattern_type) and feature_sequence is not None:
            result['tier4_prob'] = self._deep_model.predict_sequence(pattern_type, feature_sequence)

        # Ensemble: weight based on which models are available
        has_t3 = self._classifier and self._classifier.has_model(pattern_type)
        has_t4 = self._deep_model and self._deep_model.has_model(pattern_type) and feature_sequence is not None

        if has_t3 and has_t4:
            result['ensemble_prob'] = result['tier3_prob'] * 0.6 + result['tier4_prob'] * 0.4
        elif has_t3:
            result['ensemble_prob'] = result['tier3_prob']
        elif has_t4:
            result['ensemble_prob'] = result['tier4_prob']

        # Signal classification
        if result['ensemble_prob'] >= 0.65:
            result['signal'] = 'strong_bullish'
        elif result['ensemble_prob'] >= 0.55:
            result['signal'] = 'bullish'
        elif result['ensemble_prob'] <= 0.35:
            result['signal'] = 'strong_bearish'
        elif result['ensemble_prob'] <= 0.45:
            result['signal'] = 'bearish'
        else:
            result['signal'] = 'neutral'

        return result


# Singletons
_deep_model_instance = None
_ensemble_instance = None

def get_deep_model() -> DeepPatternModel:
    global _deep_model_instance
    if _deep_model_instance is None:
        _deep_model_instance = DeepPatternModel()
    return _deep_model_instance

def get_ensemble_predictor() -> EnsemblePredictor:
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = EnsemblePredictor()
    return _ensemble_instance
