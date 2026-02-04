"""
ML Pattern Classifier - Ensemble Models (Tier 3)

Trains per-pattern-type classifiers using scikit-learn ensemble methods.
Uses features from FeatureEngineer and labels from Backtester.

Ensemble: GradientBoosting + RandomForest + LogisticRegression
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from .feature_engineering import get_feature_engineer, NUM_FEATURES

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "ml_models" / "classifiers"


class MLPatternClassifier:
    """
    Ensemble classifier for Smart Money pattern prediction.

    One model per pattern type. Each model predicts probability
    that a pattern at the current bar will be profitable.
    """

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict] = {}  # pattern_type -> {'gbm': ..., 'rf': ..., 'lr': ..., 'scaler': ...}
        self.metrics: Dict[str, Dict] = {}
        self._load_existing_models()

    def _load_existing_models(self):
        """Load previously trained models from disk."""
        for path in MODEL_DIR.glob("*_ensemble.joblib"):
            try:
                parts = path.stem.replace("_ensemble", "").split("_")
                # Last two parts are symbol and timeframe
                # Everything before is the pattern type
                if len(parts) >= 3:
                    pattern_type = "_".join(parts[:-2])
                    data = joblib.load(path)
                    self.models[pattern_type] = data
                    logger.info(f"Loaded model: {pattern_type}")
            except Exception as e:
                logger.warning(f"Failed to load model {path}: {e}")

    def train(
        self,
        symbol: str,
        timeframe: str = 'D1',
        lookback_days: int = 730,
    ) -> Dict:
        """
        Train ensemble classifiers for all pattern types.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            lookback_days: Training data lookback

        Returns:
            Dict with training metrics per pattern type
        """
        logger.info(f"Training classifiers: {symbol} {timeframe}")

        engineer = get_feature_engineer()
        X, y_dict = engineer.build_training_dataset(symbol, timeframe, lookback_days)

        if X.shape[0] < 100:
            return {"error": f"Insufficient training data: {X.shape[0]} samples"}

        results = {}

        for pattern_type, y_raw in y_dict.items():
            # Filter to only bars with signals (y != -1)
            mask = y_raw >= 0
            if np.sum(mask) < 30:
                logger.info(f"Skipping {pattern_type}: only {np.sum(mask)} samples")
                results[pattern_type] = {"status": "skipped", "reason": "insufficient_samples", "samples": int(np.sum(mask))}
                continue

            X_pattern = X[mask]
            y_pattern = y_raw[mask]

            try:
                metrics = self._train_pattern(pattern_type, X_pattern, y_pattern, symbol, timeframe)
                results[pattern_type] = metrics
                self._save_metrics(symbol, timeframe, pattern_type, metrics)
            except Exception as e:
                logger.error(f"Failed to train {pattern_type}: {e}")
                results[pattern_type] = {"status": "error", "error": str(e)}

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_features": NUM_FEATURES,
            "total_samples": X.shape[0],
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
        """Train ensemble for a single pattern type."""
        logger.info(f"Training {pattern_type}: {X.shape[0]} samples, {np.sum(y)} positive")

        # Time series split (no data leakage)
        tscv = TimeSeriesSplit(n_splits=3)
        all_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Skip folds with insufficient class variety
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train ensemble members
            gbm = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_leaf=5, random_state=42
            )
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            )
            lr = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            )

            gbm.fit(X_train_scaled, y_train)
            rf.fit(X_train_scaled, y_train)
            lr.fit(X_train_scaled, y_train)

            # Ensemble prediction (weighted average of probabilities)
            p_gbm = gbm.predict_proba(X_test_scaled)[:, 1]
            p_rf = rf.predict_proba(X_test_scaled)[:, 1]
            p_lr = lr.predict_proba(X_test_scaled)[:, 1]
            p_ensemble = p_gbm * 0.4 + p_rf * 0.3 + p_lr * 0.3

            y_pred = (p_ensemble >= 0.5).astype(int)

            fold_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, p_ensemble) if len(np.unique(y_test)) > 1 else 0.5,
            }
            all_metrics.append(fold_metrics)

        if not all_metrics:
            return {"status": "error", "reason": "no_valid_folds"}

        # Train final model on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gbm = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        )
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        gbm.fit(X_scaled, y)
        rf.fit(X_scaled, y)
        lr.fit(X_scaled, y)

        # Get feature importance from GBM
        importance = gbm.feature_importances_
        feature_names = get_feature_engineer().get_feature_names()
        top_features = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1], reverse=True
        )[:10]

        # Save model
        model_data = {
            'gbm': gbm, 'rf': rf, 'lr': lr,
            'scaler': scaler,
            'symbol': symbol, 'timeframe': timeframe,
            'trained_at': datetime.utcnow().isoformat(),
            'n_samples': len(y),
        }
        self.models[pattern_type] = model_data

        model_path = MODEL_DIR / f"{pattern_type}_{symbol}_{timeframe}_ensemble.joblib"
        joblib.dump(model_data, model_path)

        # Average metrics across folds
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = round(np.mean([m[key] for m in all_metrics]), 4)

        avg_metrics['status'] = 'trained'
        avg_metrics['samples'] = len(y)
        avg_metrics['positive_ratio'] = round(np.mean(y), 4)
        avg_metrics['top_features'] = [{'name': n, 'importance': round(float(i), 4)} for n, i in top_features]

        self.metrics[pattern_type] = avg_metrics
        return avg_metrics

    def predict(self, pattern_type: str, features: np.ndarray) -> float:
        """
        Get probability prediction for a pattern type.

        Args:
            pattern_type: Pattern type (e.g., 'fvg', 'order_block')
            features: Feature vector of shape (42,) or (n, 42)

        Returns:
            Probability (0-1) that the pattern will be profitable
        """
        if pattern_type not in self.models:
            return 0.5  # Default when no model available

        model_data = self.models[pattern_type]
        scaler = model_data['scaler']

        if features.ndim == 1:
            features = features.reshape(1, -1)

        X_scaled = scaler.transform(features)

        # Ensemble prediction
        p_gbm = model_data['gbm'].predict_proba(X_scaled)[:, 1]
        p_rf = model_data['rf'].predict_proba(X_scaled)[:, 1]
        p_lr = model_data['lr'].predict_proba(X_scaled)[:, 1]

        p_ensemble = p_gbm * 0.4 + p_rf * 0.3 + p_lr * 0.3

        return float(p_ensemble[0]) if len(p_ensemble) == 1 else p_ensemble.tolist()

    def predict_all(self, features: np.ndarray) -> Dict[str, float]:
        """Get predictions for all trained pattern types."""
        predictions = {}
        for pattern_type in self.models:
            predictions[pattern_type] = self.predict(pattern_type, features)
        return predictions

    def get_feature_importance(self, pattern_type: str) -> List[Dict]:
        """Get feature importance for a specific pattern type."""
        if pattern_type not in self.models:
            return []

        model_data = self.models[pattern_type]
        gbm = model_data['gbm']
        importance = gbm.feature_importances_
        feature_names = get_feature_engineer().get_feature_names()

        return sorted(
            [{'name': n, 'importance': round(float(i), 4)} for n, i in zip(feature_names, importance)],
            key=lambda x: x['importance'], reverse=True
        )

    def get_status(self) -> Dict:
        """Get status of all trained classifiers."""
        return {
            "models_trained": len(self.models),
            "pattern_types": list(self.models.keys()),
            "metrics": self.metrics,
            "model_dir": str(MODEL_DIR),
        }

    def has_model(self, pattern_type: str) -> bool:
        """Check if a model is trained for a pattern type."""
        return pattern_type in self.models

    def _save_metrics(self, symbol: str, timeframe: str, pattern_type: str, metrics: Dict):
        """Save training metrics to SQLite."""
        try:
            from ..database import Database
            db = Database()
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ml_model_metrics
                    (symbol, timeframe, pattern_type, model_type,
                     accuracy, precision_score, recall, f1_score, auc_roc,
                     feature_importance, train_samples, test_samples, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timeframe, pattern_type, 'ensemble_gbm_rf_lr',
                    metrics.get('accuracy', 0), metrics.get('precision', 0),
                    metrics.get('recall', 0), metrics.get('f1', 0),
                    metrics.get('auc_roc', 0),
                    json.dumps(metrics.get('top_features', [])),
                    metrics.get('samples', 0), 0,
                    datetime.utcnow().isoformat(),
                ))
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")


# Singleton
_classifier_instance = None

def get_classifier() -> MLPatternClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = MLPatternClassifier()
    return _classifier_instance
