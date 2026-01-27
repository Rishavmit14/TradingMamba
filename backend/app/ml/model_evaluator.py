"""
Model Evaluation and Precision Tracking System

Tracks model performance over time and provides metrics for:
- Concept classification accuracy
- Signal prediction precision
- Trading outcome correlation
- Model improvement trends

100% FREE - Uses only scikit-learn metrics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import logging

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction for tracking"""
    prediction_id: str
    timestamp: str
    model_name: str
    prediction_type: str  # 'concept', 'signal', 'direction'
    predicted_value: any
    actual_value: any = None
    confidence: float = 0.0
    is_correct: bool = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Comprehensive metrics for a model"""
    model_name: str
    evaluation_timestamp: str
    n_samples: int
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = None
    mse: float = None
    mae: float = None
    r2: float = None
    confusion_matrix: List = None
    class_report: Dict = None
    custom_metrics: Dict = field(default_factory=dict)


class PredictionTracker:
    """
    Tracks all predictions for later evaluation.
    Enables correlation with actual outcomes.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.tracking_dir = self.data_dir / "ml_tracking"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        self.predictions: List[PredictionRecord] = []
        self.prediction_index: Dict[str, int] = {}  # id -> index

    def record_prediction(
        self,
        model_name: str,
        prediction_type: str,
        predicted_value: any,
        confidence: float = 0.0,
        metadata: Dict = None
    ) -> str:
        """Record a new prediction"""
        prediction_id = f"{model_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        record = PredictionRecord(
            prediction_id=prediction_id,
            timestamp=datetime.utcnow().isoformat(),
            model_name=model_name,
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            confidence=confidence,
            metadata=metadata or {}
        )

        self.predictions.append(record)
        self.prediction_index[prediction_id] = len(self.predictions) - 1

        return prediction_id

    def record_outcome(self, prediction_id: str, actual_value: any) -> bool:
        """Record the actual outcome for a prediction"""
        if prediction_id not in self.prediction_index:
            return False

        idx = self.prediction_index[prediction_id]
        self.predictions[idx].actual_value = actual_value

        # Determine if correct
        pred = self.predictions[idx].predicted_value
        if isinstance(pred, (list, tuple)):
            self.predictions[idx].is_correct = actual_value in pred
        else:
            self.predictions[idx].is_correct = pred == actual_value

        return True

    def get_predictions_for_evaluation(
        self,
        model_name: str = None,
        prediction_type: str = None,
        with_outcomes_only: bool = True
    ) -> List[PredictionRecord]:
        """Get predictions filtered by criteria"""
        filtered = self.predictions

        if model_name:
            filtered = [p for p in filtered if p.model_name == model_name]

        if prediction_type:
            filtered = [p for p in filtered if p.prediction_type == prediction_type]

        if with_outcomes_only:
            filtered = [p for p in filtered if p.actual_value is not None]

        return filtered

    def save(self):
        """Save tracking data"""
        save_path = self.tracking_dir / "predictions.json"
        with open(save_path, 'w') as f:
            json.dump([asdict(p) for p in self.predictions], f, indent=2, default=str)

    def load(self):
        """Load tracking data"""
        save_path = self.tracking_dir / "predictions.json"
        if save_path.exists():
            with open(save_path) as f:
                data = json.load(f)
                self.predictions = [PredictionRecord(**p) for p in data]
                self.prediction_index = {p.prediction_id: i for i, p in enumerate(self.predictions)}


class ModelEvaluator:
    """
    Evaluates model performance with comprehensive metrics.
    """

    def __init__(self, tracker: PredictionTracker = None):
        self.tracker = tracker or PredictionTracker()
        self.evaluation_history: List[ModelMetrics] = []

    def evaluate_classifier(
        self,
        y_true: List,
        y_pred: List,
        model_name: str,
        labels: List = None
    ) -> ModelMetrics:
        """Evaluate a classification model"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = ModelMetrics(
            model_name=model_name,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            n_samples=len(y_true)
        )

        # Basic metrics
        metrics.accuracy = float(accuracy_score(y_true, y_pred))
        metrics.precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics.recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics.f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics.confusion_matrix = cm.tolist()

        # Classification report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        metrics.class_report = report

        self.evaluation_history.append(metrics)
        return metrics

    def evaluate_regressor(
        self,
        y_true: List,
        y_pred: List,
        model_name: str
    ) -> ModelMetrics:
        """Evaluate a regression model"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = ModelMetrics(
            model_name=model_name,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            n_samples=len(y_true)
        )

        metrics.mse = float(mean_squared_error(y_true, y_pred))
        metrics.mae = float(mean_absolute_error(y_true, y_pred))
        metrics.r2 = float(r2_score(y_true, y_pred))

        self.evaluation_history.append(metrics)
        return metrics

    def evaluate_from_tracker(self, model_name: str, prediction_type: str) -> Optional[ModelMetrics]:
        """Evaluate model using tracked predictions"""
        predictions = self.tracker.get_predictions_for_evaluation(
            model_name=model_name,
            prediction_type=prediction_type,
            with_outcomes_only=True
        )

        if not predictions:
            return None

        y_true = [p.actual_value for p in predictions]
        y_pred = [p.predicted_value for p in predictions]

        # Determine if classification or regression
        if all(isinstance(v, (int, bool, str)) for v in y_true):
            return self.evaluate_classifier(y_true, y_pred, model_name)
        else:
            return self.evaluate_regressor(y_true, y_pred, model_name)


class PerformanceDashboard:
    """
    Aggregates performance metrics across all models.
    Tracks improvement over time.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.metrics_dir = self.data_dir / "ml_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.model_histories: Dict[str, List[ModelMetrics]] = defaultdict(list)
        self.trading_outcomes: List[Dict] = []

    def record_metrics(self, metrics: ModelMetrics):
        """Record metrics for a model"""
        self.model_histories[metrics.model_name].append(metrics)
        self._save_metrics(metrics)

    def record_trading_outcome(
        self,
        signal_id: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        profit_loss: float,
        concepts_used: List[str],
        model_confidence: float
    ):
        """Record actual trading outcome for model feedback"""
        outcome = {
            'signal_id': signal_id,
            'timestamp': datetime.utcnow().isoformat(),
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'is_winner': profit_loss > 0,
            'concepts_used': concepts_used,
            'model_confidence': model_confidence,
        }
        self.trading_outcomes.append(outcome)
        self._save_outcome(outcome)

    def get_model_trend(self, model_name: str, metric: str = 'f1') -> Dict:
        """Get performance trend for a model"""
        history = self.model_histories.get(model_name, [])

        if not history:
            return {'trend': 'no_data', 'values': []}

        values = [getattr(m, metric, None) for m in history if getattr(m, metric, None) is not None]

        if len(values) < 2:
            return {'trend': 'insufficient_data', 'values': values}

        # Calculate trend
        if values[-1] > values[0]:
            trend = 'improving'
            improvement = (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
        elif values[-1] < values[0]:
            trend = 'declining'
            improvement = (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
        else:
            trend = 'stable'
            improvement = 0

        return {
            'trend': trend,
            'improvement_percent': improvement,
            'current_value': values[-1],
            'best_value': max(values),
            'worst_value': min(values),
            'n_evaluations': len(values),
            'values': values[-20:]  # Last 20
        }

    def get_concept_performance(self) -> Dict[str, Dict]:
        """Analyze which Smart Money concepts correlate with winning trades"""
        if not self.trading_outcomes:
            return {}

        concept_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})

        for outcome in self.trading_outcomes:
            for concept in outcome.get('concepts_used', []):
                if outcome['is_winner']:
                    concept_stats[concept]['wins'] += 1
                else:
                    concept_stats[concept]['losses'] += 1
                concept_stats[concept]['total_pnl'] += outcome['profit_loss']

        # Calculate win rates
        result = {}
        for concept, stats in concept_stats.items():
            total = stats['wins'] + stats['losses']
            result[concept] = {
                'win_rate': stats['wins'] / total if total > 0 else 0,
                'total_trades': total,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': stats['total_pnl'] / total if total > 0 else 0,
            }

        return dict(sorted(result.items(), key=lambda x: x[1]['win_rate'], reverse=True))

    def get_confidence_calibration(self) -> Dict:
        """Check if model confidence correlates with actual accuracy"""
        if not self.trading_outcomes:
            return {}

        # Bin outcomes by confidence
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
        calibration = {}

        for low, high in bins:
            bin_outcomes = [o for o in self.trading_outcomes
                          if low <= o.get('model_confidence', 0) < high]

            if bin_outcomes:
                actual_win_rate = sum(1 for o in bin_outcomes if o['is_winner']) / len(bin_outcomes)
                expected_win_rate = (low + high) / 2

                calibration[f'{low:.0%}-{high:.0%}'] = {
                    'expected': expected_win_rate,
                    'actual': actual_win_rate,
                    'n_trades': len(bin_outcomes),
                    'calibration_error': abs(expected_win_rate - actual_win_rate)
                }

        return calibration

    def get_summary_report(self) -> Dict:
        """Generate comprehensive performance summary"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'models': {},
            'trading_performance': {},
            'concept_analysis': {},
            'recommendations': []
        }

        # Model performance
        for model_name, history in self.model_histories.items():
            if history:
                latest = history[-1]
                trend = self.get_model_trend(model_name)
                report['models'][model_name] = {
                    'current_f1': latest.f1,
                    'current_accuracy': latest.accuracy,
                    'trend': trend['trend'],
                    'n_evaluations': len(history),
                }

        # Trading performance
        if self.trading_outcomes:
            winners = [o for o in self.trading_outcomes if o['is_winner']]
            report['trading_performance'] = {
                'total_trades': len(self.trading_outcomes),
                'win_rate': len(winners) / len(self.trading_outcomes),
                'total_pnl': sum(o['profit_loss'] for o in self.trading_outcomes),
                'avg_winner': sum(o['profit_loss'] for o in winners) / len(winners) if winners else 0,
                'avg_loser': sum(o['profit_loss'] for o in self.trading_outcomes if not o['is_winner']) / (len(self.trading_outcomes) - len(winners)) if len(self.trading_outcomes) > len(winners) else 0,
            }

        # Concept analysis
        report['concept_analysis'] = self.get_concept_performance()

        # Generate recommendations
        recommendations = []

        # Check model trends
        for model_name, history in self.model_histories.items():
            trend = self.get_model_trend(model_name)
            if trend['trend'] == 'declining':
                recommendations.append(f"Model '{model_name}' is declining. Consider retraining with more data.")

        # Check concept performance
        concept_perf = report['concept_analysis']
        for concept, stats in concept_perf.items():
            if stats['total_trades'] >= 10 and stats['win_rate'] < 0.4:
                recommendations.append(f"Concept '{concept}' has low win rate ({stats['win_rate']:.0%}). Review its usage.")
            if stats['total_trades'] >= 10 and stats['win_rate'] > 0.7:
                recommendations.append(f"Concept '{concept}' shows high win rate ({stats['win_rate']:.0%}). Consider increasing its weight.")

        # Check calibration
        calibration = self.get_confidence_calibration()
        for bin_name, cal in calibration.items():
            if cal['calibration_error'] > 0.2 and cal['n_trades'] >= 10:
                recommendations.append(f"Confidence bin {bin_name} is miscalibrated (error: {cal['calibration_error']:.0%}). Consider adjusting confidence scoring.")

        report['recommendations'] = recommendations

        return report

    def _save_metrics(self, metrics: ModelMetrics):
        """Save metrics to disk"""
        save_path = self.metrics_dir / f"{metrics.model_name}_metrics.jsonl"
        with open(save_path, 'a') as f:
            f.write(json.dumps(asdict(metrics), default=str) + '\n')

    def _save_outcome(self, outcome: Dict):
        """Save trading outcome to disk"""
        save_path = self.metrics_dir / "trading_outcomes.jsonl"
        with open(save_path, 'a') as f:
            f.write(json.dumps(outcome) + '\n')

    def load(self):
        """Load historical data"""
        # Load metrics
        for metrics_file in self.metrics_dir.glob("*_metrics.jsonl"):
            model_name = metrics_file.stem.replace('_metrics', '')
            with open(metrics_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        self.model_histories[model_name].append(ModelMetrics(**data))
                    except:
                        pass

        # Load outcomes
        outcomes_file = self.metrics_dir / "trading_outcomes.jsonl"
        if outcomes_file.exists():
            with open(outcomes_file) as f:
                for line in f:
                    try:
                        self.trading_outcomes.append(json.loads(line))
                    except:
                        pass


class AutoTuner:
    """
    Automatically tunes model parameters based on performance feedback.
    Implements simple optimization without external libraries.
    """

    def __init__(self, dashboard: PerformanceDashboard):
        self.dashboard = dashboard
        self.tuning_history: List[Dict] = []

    def suggest_concept_weights(self) -> Dict[str, float]:
        """Suggest concept weights based on trading performance"""
        concept_perf = self.dashboard.get_concept_performance()

        if not concept_perf:
            return {}

        # Base weights
        weights = {}
        for concept, stats in concept_perf.items():
            if stats['total_trades'] >= 5:
                # Weight based on win rate and sample size
                confidence = min(stats['total_trades'] / 20, 1.0)  # More trades = more confidence
                weight = stats['win_rate'] * confidence + 0.5 * (1 - confidence)
                weights[concept] = round(weight, 3)

        return weights

    def suggest_confidence_adjustment(self) -> Dict:
        """Suggest adjustments to confidence scoring"""
        calibration = self.dashboard.get_confidence_calibration()

        adjustments = {}
        for bin_name, cal in calibration.items():
            if cal['n_trades'] >= 10:
                # If actual < expected, model is overconfident
                # If actual > expected, model is underconfident
                error = cal['actual'] - cal['expected']
                adjustments[bin_name] = {
                    'direction': 'increase' if error > 0.1 else 'decrease' if error < -0.1 else 'keep',
                    'magnitude': abs(error),
                }

        return adjustments

    def generate_tuning_recommendations(self) -> List[str]:
        """Generate actionable tuning recommendations"""
        recommendations = []

        # Concept weights
        weights = self.suggest_concept_weights()
        high_performers = [c for c, w in weights.items() if w > 0.7]
        low_performers = [c for c, w in weights.items() if w < 0.4]

        if high_performers:
            recommendations.append(f"Increase weight for high-performing concepts: {', '.join(high_performers)}")
        if low_performers:
            recommendations.append(f"Decrease weight for low-performing concepts: {', '.join(low_performers)}")

        # Confidence adjustment
        conf_adj = self.suggest_confidence_adjustment()
        overconfident = [b for b, a in conf_adj.items() if a['direction'] == 'decrease' and a['magnitude'] > 0.15]
        underconfident = [b for b, a in conf_adj.items() if a['direction'] == 'increase' and a['magnitude'] > 0.15]

        if overconfident:
            recommendations.append(f"Model is overconfident in bins: {', '.join(overconfident)}. Consider lowering confidence scores.")
        if underconfident:
            recommendations.append(f"Model is underconfident in bins: {', '.join(underconfident)}. Consider raising confidence scores.")

        return recommendations


def test_evaluator():
    """Test the evaluation system"""
    print("=" * 60)
    print("MODEL EVALUATION SYSTEM TEST")
    print("=" * 60)

    # Create test predictions
    tracker = PredictionTracker()

    # Simulate some predictions
    concepts = ['order_block', 'fair_value_gap', 'liquidity', 'market_structure']

    for i in range(50):
        pred_id = tracker.record_prediction(
            model_name='concept_classifier',
            prediction_type='concept',
            predicted_value=concepts[i % len(concepts)],
            confidence=0.5 + np.random.rand() * 0.4
        )

        # Simulate outcome (70% accuracy)
        if np.random.rand() < 0.7:
            actual = concepts[i % len(concepts)]
        else:
            actual = concepts[(i + 1) % len(concepts)]

        tracker.record_outcome(pred_id, actual)

    # Evaluate
    evaluator = ModelEvaluator(tracker)
    metrics = evaluator.evaluate_from_tracker('concept_classifier', 'concept')

    print(f"\n📊 Classifier Metrics:")
    print(f"   Accuracy: {metrics.accuracy:.2%}")
    print(f"   Precision: {metrics.precision:.2%}")
    print(f"   Recall: {metrics.recall:.2%}")
    print(f"   F1 Score: {metrics.f1:.2%}")

    # Test dashboard
    dashboard = PerformanceDashboard()
    dashboard.record_metrics(metrics)

    # Simulate trading outcomes
    for i in range(30):
        dashboard.record_trading_outcome(
            signal_id=f'signal_{i}',
            direction='bullish' if i % 2 == 0 else 'bearish',
            entry_price=1.1000 + np.random.rand() * 0.01,
            exit_price=1.1000 + np.random.rand() * 0.02 - 0.005,
            profit_loss=np.random.rand() * 100 - 30,  # -30 to +70
            concepts_used=[concepts[i % len(concepts)], concepts[(i+1) % len(concepts)]],
            model_confidence=0.5 + np.random.rand() * 0.4
        )

    # Generate report
    report = dashboard.get_summary_report()

    print(f"\n📈 Trading Performance:")
    print(f"   Total Trades: {report['trading_performance']['total_trades']}")
    print(f"   Win Rate: {report['trading_performance']['win_rate']:.2%}")
    print(f"   Total P&L: {report['trading_performance']['total_pnl']:.2f}")

    print(f"\n🎯 Concept Performance:")
    for concept, stats in list(report['concept_analysis'].items())[:3]:
        print(f"   {concept}: {stats['win_rate']:.0%} win rate ({stats['total_trades']} trades)")

    print(f"\n💡 Recommendations:")
    for rec in report['recommendations'][:3]:
        print(f"   • {rec}")

    # Test auto-tuner
    tuner = AutoTuner(dashboard)
    weights = tuner.suggest_concept_weights()

    print(f"\n⚙️ Suggested Concept Weights:")
    for concept, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"   {concept}: {weight:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_evaluator()
