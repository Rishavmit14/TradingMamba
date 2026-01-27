"""
ICT Concept Classifier

Multi-label classifier that identifies ICT concepts in text.
Uses ensemble of models for robust predictions.

100% FREE - scikit-learn based, no external APIs.
Learns incrementally as more data is added.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
    from sklearn.preprocessing import MultiLabelBinarizer
    import joblib
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn")

from .feature_extractor import ICTFeatureExtractor, ICT_VOCABULARY


class ICTConceptClassifier:
    """
    Classifies text segments into ICT concepts.

    Features:
    - Multi-label classification (text can contain multiple concepts)
    - Ensemble of multiple models for robustness
    - Incremental learning support
    - Confidence scoring
    - Model performance tracking
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.models_dir = self.data_dir / "ml_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Feature extractor
        self.feature_extractor = ICTFeatureExtractor(str(self.data_dir))

        # Concept labels
        self.concept_labels = list(ICT_VOCABULARY.keys())
        self.mlb = MultiLabelBinarizer(classes=self.concept_labels)
        self.mlb.fit([self.concept_labels])  # Fit with all possible labels

        # Ensemble of classifiers
        self.classifiers = {
            'random_forest': OneVsRestClassifier(
                RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            ),
            'logistic': OneVsRestClassifier(
                LogisticRegression(max_iter=1000, random_state=42)
            ),
            'gradient_boost': OneVsRestClassifier(
                GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
            ),
        }

        # Model weights for ensemble (learned from validation)
        self.model_weights = {name: 1.0 for name in self.classifiers}

        # Performance tracking
        self.performance_history = []
        self.is_fitted = False

        # Training data accumulator for incremental learning
        self.training_data = {
            'texts': [],
            'labels': [],
            'timestamps': []
        }

    def _extract_labels_from_transcript(self, transcript: Dict) -> List[str]:
        """Extract concept labels from transcript based on keyword matching"""
        text = transcript.get('full_text', '').lower()
        labels = []

        for concept, terms in ICT_VOCABULARY.items():
            for term in terms:
                if term in text:
                    labels.append(concept)
                    break

        return labels if labels else ['market_structure']  # Default label

    def prepare_training_data(self, transcripts: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from transcripts"""
        # Store for incremental learning
        for t in transcripts:
            self.training_data['texts'].append(t.get('full_text', ''))
            self.training_data['labels'].append(self._extract_labels_from_transcript(t))
            self.training_data['timestamps'].append(datetime.utcnow().isoformat())

        # Extract features
        X, feature_names = self.feature_extractor.fit_transform(transcripts)

        # Prepare labels
        labels = [self._extract_labels_from_transcript(t) for t in transcripts]
        y = self.mlb.transform(labels)

        return X, y

    def fit(self, transcripts: List[Dict], validation_split: float = 0.2) -> Dict:
        """Train the classifier ensemble"""
        print(f"Training on {len(transcripts)} transcripts...")

        X, y = self.prepare_training_data(transcripts)

        # Split for validation
        if len(transcripts) > 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y

        # Train each classifier
        model_scores = {}
        for name, clf in self.classifiers.items():
            print(f"  Training {name}...")
            try:
                clf.fit(X_train, y_train)

                # Evaluate on validation set
                y_pred = clf.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                model_scores[name] = f1
                print(f"    {name} F1 score: {f1:.3f}")
            except Exception as e:
                print(f"    {name} failed: {e}")
                model_scores[name] = 0.0

        # Update model weights based on performance
        total_score = sum(model_scores.values())
        if total_score > 0:
            self.model_weights = {name: score / total_score for name, score in model_scores.items()}

        # Calculate overall metrics
        y_pred_ensemble = self._ensemble_predict(X_val)
        overall_f1 = f1_score(y_val, y_pred_ensemble, average='weighted', zero_division=0)

        # Track performance
        performance = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_samples': len(transcripts),
            'model_scores': model_scores,
            'ensemble_f1': overall_f1,
            'model_weights': self.model_weights.copy()
        }
        self.performance_history.append(performance)

        self.is_fitted = True

        print(f"\nEnsemble F1 score: {overall_f1:.3f}")
        print(f"Model weights: {self.model_weights}")

        return performance

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using weighted ensemble"""
        predictions = []

        for name, clf in self.classifiers.items():
            try:
                pred_proba = clf.predict_proba(X)
                # Handle different output formats
                if isinstance(pred_proba, list):
                    pred_proba = np.array([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in pred_proba]).T
                predictions.append((pred_proba, self.model_weights[name]))
            except:
                pass

        if not predictions:
            return np.zeros((X.shape[0], len(self.concept_labels)))

        # Weighted average of predictions
        weighted_sum = sum(pred * weight for pred, weight in predictions)
        total_weight = sum(weight for _, weight in predictions)

        avg_proba = weighted_sum / total_weight

        # Convert to binary predictions (threshold = 0.3 for multi-label)
        return (avg_proba > 0.3).astype(int)

    def predict(self, texts: List[str]) -> List[Dict]:
        """Predict concepts for new texts"""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        # Create pseudo-transcripts for feature extraction
        transcripts = [{'full_text': text, 'segments': [], 'word_count': len(text.split())} for text in texts]

        # Extract features
        X, _ = self.feature_extractor.transform(transcripts)

        # Ensemble prediction
        y_pred = self._ensemble_predict(X)

        # Get probabilities for confidence scores
        predictions = []
        for i, text in enumerate(texts):
            pred_labels = self.mlb.inverse_transform(y_pred[i:i+1])[0]

            # Calculate confidence from individual models
            confidences = {}
            for label in pred_labels:
                label_idx = self.concept_labels.index(label)
                conf_scores = []
                for name, clf in self.classifiers.items():
                    try:
                        proba = clf.predict_proba(X[i:i+1])
                        if isinstance(proba, list):
                            conf = proba[label_idx][0, 1] if proba[label_idx].shape[1] > 1 else proba[label_idx][0, 0]
                        else:
                            conf = proba[0, label_idx]
                        conf_scores.append(conf * self.model_weights[name])
                    except:
                        pass
                confidences[label] = sum(conf_scores) / sum(self.model_weights.values()) if conf_scores else 0.5

            predictions.append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'concepts': list(pred_labels),
                'confidences': confidences,
                'n_concepts': len(pred_labels)
            })

        return predictions

    def predict_segment(self, segment: Dict) -> Dict:
        """Predict concepts for a single transcript segment"""
        result = self.predict([segment.get('text', '')])[0]
        result['start_time'] = segment.get('start_time', 0)
        result['end_time'] = segment.get('end_time', 0)
        return result

    def incremental_fit(self, new_transcripts: List[Dict]) -> Dict:
        """Add new training data and retrain"""
        print(f"Incremental training with {len(new_transcripts)} new transcripts...")

        # Add to training data
        for t in new_transcripts:
            self.training_data['texts'].append(t.get('full_text', ''))
            self.training_data['labels'].append(self._extract_labels_from_transcript(t))
            self.training_data['timestamps'].append(datetime.utcnow().isoformat())

        # Reconstruct full dataset
        all_transcripts = [
            {'full_text': text, 'segments': [], 'word_count': len(text.split())}
            for text in self.training_data['texts']
        ]

        # Retrain
        return self.fit(all_transcripts)

    def get_performance_trend(self) -> Dict:
        """Get performance improvement over time"""
        if not self.performance_history:
            return {'trend': 'no_data', 'history': []}

        f1_scores = [p['ensemble_f1'] for p in self.performance_history]

        if len(f1_scores) >= 2:
            trend = 'improving' if f1_scores[-1] > f1_scores[0] else 'declining'
            improvement = f1_scores[-1] - f1_scores[0]
        else:
            trend = 'insufficient_data'
            improvement = 0

        return {
            'trend': trend,
            'improvement': improvement,
            'current_f1': f1_scores[-1] if f1_scores else 0,
            'best_f1': max(f1_scores) if f1_scores else 0,
            'n_training_runs': len(self.performance_history),
            'history': self.performance_history[-10:]  # Last 10 training runs
        }

    def save(self, path: str = None):
        """Save the trained classifier"""
        if path is None:
            path = self.models_dir / "concept_classifier.joblib"

        # Save feature extractor separately
        self.feature_extractor.save()

        joblib.dump({
            'classifiers': self.classifiers,
            'model_weights': self.model_weights,
            'mlb': self.mlb,
            'concept_labels': self.concept_labels,
            'performance_history': self.performance_history,
            'training_data': self.training_data,
            'is_fitted': self.is_fitted,
        }, path)
        print(f"Classifier saved to {path}")

    def load(self, path: str = None):
        """Load a trained classifier"""
        if path is None:
            path = self.models_dir / "concept_classifier.joblib"

        # Load feature extractor
        self.feature_extractor.load()

        data = joblib.load(path)
        self.classifiers = data['classifiers']
        self.model_weights = data['model_weights']
        self.mlb = data['mlb']
        self.concept_labels = data['concept_labels']
        self.performance_history = data['performance_history']
        self.training_data = data['training_data']
        self.is_fitted = data['is_fitted']
        print(f"Classifier loaded from {path}")
        return self


class ConceptSequenceAnalyzer:
    """
    Analyzes sequences of ICT concepts to learn patterns.
    Useful for understanding how ICT teaches concepts progressively.
    """

    def __init__(self):
        self.transition_matrix = None
        self.concept_labels = list(ICT_VOCABULARY.keys())
        self.n_concepts = len(self.concept_labels)

    def build_transition_matrix(self, transcripts: List[Dict]) -> np.ndarray:
        """Build concept transition probability matrix"""
        transitions = np.zeros((self.n_concepts, self.n_concepts))

        for transcript in transcripts:
            segments = transcript.get('segments', [])
            prev_concepts = set()

            for segment in segments:
                text = segment.get('text', '').lower()
                current_concepts = set()

                for i, concept in enumerate(self.concept_labels):
                    for term in ICT_VOCABULARY[concept]:
                        if term in text:
                            current_concepts.add(i)
                            break

                # Record transitions
                for prev in prev_concepts:
                    for curr in current_concepts:
                        transitions[prev, curr] += 1

                prev_concepts = current_concepts

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(transitions, row_sums,
                                           where=row_sums != 0,
                                           out=np.zeros_like(transitions))

        return self.transition_matrix

    def get_likely_next_concepts(self, current_concept: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get most likely next concepts after a given concept"""
        if self.transition_matrix is None:
            return []

        try:
            idx = self.concept_labels.index(current_concept)
        except ValueError:
            return []

        probs = self.transition_matrix[idx]
        top_indices = np.argsort(probs)[::-1][:top_k]

        return [(self.concept_labels[i], float(probs[i])) for i in top_indices if probs[i] > 0]

    def get_concept_clusters(self) -> Dict[str, List[str]]:
        """Find clusters of frequently co-occurring concepts"""
        if self.transition_matrix is None:
            return {}

        # Use symmetric version for clustering
        symmetric = (self.transition_matrix + self.transition_matrix.T) / 2

        # Simple clustering based on high transition probability
        clusters = defaultdict(list)
        threshold = 0.1

        for i, concept in enumerate(self.concept_labels):
            for j, other in enumerate(self.concept_labels):
                if i != j and symmetric[i, j] > threshold:
                    clusters[concept].append(other)

        return dict(clusters)


# Test function
def test_classifier():
    """Test the concept classifier"""
    print("Testing ICT Concept Classifier")
    print("=" * 50)

    # Sample transcripts
    sample_transcripts = [
        {
            'video_id': 'test1',
            'full_text': '''Order blocks are key institutional levels.
                           When price returns to an order block, smart money often reacts.
                           Look for fair value gaps as entry points.
                           The premium zone is above equilibrium.''',
            'segments': [
                {'start_time': 0, 'end_time': 30, 'text': 'Order blocks are key'},
                {'start_time': 30, 'end_time': 60, 'text': 'fair value gaps as entry'},
            ],
        },
        {
            'video_id': 'test2',
            'full_text': '''Break of structure confirms the trend.
                           Change of character signals reversal.
                           Liquidity sweeps trap retail traders.
                           Trade from discount to premium.''',
            'segments': [],
        },
        {
            'video_id': 'test3',
            'full_text': '''The kill zone is optimal trading time.
                           London session has the most volatility.
                           Look for OTE in the discount zone.
                           Silver bullet setups at specific times.''',
            'segments': [],
        },
    ]

    classifier = ICTConceptClassifier()

    # Train
    print("\n1. Training classifier...")
    performance = classifier.fit(sample_transcripts)
    print(f"   Training complete. Ensemble F1: {performance['ensemble_f1']:.3f}")

    # Predict
    print("\n2. Testing predictions...")
    test_texts = [
        "The order block at this level shows institutional interest",
        "Wait for the liquidity sweep before entering",
        "This is a classic break of structure signal"
    ]

    predictions = classifier.predict(test_texts)
    for pred in predictions:
        print(f"\n   Text: {pred['text'][:50]}...")
        print(f"   Concepts: {pred['concepts']}")
        print(f"   Confidences: {pred['confidences']}")

    # Test sequence analyzer
    print("\n3. Testing sequence analyzer...")
    analyzer = ConceptSequenceAnalyzer()
    analyzer.build_transition_matrix(sample_transcripts)

    print("   Likely concepts after 'order_block':")
    next_concepts = analyzer.get_likely_next_concepts('order_block')
    for concept, prob in next_concepts:
        print(f"      {concept}: {prob:.3f}")

    print("\n" + "=" * 50)
    print("Classifier test complete!")


if __name__ == "__main__":
    test_classifier()
