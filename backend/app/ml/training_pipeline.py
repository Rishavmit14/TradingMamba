"""
ICT Trading ML Training Pipeline

Orchestrates the entire ML workflow:
1. Load transcripts
2. Extract features
3. Train concept classifier
4. Build pattern recognition models
5. Train signal prediction model
6. Evaluate and track performance

100% FREE - Uses only open-source libraries.
Scalable - Handles incremental data addition.
Self-improving - Tracks and optimizes over time.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import logging

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    import joblib
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn")

from .feature_extractor import ICTFeatureExtractor, ConceptEmbedding, ICT_VOCABULARY
from .concept_classifier import ICTConceptClassifier, ConceptSequenceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICTKnowledgeBase:
    """
    Central knowledge base that learns from all transcripts.
    Combines multiple learning approaches for comprehensive ICT understanding.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.concepts_dir = self.data_dir / "concepts"
        self.models_dir = self.data_dir / "ml_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.feature_extractor = ICTFeatureExtractor(str(self.data_dir))
        self.concept_classifier = ICTConceptClassifier(str(self.data_dir))
        self.concept_embeddings = ConceptEmbedding(embedding_dim=64)
        self.sequence_analyzer = ConceptSequenceAnalyzer()

        # Knowledge storage
        self.concept_definitions = {}  # Learned definitions
        self.concept_relationships = {}  # How concepts relate
        self.concept_contexts = defaultdict(list)  # Example contexts
        self.trading_rules = []  # Extracted trading rules

        # Metadata
        self.n_transcripts_processed = 0
        self.last_training_time = None
        self.training_history = []

    def load_transcripts(self, limit: int = None) -> List[Dict]:
        """Load all available transcripts"""
        transcripts = []
        transcript_files = sorted(self.transcripts_dir.glob("*.json"))

        if limit:
            transcript_files = transcript_files[:limit]

        for tf in transcript_files:
            try:
                with open(tf) as f:
                    transcript = json.load(f)
                    if transcript.get('full_text'):
                        transcripts.append(transcript)
            except Exception as e:
                logger.warning(f"Error loading {tf}: {e}")

        logger.info(f"Loaded {len(transcripts)} transcripts")
        return transcripts

    def extract_concept_definitions(self, transcripts: List[Dict]) -> Dict[str, Dict]:
        """Extract concept definitions from transcripts"""
        definitions = defaultdict(lambda: {'examples': [], 'contexts': [], 'frequency': 0})

        definition_patterns = [
            r"(?:is|are|means?|refers? to|defined as)\s+(.{20,200})",
            r"(?:what|when|how)\s+(?:is|are|do)\s+(.{20,200})",
        ]

        import re

        for transcript in transcripts:
            text = transcript.get('full_text', '')
            segments = transcript.get('segments', [])

            for concept, terms in ICT_VOCABULARY.items():
                for term in terms:
                    # Find sentences containing the term
                    pattern = rf"[^.]*\b{re.escape(term)}\b[^.]*\."
                    matches = re.findall(pattern, text.lower())

                    for match in matches[:5]:  # Limit examples per concept
                        definitions[concept]['examples'].append(match.strip())
                        definitions[concept]['frequency'] += 1

                    # Look for definitions
                    for def_pattern in definition_patterns:
                        full_pattern = rf"\b{re.escape(term)}\b\s*{def_pattern}"
                        def_matches = re.findall(full_pattern, text.lower())
                        definitions[concept]['contexts'].extend(def_matches[:3])

        self.concept_definitions = dict(definitions)
        return self.concept_definitions

    def extract_trading_rules(self, transcripts: List[Dict]) -> List[Dict]:
        """Extract trading rules mentioned in transcripts"""
        import re

        rules = []
        rule_patterns = [
            (r"(?:when|if)\s+(.{10,100}),?\s*(?:then|we|you)\s+(.{10,100})", "conditional"),
            (r"(?:always|never|must)\s+(.{10,100})", "imperative"),
            (r"(?:look for|wait for)\s+(.{10,100})(?:before|then)", "setup"),
            (r"(?:entry|exit|stop loss|take profit)\s+(?:at|when|is)\s+(.{10,100})", "execution"),
        ]

        for transcript in transcripts:
            text = transcript.get('full_text', '').lower()
            video_id = transcript.get('video_id', '')

            for pattern, rule_type in rule_patterns:
                matches = re.findall(pattern, text)
                for match in matches[:10]:  # Limit per transcript
                    if isinstance(match, tuple):
                        rule_text = ' -> '.join(match)
                    else:
                        rule_text = match

                    # Check if rule mentions ICT concepts
                    mentioned_concepts = []
                    for concept, terms in ICT_VOCABULARY.items():
                        for term in terms:
                            if term in rule_text:
                                mentioned_concepts.append(concept)
                                break

                    if mentioned_concepts:
                        rules.append({
                            'text': rule_text.strip(),
                            'type': rule_type,
                            'concepts': list(set(mentioned_concepts)),
                            'source_video': video_id,
                        })

        # Deduplicate similar rules
        unique_rules = []
        seen = set()
        for rule in rules:
            rule_key = rule['text'][:50]
            if rule_key not in seen:
                seen.add(rule_key)
                unique_rules.append(rule)

        self.trading_rules = unique_rules
        logger.info(f"Extracted {len(unique_rules)} trading rules")
        return unique_rules

    def train(self, transcripts: List[Dict] = None, incremental: bool = False) -> Dict:
        """Train all ML components"""
        logger.info("Starting training pipeline...")

        if transcripts is None:
            transcripts = self.load_transcripts()

        if not transcripts:
            logger.warning("No transcripts available for training")
            return {'status': 'no_data'}

        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_transcripts': len(transcripts),
            'components': {}
        }

        # 1. Train concept classifier
        logger.info("Training concept classifier...")
        if incremental and self.concept_classifier.is_fitted:
            clf_result = self.concept_classifier.incremental_fit(transcripts)
        else:
            clf_result = self.concept_classifier.fit(transcripts)
        results['components']['classifier'] = clf_result

        # 2. Build concept embeddings
        logger.info("Building concept embeddings...")
        self.concept_embeddings.fit(transcripts)
        results['components']['embeddings'] = {'status': 'trained', 'dim': self.concept_embeddings.embedding_dim}

        # 3. Build sequence analyzer
        logger.info("Analyzing concept sequences...")
        self.sequence_analyzer.build_transition_matrix(transcripts)
        results['components']['sequence_analyzer'] = {'status': 'trained'}

        # 4. Extract concept definitions
        logger.info("Extracting concept definitions...")
        definitions = self.extract_concept_definitions(transcripts)
        results['components']['definitions'] = {'n_concepts': len(definitions)}

        # 5. Extract trading rules
        logger.info("Extracting trading rules...")
        rules = self.extract_trading_rules(transcripts)
        results['components']['rules'] = {'n_rules': len(rules)}

        # Update metadata
        self.n_transcripts_processed = len(transcripts)
        self.last_training_time = datetime.utcnow().isoformat()
        self.training_history.append(results)

        logger.info(f"Training complete. Processed {len(transcripts)} transcripts.")
        return results

    def query_concept(self, concept_name: str) -> Dict:
        """Get comprehensive information about a concept"""
        if concept_name not in ICT_VOCABULARY:
            return {'error': f'Unknown concept: {concept_name}'}

        result = {
            'name': concept_name,
            'terms': ICT_VOCABULARY[concept_name],
        }

        # Add definition if available
        if concept_name in self.concept_definitions:
            result['definition'] = self.concept_definitions[concept_name]

        # Add related concepts from embeddings
        if self.concept_embeddings.embeddings is not None:
            similar = self.concept_embeddings.get_similar_concepts(concept_name, top_k=5)
            result['related_concepts'] = similar

        # Add sequence information
        next_concepts = self.sequence_analyzer.get_likely_next_concepts(concept_name)
        result['typically_followed_by'] = next_concepts

        # Add relevant rules
        relevant_rules = [r for r in self.trading_rules if concept_name in r.get('concepts', [])]
        result['trading_rules'] = relevant_rules[:5]

        return result

    def predict_concepts(self, text: str) -> Dict:
        """Predict ICT concepts in new text"""
        if not self.concept_classifier.is_fitted:
            return {'error': 'Model not trained. Call train() first.'}

        predictions = self.concept_classifier.predict([text])
        return predictions[0] if predictions else {}

    def get_learning_progress(self) -> Dict:
        """Get summary of learning progress"""
        classifier_trend = self.concept_classifier.get_performance_trend()

        return {
            'n_transcripts_processed': self.n_transcripts_processed,
            'last_training': self.last_training_time,
            'n_training_runs': len(self.training_history),
            'n_concepts_defined': len(self.concept_definitions),
            'n_rules_extracted': len(self.trading_rules),
            'classifier_performance': classifier_trend,
            'model_components': {
                'classifier': 'trained' if self.concept_classifier.is_fitted else 'not_trained',
                'embeddings': 'trained' if self.concept_embeddings.embeddings is not None else 'not_trained',
                'sequences': 'trained' if self.sequence_analyzer.transition_matrix is not None else 'not_trained',
            }
        }

    def save(self):
        """Save the entire knowledge base"""
        logger.info("Saving knowledge base...")

        # Save components
        self.concept_classifier.save()

        # Save embeddings
        embeddings_path = self.models_dir / "concept_embeddings.npz"
        self.concept_embeddings.save(str(embeddings_path))

        # Save knowledge
        knowledge_path = self.models_dir / "knowledge_base.json"
        with open(knowledge_path, 'w') as f:
            json.dump({
                'concept_definitions': self.concept_definitions,
                'trading_rules': self.trading_rules,
                'n_transcripts_processed': self.n_transcripts_processed,
                'last_training_time': self.last_training_time,
                'training_history': self.training_history[-10:],  # Last 10
            }, f, indent=2, default=str)

        logger.info(f"Knowledge base saved to {self.models_dir}")

    def load(self):
        """Load the knowledge base"""
        logger.info("Loading knowledge base...")

        # Load classifier
        try:
            self.concept_classifier.load()
        except:
            logger.warning("Could not load classifier")

        # Load embeddings
        try:
            embeddings_path = self.models_dir / "concept_embeddings.npz"
            self.concept_embeddings.load(str(embeddings_path))
        except:
            logger.warning("Could not load embeddings")

        # Load knowledge
        try:
            knowledge_path = self.models_dir / "knowledge_base.json"
            with open(knowledge_path) as f:
                data = json.load(f)
                self.concept_definitions = data.get('concept_definitions', {})
                self.trading_rules = data.get('trading_rules', [])
                self.n_transcripts_processed = data.get('n_transcripts_processed', 0)
                self.last_training_time = data.get('last_training_time')
                self.training_history = data.get('training_history', [])
        except:
            logger.warning("Could not load knowledge base")

        return self


class SignalPredictionModel:
    """
    Predicts trading signals based on learned ICT concepts.
    Combines concept presence with market context for predictions.
    """

    def __init__(self, knowledge_base: ICTKnowledgeBase):
        self.kb = knowledge_base
        self.models_dir = self.kb.models_dir

        # Ensemble of regressors for signal strength prediction
        self.signal_model = None
        self.is_fitted = False

    def prepare_signal_features(self, concept_vector: np.ndarray, market_features: Dict) -> np.ndarray:
        """Combine concept and market features"""
        market_array = np.array([
            market_features.get('trend_strength', 0),
            market_features.get('volatility', 0),
            market_features.get('volume_ratio', 1),
            market_features.get('distance_to_key_level', 0),
            market_features.get('session_score', 0),  # Kill zone relevance
            market_features.get('mtf_alignment', 0),  # Multi-timeframe alignment
        ])

        return np.concatenate([concept_vector, market_array])

    def train(self, training_data: List[Dict]):
        """
        Train signal prediction model.

        training_data format:
        [
            {
                'concept_vector': [...],  # From classifier
                'market_features': {...},
                'actual_outcome': 1/-1/0,  # Win/Loss/Neutral
                'profit_factor': 1.5,  # Actual R:R achieved
            },
            ...
        ]
        """
        if not training_data:
            logger.warning("No training data for signal model")
            return

        X = []
        y = []

        for sample in training_data:
            features = self.prepare_signal_features(
                np.array(sample['concept_vector']),
                sample['market_features']
            )
            X.append(features)
            y.append(sample.get('profit_factor', 0))

        X = np.array(X)
        y = np.array(y)

        # Train ensemble
        from sklearn.ensemble import VotingRegressor

        self.signal_model = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)),
            ('ridge', Ridge(alpha=1.0)),
        ])

        self.signal_model.fit(X, y)
        self.is_fitted = True

        # Evaluate
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.signal_model, X, y, cv=min(5, len(X)))
        logger.info(f"Signal model trained. CV R2: {scores.mean():.3f}")

    def predict_signal_strength(self, concept_vector: np.ndarray, market_features: Dict) -> Dict:
        """Predict signal strength and confidence"""
        if not self.is_fitted:
            # Return rule-based prediction if model not trained
            return self._rule_based_prediction(concept_vector, market_features)

        features = self.prepare_signal_features(concept_vector, market_features)
        prediction = self.signal_model.predict(features.reshape(1, -1))[0]

        return {
            'signal_strength': float(prediction),
            'direction': 'bullish' if prediction > 0 else 'bearish',
            'confidence': min(abs(prediction) / 2, 1.0),  # Normalize to 0-1
            'method': 'ml_model'
        }

    def _rule_based_prediction(self, concept_vector: np.ndarray, market_features: Dict) -> Dict:
        """Fallback rule-based prediction"""
        # Sum concept weights
        concept_score = np.sum(concept_vector)

        # Adjust for market context
        trend = market_features.get('trend_strength', 0)
        session = market_features.get('session_score', 0.5)

        signal = concept_score * 0.5 + trend * 0.3 + session * 0.2

        return {
            'signal_strength': float(signal),
            'direction': 'bullish' if signal > 0 else 'bearish',
            'confidence': min(abs(signal), 1.0),
            'method': 'rule_based'
        }


def run_training_pipeline(data_dir: str = None, incremental: bool = False):
    """Run the full training pipeline"""
    print("=" * 60)
    print("ICT TRADING ML TRAINING PIPELINE")
    print("=" * 60)

    kb = ICTKnowledgeBase(data_dir)

    # Load existing if incremental
    if incremental:
        try:
            kb.load()
            print("Loaded existing knowledge base for incremental training")
        except:
            print("No existing knowledge base, starting fresh")

    # Train
    results = kb.train(incremental=incremental)

    # Save
    kb.save()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Transcripts processed: {results['n_transcripts']}")
    print(f"Classifier F1: {results['components']['classifier'].get('ensemble_f1', 'N/A'):.3f}")
    print(f"Concepts defined: {results['components']['definitions']['n_concepts']}")
    print(f"Rules extracted: {results['components']['rules']['n_rules']}")
    print("=" * 60)

    # Show learning progress
    progress = kb.get_learning_progress()
    print("\nLEARNING PROGRESS:")
    print(f"  Total transcripts: {progress['n_transcripts_processed']}")
    print(f"  Training runs: {progress['n_training_runs']}")
    print(f"  Classifier trend: {progress['classifier_performance']['trend']}")

    return kb


if __name__ == "__main__":
    kb = run_training_pipeline()
