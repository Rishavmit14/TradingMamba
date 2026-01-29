"""
Feature Extractor for Smart Money Concepts

Converts raw transcript text into numerical features for ML models.
Uses TF-IDF, n-grams, and custom Smart Money concept embeddings.

100% FREE - Uses scikit-learn and numpy only.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn joblib")


# Smart Money specific vocabulary for feature extraction
SMART_MONEY_VOCABULARY = {
    # Market Structure
    'market_structure': ['bos', 'break of structure', 'choch', 'change of character',
                         'higher high', 'higher low', 'lower high', 'lower low',
                         'hh', 'hl', 'lh', 'll', 'trend', 'structure'],

    # Order Blocks
    'order_block': ['order block', 'orderblock', 'ob', 'bullish ob', 'bearish ob',
                    'mitigation', 'mitigated', 'unmitigated'],

    # Fair Value Gaps
    'fair_value_gap': ['fair value gap', 'fvg', 'imbalance', 'inefficiency',
                       'balanced price range', 'bpr'],

    # Liquidity
    'liquidity': ['liquidity', 'buy side liquidity', 'sell side liquidity',
                  'bsl', 'ssl', 'liquidity sweep', 'liquidity grab',
                  'stop hunt', 'equal highs', 'equal lows', 'eqh', 'eql'],

    # Premium/Discount
    'premium_discount': ['premium', 'discount', 'equilibrium', 'eq',
                         'premium array', 'discount array', '50%', 'fifty percent'],

    # Kill Zones
    'kill_zones': ['kill zone', 'killzone', 'asian session', 'london session',
                   'new york session', 'london open', 'ny open', 'asian range'],

    # Entry Models
    'entry_models': ['optimal trade entry', 'ote', 'silver bullet',
                     'power of three', 'po3', 'accumulation', 'manipulation',
                     'distribution', 'amd'],

    # Institutional
    'institutional': ['smart money', 'institutional', 'market maker',
                      'banks', 'hedge funds', 'retail traders'],

    # Breaker/Mitigation
    'breaker': ['breaker', 'breaker block', 'mitigation block',
                'rejection block', 'propulsion block'],

    # Time-based
    'time_based': ['macro', 'micro', 'quarterly shift', 'monthly',
                   'weekly', 'daily', 'hourly', 'time and price'],

    # Price Action
    'price_action': ['candlestick', 'engulfing', 'pin bar', 'doji',
                     'hammer', 'shooting star', 'wick', 'body'],
}


# =============================================================================
# ICT SENTIMENT/TONE VOCABULARY - For detecting HOW ICT judges pattern quality
# =============================================================================
# This captures the QUALITATIVE language ICT uses to describe patterns
# Not just WHAT patterns exist, but HOW GOOD they are according to ICT

ICT_SENTIMENT_VOCABULARY = {
    # POSITIVE QUALITY INDICATORS - ICT approves of this setup
    'high_quality': [
        'beautiful', 'gorgeous', 'pristine', 'textbook', 'perfect',
        'clean', 'clear', 'obvious', 'crisp', 'sharp',
        'ideal', 'optimal', 'best', 'excellent', 'fantastic',
        'love this', 'i like this', 'this is what you want',
        'high probability', 'high prob', 'a plus', 'a+',
        'this is money', 'easy money', 'free money',
        'no brainer', 'slam dunk', 'gift',
    ],

    # NEGATIVE QUALITY INDICATORS - ICT disapproves of this setup
    'low_quality': [
        'ugly', 'messy', 'sloppy', 'weak', 'poor',
        'unclear', 'muddy', 'choppy', 'rough',
        'avoid', 'skip', 'pass on', 'stay away',
        'not clean', 'not clear', 'not ideal',
        'low probability', 'low prob', 'risky',
        'questionable', 'sketchy', 'marginal',
        'dont take this', "don't take this", 'wouldnt trade',
        "wouldn't trade", 'not worth it',
    ],

    # CONFIDENCE/CERTAINTY - How sure ICT is
    'high_confidence': [
        'always', 'every time', 'without fail', 'guaranteed',
        'definitely', 'absolutely', 'certainly', 'for sure',
        'this works', 'this is reliable', 'consistent',
        'you will see', 'watch this', 'mark my words',
        'i promise', 'trust me', 'believe me',
    ],

    'low_confidence': [
        'sometimes', 'maybe', 'might', 'could be',
        'not always', 'occasionally', 'depends',
        'if and only if', 'only when', 'be careful',
        'watch out', 'be aware', 'heads up',
    ],

    # EMPHASIS/IMPORTANCE - How critical ICT considers this
    'critical_importance': [
        'key', 'crucial', 'critical', 'essential', 'vital',
        'most important', 'pay attention', 'listen carefully',
        'this is big', 'this matters', 'remember this',
        'write this down', 'take note', 'underline this',
        'this is the secret', 'holy grail', 'game changer',
        'edge', 'your edge', 'the edge',
    ],

    # INSTITUTIONAL QUALITY - When ICT says it's "bank level"
    'institutional_quality': [
        'institutional', 'bank level', 'hedge fund',
        'smart money', 'big boys', 'whales',
        'professional', 'pro level', 'elite',
        'this is how banks', 'this is what they do',
        'real trading', 'the real game',
    ],

    # TIMING QUALITY - When ICT approves of entry timing
    'good_timing': [
        'right on time', 'perfect timing', 'exactly when',
        'kill zone', 'sweet spot', 'golden hour',
        'this is when', 'this is the time',
        'now is the time', 'optimal time',
    ],

    # PATTERN FRESHNESS - ICT's view on pattern age
    'fresh_pattern': [
        'fresh', 'new', 'untested', 'virgin',
        'first touch', 'first test', 'never touched',
        'unmitigated', 'still valid', 'still there',
    ],

    'stale_pattern': [
        'old', 'stale', 'used up', 'expired',
        'already tested', 'already touched', 'mitigated',
        'been there', 'already hit', 'used',
    ],

    # RISK ASSESSMENT - ICT's risk language
    'low_risk': [
        'low risk', 'safe', 'secure', 'protected',
        'tight stop', 'small stop', 'defined risk',
        'controlled', 'manageable',
    ],

    'high_risk': [
        'high risk', 'risky', 'dangerous', 'exposed',
        'wide stop', 'big stop', 'large risk',
        'be careful', 'watch out', 'protect yourself',
    ],
}


# =============================================================================
# ICT SENTIMENT ANALYZER - Detects HOW ICT judges pattern quality
# =============================================================================

class ICTSentimentAnalyzer:
    """
    Analyzes ICT's teaching TONE and SENTIMENT to understand pattern quality.

    This is what makes the ML a "Genius Student" - understanding not just
    WHAT patterns exist, but HOW GOOD they are according to ICT's teaching.

    Example:
    - "This is a beautiful order block" → HIGH QUALITY signal
    - "This FVG is messy, I wouldn't trade it" → LOW QUALITY signal
    - "Pay attention, this is the key" → CRITICAL IMPORTANCE signal

    100% FREE - Uses regex pattern matching only.
    """

    def __init__(self):
        self.sentiment_scores = {}
        self.pattern_quality_cache = {}

    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze overall sentiment/tone of ICT's teaching in text.

        Returns:
            Dictionary with sentiment scores for each category
        """
        text_lower = text.lower()
        word_count = len(text_lower.split())

        results = {
            'sentiment_scores': {},
            'dominant_sentiment': None,
            'quality_signal': 'NEUTRAL',  # HIGH, NEUTRAL, LOW
            'confidence_signal': 'NEUTRAL',  # HIGH, NEUTRAL, LOW
            'importance_score': 0.0,
            'risk_signal': 'NEUTRAL',  # LOW_RISK, NEUTRAL, HIGH_RISK
        }

        # Count matches for each sentiment category
        for category, terms in ICT_SENTIMENT_VOCABULARY.items():
            count = 0
            matched_terms = []
            for term in terms:
                matches = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                if matches > 0:
                    count += matches
                    matched_terms.append(term)

            results['sentiment_scores'][category] = {
                'count': count,
                'density': count / max(word_count, 1) * 1000,
                'matched_terms': matched_terms,
            }

        # Determine dominant sentiment
        max_count = 0
        for category, data in results['sentiment_scores'].items():
            if data['count'] > max_count:
                max_count = data['count']
                results['dominant_sentiment'] = category

        # Calculate quality signal
        high_q = results['sentiment_scores']['high_quality']['count']
        low_q = results['sentiment_scores']['low_quality']['count']
        inst_q = results['sentiment_scores']['institutional_quality']['count']

        quality_score = high_q + inst_q - (low_q * 1.5)  # Negative weighted more
        if quality_score > 2:
            results['quality_signal'] = 'HIGH'
        elif quality_score < -1:
            results['quality_signal'] = 'LOW'
        else:
            results['quality_signal'] = 'NEUTRAL'

        # Calculate confidence signal
        high_c = results['sentiment_scores']['high_confidence']['count']
        low_c = results['sentiment_scores']['low_confidence']['count']

        if high_c > low_c + 1:
            results['confidence_signal'] = 'HIGH'
        elif low_c > high_c + 1:
            results['confidence_signal'] = 'LOW'

        # Calculate importance score (0-1)
        importance = results['sentiment_scores']['critical_importance']['count']
        results['importance_score'] = min(importance / 3, 1.0)

        # Calculate risk signal
        low_r = results['sentiment_scores']['low_risk']['count']
        high_r = results['sentiment_scores']['high_risk']['count']

        if low_r > high_r:
            results['risk_signal'] = 'LOW_RISK'
        elif high_r > low_r:
            results['risk_signal'] = 'HIGH_RISK'

        return results

    def extract_pattern_quality_context(self, text: str, pattern_type: str) -> Dict[str, any]:
        """
        Extract sentiment specifically around a pattern type mention.

        This captures ICT's opinion ABOUT a specific pattern type.

        Example:
        - Text: "This order block is beautiful, look at the clean wick"
        - Pattern: "order_block"
        - Returns: {quality: HIGH, reasoning: ["beautiful", "clean"]}
        """
        text_lower = text.lower()

        # Find pattern mentions
        pattern_terms = SMART_MONEY_VOCABULARY.get(pattern_type, [])
        if not pattern_terms:
            return {'quality': 'UNKNOWN', 'context': None, 'reasoning': []}

        # Window around pattern mention (words before and after)
        window_size = 20  # words

        quality_indicators = []
        negative_indicators = []

        words = text_lower.split()

        for i, word in enumerate(words):
            # Check if this is a pattern mention
            context = ' '.join(words[max(0, i-3):i+4])
            is_pattern_mention = any(term in context for term in pattern_terms)

            if is_pattern_mention:
                # Get window around pattern mention
                start = max(0, i - window_size)
                end = min(len(words), i + window_size)
                window_text = ' '.join(words[start:end])

                # Check for positive indicators
                for term in ICT_SENTIMENT_VOCABULARY['high_quality']:
                    if term in window_text:
                        quality_indicators.append(term)

                for term in ICT_SENTIMENT_VOCABULARY['institutional_quality']:
                    if term in window_text:
                        quality_indicators.append(f"institutional: {term}")

                # Check for negative indicators
                for term in ICT_SENTIMENT_VOCABULARY['low_quality']:
                    if term in window_text:
                        negative_indicators.append(term)

        # Determine quality
        if len(quality_indicators) > len(negative_indicators):
            quality = 'HIGH'
        elif len(negative_indicators) > len(quality_indicators):
            quality = 'LOW'
        else:
            quality = 'NEUTRAL'

        return {
            'quality': quality,
            'positive_signals': list(set(quality_indicators)),
            'negative_signals': list(set(negative_indicators)),
            'reasoning': quality_indicators + negative_indicators,
        }

    def get_teaching_emphasis(self, text: str) -> List[Dict[str, any]]:
        """
        Find moments where ICT is emphasizing something important.

        Returns list of emphasized concepts with their importance context.
        """
        text_lower = text.lower()
        emphasized = []

        # Find emphasis phrases
        emphasis_patterns = [
            r'pay attention[^.]*',
            r'listen carefully[^.]*',
            r'this is (?:the )?key[^.]*',
            r'remember this[^.]*',
            r'write this down[^.]*',
            r'this is important[^.]*',
            r'this is what you want[^.]*',
            r'this is how [^.]*',
        ]

        for pattern in emphasis_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Extract what concept is being emphasized
                concept = None
                for concept_name, terms in SMART_MONEY_VOCABULARY.items():
                    for term in terms:
                        if term in match:
                            concept = concept_name
                            break
                    if concept:
                        break

                emphasized.append({
                    'text': match,
                    'concept': concept,
                    'importance': 'HIGH',
                })

        return emphasized

    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract numerical sentiment features for ML training.

        Returns features that can be combined with other ML features.
        """
        sentiment = self.analyze_sentiment(text)

        features = {}

        # Raw counts as features
        for category, data in sentiment['sentiment_scores'].items():
            features[f'sentiment_{category}_count'] = data['count']
            features[f'sentiment_{category}_density'] = data['density']

        # Derived signals as numerical features
        features['sentiment_quality_score'] = {
            'HIGH': 1.0, 'NEUTRAL': 0.0, 'LOW': -1.0
        }[sentiment['quality_signal']]

        features['sentiment_confidence_score'] = {
            'HIGH': 1.0, 'NEUTRAL': 0.0, 'LOW': -1.0
        }[sentiment['confidence_signal']]

        features['sentiment_importance'] = sentiment['importance_score']

        features['sentiment_risk_score'] = {
            'LOW_RISK': 1.0, 'NEUTRAL': 0.0, 'HIGH_RISK': -1.0
        }[sentiment['risk_signal']]

        return features


class SmartMoneyFeatureExtractor:
    """
    Extracts ML-ready features from Smart Money trading transcripts.

    Features:
    1. TF-IDF vectors for general text
    2. Smart Money concept counts and densities
    3. Concept co-occurrence features
    4. Temporal features (sequence position)
    5. Custom Smart Money embeddings
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.models_dir = self.data_dir / "ml_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )

        self.concept_vectorizer = CountVectorizer(
            vocabulary=self._build_ict_vocabulary(),
            ngram_range=(1, 3),
            lowercase=True
        )

        self.scaler = StandardScaler()
        self.is_fitted = False

        # NEW: Sentiment analyzer for ICT's teaching tone
        self.sentiment_analyzer = ICTSentimentAnalyzer()

    def _build_ict_vocabulary(self) -> List[str]:
        """Build flat vocabulary list from Smart Money concepts"""
        vocab = []
        for terms in SMART_MONEY_VOCABULARY.values():
            vocab.extend(terms)
        return list(set(vocab))

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess transcript text"""
        # Lowercase
        text = text.lower()

        # Remove timestamps and common filler words
        text = re.sub(r'\d+:\d+:\d+', '', text)
        text = re.sub(r'\[.*?\]', '', text)

        # Normalize Smart Money terms
        replacements = {
            r'\border\s*block\b': 'orderblock',
            r'\bfair\s*value\s*gap\b': 'fairvaluegap',
            r'\bchange\s*of\s*character\b': 'choch',
            r'\bbreak\s*of\s*structure\b': 'bos',
            r'\bkill\s*zone\b': 'killzone',
            r'\bpower\s*of\s*three\b': 'powerofthree',
            r'\boptimal\s*trade\s*entry\b': 'ote',
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extract_concept_features(self, text: str) -> Dict[str, float]:
        """Extract Smart Money concept-specific features"""
        text_lower = text.lower()
        word_count = len(text_lower.split())

        features = {}

        for concept_name, terms in SMART_MONEY_VOCABULARY.items():
            count = 0
            for term in terms:
                count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))

            # Raw count
            features[f'{concept_name}_count'] = count
            # Density (normalized by text length)
            features[f'{concept_name}_density'] = count / max(word_count, 1) * 1000

        return features

    def extract_cooccurrence_features(self, text: str, window_size: int = 50) -> Dict[str, int]:
        """Extract concept co-occurrence features within text windows"""
        words = text.lower().split()
        cooccurrences = defaultdict(int)

        concept_positions = defaultdict(list)

        # Find positions of each concept
        for i, word in enumerate(words):
            for concept_name, terms in SMART_MONEY_VOCABULARY.items():
                for term in terms:
                    if term in ' '.join(words[max(0, i-2):i+3]):
                        concept_positions[concept_name].append(i)
                        break

        # Count co-occurrences within window
        concept_names = list(SMART_MONEY_VOCABULARY.keys())
        for i, c1 in enumerate(concept_names):
            for c2 in concept_names[i+1:]:
                count = 0
                for pos1 in concept_positions[c1]:
                    for pos2 in concept_positions[c2]:
                        if abs(pos1 - pos2) <= window_size:
                            count += 1
                cooccurrences[f'{c1}_{c2}_cooc'] = count

        return dict(cooccurrences)

    def extract_segment_features(self, segments: List[Dict]) -> Dict[str, float]:
        """Extract temporal features from transcript segments"""
        if not segments:
            return {
                'total_duration': 0,
                'segment_count': 0,
                'avg_segment_length': 0,
                'concept_intro_time': 0,  # When first concept appears
                'concept_intro_ratio': 0,  # Added to match non-empty case
            }

        total_duration = segments[-1].get('end_time', 0) if segments else 0
        segment_count = len(segments)

        # Find when first Smart Money concept appears
        concept_intro_time = total_duration  # Default to end
        for seg in segments:
            text = seg.get('text', '').lower()
            for terms in SMART_MONEY_VOCABULARY.values():
                for term in terms:
                    if term in text:
                        concept_intro_time = min(concept_intro_time, seg.get('start_time', 0))
                        break

        return {
            'total_duration': total_duration,
            'segment_count': segment_count,
            'avg_segment_length': total_duration / max(segment_count, 1),
            'concept_intro_time': concept_intro_time,
            'concept_intro_ratio': concept_intro_time / max(total_duration, 1),
        }

    def fit(self, transcripts: List[Dict]) -> 'SmartMoneyFeatureExtractor':
        """Fit the feature extractor on training transcripts"""
        texts = [self.preprocess_text(t.get('full_text', '')) for t in transcripts]

        # Adjust TF-IDF parameters for small document sets (1-2 docs)
        n_docs = len(texts)
        if n_docs < 3:
            self.tfidf_vectorizer.set_params(min_df=1, max_df=1.0)

        # Fit TF-IDF
        self.tfidf_vectorizer.fit(texts)

        # Fit concept vectorizer (already has fixed vocabulary)
        self.concept_vectorizer.fit(texts)

        # Extract all numerical features for scaler
        all_features = []
        for transcript in transcripts:
            features = self._extract_numerical_features(transcript)
            all_features.append(list(features.values()))

        if all_features:
            self.scaler.fit(all_features)

        self.is_fitted = True
        return self

    def _extract_numerical_features(self, transcript: Dict) -> Dict[str, float]:
        """Extract all numerical features from a transcript"""
        text = transcript.get('full_text', '')
        segments = transcript.get('segments', [])

        features = {}

        # Concept features
        features.update(self.extract_concept_features(text))

        # Co-occurrence features
        features.update(self.extract_cooccurrence_features(text))

        # Segment features
        features.update(self.extract_segment_features(segments))

        # NEW: Sentiment/Tone features - HOW ICT judges pattern quality
        features.update(self.sentiment_analyzer.extract_sentiment_features(text))

        # Basic stats
        features['word_count'] = transcript.get('word_count', len(text.split()))
        features['char_count'] = len(text)

        return features

    def extract_pattern_quality(self, text: str, pattern_type: str) -> Dict[str, any]:
        """
        Extract ICT's quality judgment about a specific pattern.

        This is the KEY method for understanding HOW GOOD a pattern is
        according to ICT's teaching, not just that it exists.

        Args:
            text: Transcript text
            pattern_type: Type of pattern (e.g., 'order_block', 'fair_value_gap')

        Returns:
            Quality assessment with reasoning
        """
        return self.sentiment_analyzer.extract_pattern_quality_context(text, pattern_type)

    def get_emphasized_teachings(self, text: str) -> List[Dict]:
        """
        Find moments where ICT emphasizes something important.

        These are the "pay attention" moments that matter most.
        """
        return self.sentiment_analyzer.get_teaching_emphasis(text)

    def transform(self, transcripts: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Transform transcripts into feature matrix"""
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")

        texts = [self.preprocess_text(t.get('full_text', '')) for t in transcripts]

        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()

        # Numerical features
        numerical_features = []
        for transcript in transcripts:
            features = self._extract_numerical_features(transcript)
            numerical_features.append(list(features.values()))

        numerical_features = np.array(numerical_features)
        if numerical_features.shape[0] > 0:
            numerical_features = self.scaler.transform(numerical_features)

        # Combine all features
        all_features = np.hstack([tfidf_features, numerical_features])

        # Feature names
        tfidf_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        numerical_names = list(self._extract_numerical_features(transcripts[0]).keys()) if transcripts else []
        feature_names = tfidf_names + numerical_names

        return all_features, feature_names

    def fit_transform(self, transcripts: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform in one step"""
        self.fit(transcripts)
        return self.transform(transcripts)

    def save(self, path: str = None):
        """Save the fitted feature extractor"""
        if path is None:
            path = self.models_dir / "feature_extractor.joblib"

        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'concept_vectorizer': self.concept_vectorizer,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
        }, path)
        print(f"Feature extractor saved to {path}")

    def load(self, path: str = None):
        """Load a fitted feature extractor"""
        if path is None:
            path = self.models_dir / "feature_extractor.joblib"

        data = joblib.load(path)
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.concept_vectorizer = data['concept_vectorizer']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        print(f"Feature extractor loaded from {path}")
        return self


class ConceptEmbedding:
    """
    Creates dense embeddings for Smart Money concepts.
    Uses word co-occurrence to learn concept relationships.

    100% FREE - No external APIs needed.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.concept_names = list(SMART_MONEY_VOCABULARY.keys())
        self.embeddings = None
        self.cooccurrence_matrix = None

    def fit(self, transcripts: List[Dict], window_size: int = 100):
        """Learn concept embeddings from transcripts"""
        n_concepts = len(self.concept_names)
        self.cooccurrence_matrix = np.zeros((n_concepts, n_concepts))

        for transcript in transcripts:
            text = transcript.get('full_text', '').lower()
            words = text.split()

            # Find concept occurrences
            concept_positions = {c: [] for c in self.concept_names}

            for i, word in enumerate(words):
                context = ' '.join(words[max(0, i-5):i+6])
                for j, concept in enumerate(self.concept_names):
                    for term in SMART_MONEY_VOCABULARY[concept]:
                        if term in context:
                            concept_positions[concept].append(i)
                            break

            # Build co-occurrence matrix
            for i, c1 in enumerate(self.concept_names):
                for j, c2 in enumerate(self.concept_names):
                    if i <= j:
                        count = 0
                        for p1 in concept_positions[c1]:
                            for p2 in concept_positions[c2]:
                                if abs(p1 - p2) <= window_size:
                                    count += 1
                        self.cooccurrence_matrix[i, j] = count
                        self.cooccurrence_matrix[j, i] = count

        # Apply PPMI (Positive Pointwise Mutual Information)
        total = np.sum(self.cooccurrence_matrix)
        if total > 0:
            row_sums = np.sum(self.cooccurrence_matrix, axis=1, keepdims=True)
            col_sums = np.sum(self.cooccurrence_matrix, axis=0, keepdims=True)

            with np.errstate(divide='ignore', invalid='ignore'):
                pmi = np.log2((self.cooccurrence_matrix * total) / (row_sums * col_sums + 1e-10) + 1e-10)

            ppmi = np.maximum(pmi, 0)
        else:
            ppmi = self.cooccurrence_matrix

        # SVD to get dense embeddings
        try:
            from scipy.sparse.linalg import svds
            # Use min of actual dimensions and desired embedding dim
            k = min(self.embedding_dim, min(ppmi.shape) - 1, n_concepts - 1)
            if k > 0:
                U, S, Vt = svds(ppmi.astype(float), k=k)
                self.embeddings = U * np.sqrt(S)
            else:
                self.embeddings = np.random.randn(n_concepts, self.embedding_dim) * 0.01
        except:
            # Fallback to numpy SVD
            U, S, Vt = np.linalg.svd(ppmi)
            k = min(self.embedding_dim, len(S))
            self.embeddings = U[:, :k] * np.sqrt(S[:k])

        # Pad if needed
        if self.embeddings.shape[1] < self.embedding_dim:
            padding = np.zeros((n_concepts, self.embedding_dim - self.embeddings.shape[1]))
            self.embeddings = np.hstack([self.embeddings, padding])

        return self

    def get_embedding(self, concept_name: str) -> Optional[np.ndarray]:
        """Get embedding for a concept"""
        if self.embeddings is None:
            return None
        try:
            idx = self.concept_names.index(concept_name)
            return self.embeddings[idx]
        except ValueError:
            return None

    def get_similar_concepts(self, concept_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find concepts similar to a given concept"""
        if self.embeddings is None:
            return []

        try:
            idx = self.concept_names.index(concept_name)
        except ValueError:
            return []

        query_emb = self.embeddings[idx]

        # Cosine similarity
        similarities = []
        for i, emb in enumerate(self.embeddings):
            if i != idx:
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10)
                similarities.append((self.concept_names[i], float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def save(self, path: str):
        """Save embeddings"""
        np.savez(path,
                 embeddings=self.embeddings,
                 concept_names=self.concept_names,
                 cooccurrence=self.cooccurrence_matrix)

    def load(self, path: str):
        """Load embeddings"""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.concept_names = list(data['concept_names'])
        self.cooccurrence_matrix = data['cooccurrence']
        return self


# Test function
def test_feature_extractor():
    """Test the feature extractor with sample data"""
    # Sample transcript - NOW WITH SENTIMENT/TONE examples
    sample_transcripts = [
        {
            'video_id': 'test1',
            'full_text': '''Today we're going to talk about order blocks and fair value gaps.
                           This is a beautiful order block right here, look at how clean it is.
                           Pay attention - this is the key to understanding institutional trading.
                           When price returns to an order block, we look for a reaction.
                           Fair value gaps represent imbalance in the market.
                           We want to trade from premium to discount zones.
                           The kill zone is the best time to trade - this is money.''',
            'segments': [
                {'start_time': 0, 'end_time': 30, 'text': 'order blocks'},
                {'start_time': 30, 'end_time': 60, 'text': 'fair value gaps'},
            ],
            'word_count': 80
        },
        {
            'video_id': 'test2',
            'full_text': '''Break of structure signals trend continuation.
                           This FVG is messy, I wouldn't trade this one.
                           Be careful here, this is risky.
                           Sometimes these setups work, sometimes they don't.
                           We look for liquidity sweeps before entries.
                           Smart money accumulates at discount.''',
            'segments': [
                {'start_time': 0, 'end_time': 45, 'text': 'structure'},
            ],
            'word_count': 50
        }
    ]

    print("Testing Smart Money Feature Extractor")
    print("=" * 60)

    extractor = SmartMoneyFeatureExtractor()

    # Test concept extraction
    print("\n1. Concept Features:")
    features = extractor.extract_concept_features(sample_transcripts[0]['full_text'])
    for name, value in sorted(features.items()):
        if value > 0:
            print(f"   {name}: {value}")

    # Test co-occurrence
    print("\n2. Co-occurrence Features (non-zero):")
    cooc = extractor.extract_cooccurrence_features(sample_transcripts[0]['full_text'])
    for name, value in sorted(cooc.items()):
        if value > 0:
            print(f"   {name}: {value}")

    # NEW: Test Sentiment Analysis
    print("\n3. ICT Sentiment/Tone Analysis (NEW!):")
    print("=" * 60)

    sentiment_analyzer = ICTSentimentAnalyzer()

    print("\n   Sample 1 (Positive Teaching):")
    sentiment1 = sentiment_analyzer.analyze_sentiment(sample_transcripts[0]['full_text'])
    print(f"   Quality Signal: {sentiment1['quality_signal']}")
    print(f"   Confidence Signal: {sentiment1['confidence_signal']}")
    print(f"   Importance Score: {sentiment1['importance_score']:.2f}")
    print(f"   Dominant Sentiment: {sentiment1['dominant_sentiment']}")

    print("\n   Sample 2 (Negative Teaching):")
    sentiment2 = sentiment_analyzer.analyze_sentiment(sample_transcripts[1]['full_text'])
    print(f"   Quality Signal: {sentiment2['quality_signal']}")
    print(f"   Confidence Signal: {sentiment2['confidence_signal']}")
    print(f"   Importance Score: {sentiment2['importance_score']:.2f}")
    print(f"   Dominant Sentiment: {sentiment2['dominant_sentiment']}")

    # Test pattern quality extraction
    print("\n4. Pattern Quality Context:")
    print("=" * 60)

    quality1 = extractor.extract_pattern_quality(
        sample_transcripts[0]['full_text'], 'order_block'
    )
    print(f"\n   Order Block in Sample 1:")
    print(f"   Quality: {quality1['quality']}")
    print(f"   Positive Signals: {quality1['positive_signals']}")
    print(f"   Negative Signals: {quality1['negative_signals']}")

    quality2 = extractor.extract_pattern_quality(
        sample_transcripts[1]['full_text'], 'fair_value_gap'
    )
    print(f"\n   FVG in Sample 2:")
    print(f"   Quality: {quality2['quality']}")
    print(f"   Positive Signals: {quality2['positive_signals']}")
    print(f"   Negative Signals: {quality2['negative_signals']}")

    # Test emphasis detection
    print("\n5. Teaching Emphasis Detection:")
    print("=" * 60)
    emphasis = extractor.get_emphasized_teachings(sample_transcripts[0]['full_text'])
    for item in emphasis:
        print(f"   - {item['text'][:60]}...")
        print(f"     Concept: {item['concept']}, Importance: {item['importance']}")

    # Test full pipeline
    print("\n6. Full Feature Extraction (including Sentiment):")
    X, feature_names = extractor.fit_transform(sample_transcripts)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of features: {len(feature_names)}")

    # Show sentiment features
    sentiment_features = [f for f in feature_names if 'sentiment' in f]
    print(f"   Sentiment features added: {len(sentiment_features)}")

    # Test embeddings
    print("\n7. Concept Embeddings:")
    embedder = ConceptEmbedding(embedding_dim=16)
    embedder.fit(sample_transcripts)

    print("   Similar concepts to 'order_block':")
    similar = embedder.get_similar_concepts('order_block', top_k=3)
    for concept, sim in similar:
        print(f"      {concept}: {sim:.3f}")

    print("\n" + "=" * 60)
    print("Feature extraction test complete!")
    print("\n🎯 THE ML NOW SENSES ICT's TONE, NOT JUST KEYWORDS!")
    print("   - 'beautiful' → HIGH QUALITY signal")
    print("   - 'messy' → LOW QUALITY signal")
    print("   - 'pay attention' → HIGH IMPORTANCE signal")


if __name__ == "__main__":
    test_feature_extractor()
