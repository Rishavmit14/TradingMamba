"""
Smart Money Trading ML Training Pipeline

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

from .feature_extractor import SmartMoneyFeatureExtractor, ConceptEmbedding, SMART_MONEY_VOCABULARY
from .concept_classifier import SmartMoneyConceptClassifier, ConceptSequenceAnalyzer
from .training_database import TrainingDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional vision imports - only used if vision training is enabled
try:
    from .video_vision_analyzer import VideoVisionTrainer, VideoVisualSummary
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.info("Vision training not available (missing dependencies)")


class SmartMoneyKnowledgeBase:
    """
    Central knowledge base that learns from all transcripts.
    Combines multiple learning approaches for comprehensive Smart Money understanding.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.concepts_dir = self.data_dir / "concepts"
        self.models_dir = self.data_dir / "ml_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.feature_extractor = SmartMoneyFeatureExtractor(str(self.data_dir))
        self.concept_classifier = SmartMoneyConceptClassifier(str(self.data_dir))
        self.concept_embeddings = ConceptEmbedding(embedding_dim=64)
        self.sequence_analyzer = ConceptSequenceAnalyzer()

        # Training database for tracking all video learnings
        self.training_db = TrainingDatabase(str(self.data_dir))

        # Vision trainer for multimodal learning (optional)
        self.vision_trainer = None
        self.vision_knowledge = {}  # Visual patterns learned from videos

        # Knowledge storage
        self.concept_definitions = {}  # Learned definitions
        self.concept_relationships = {}  # How concepts relate
        self.concept_contexts = defaultdict(list)  # Example contexts
        self.trading_rules = []  # Extracted trading rules

        # Metadata
        self.n_transcripts_processed = 0
        self.last_training_time = None
        self.training_history = []
        self.trained_video_ids = []  # Track which videos have been trained

        # Load playlist metadata for video tracking
        self.playlist_meta = self._load_playlist_metadata()

    def _load_playlist_metadata(self) -> Dict:
        """Load playlist metadata to map videos to playlists"""
        playlists_dir = self.data_dir / "playlists"
        playlist_meta = {}

        if not playlists_dir.exists():
            return playlist_meta

        for pfile in playlists_dir.glob("*.json"):
            try:
                with open(pfile) as f:
                    pdata = json.load(f)
                    pid = pdata.get('playlist_id', '')
                    playlist_meta[pid] = pdata

                    # Map each video to its playlist
                    for video in pdata.get('videos', []):
                        vid = video.get('video_id', '')
                        playlist_meta[f"video:{vid}"] = {
                            'playlist_id': pid,
                            'playlist_title': pdata.get('title', ''),
                            'video_title': video.get('title', ''),
                            'tier': pdata.get('tier', 1),
                        }
            except Exception as e:
                logger.warning(f"Error loading playlist {pfile}: {e}")

        return playlist_meta

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
        """Extract concept definitions from transcripts (optimized)"""
        import re
        definitions = defaultdict(lambda: {'examples': [], 'contexts': [], 'frequency': 0})

        # Pre-compile patterns for efficiency
        compiled_patterns = {}
        for concept, terms in SMART_MONEY_VOCABULARY.items():
            # Use simpler word boundary matching
            term_pattern = '|'.join(re.escape(t) for t in terms)
            compiled_patterns[concept] = re.compile(rf'\b({term_pattern})\b', re.IGNORECASE)

        for transcript in transcripts:
            text = transcript.get('full_text', '')
            if not text:
                continue

            # Split into sentences once
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10][:200]  # Limit sentences

            for concept, pattern in compiled_patterns.items():
                for sentence in sentences:
                    if pattern.search(sentence):
                        if len(definitions[concept]['examples']) < 10:  # Limit examples
                            definitions[concept]['examples'].append(sentence[:300])
                        definitions[concept]['frequency'] += 1

        self.concept_definitions = dict(definitions)
        return self.concept_definitions

    def extract_trading_rules(self, transcripts: List[Dict]) -> List[Dict]:
        """Extract trading rules mentioned in transcripts (optimized)"""
        import re

        rules = []
        # Pre-compile rule patterns
        rule_patterns = [
            (re.compile(r"(?:when|if)\s+([^,]{10,80}),?\s*(?:then|we|you)\s+([^.]{10,80})", re.IGNORECASE), "conditional"),
            (re.compile(r"(?:always|never|must)\s+([^.]{10,80})", re.IGNORECASE), "imperative"),
            (re.compile(r"(?:look for|wait for)\s+([^.]{10,80})", re.IGNORECASE), "setup"),
        ]

        # Pre-compile concept patterns
        concept_patterns = {}
        for concept, terms in SMART_MONEY_VOCABULARY.items():
            term_pattern = '|'.join(re.escape(t) for t in terms)
            concept_patterns[concept] = re.compile(rf'\b({term_pattern})\b', re.IGNORECASE)

        for transcript in transcripts:
            text = transcript.get('full_text', '')
            if not text:
                continue

            text_lower = text.lower()
            video_id = transcript.get('video_id', '')

            for pattern, rule_type in rule_patterns:
                matches = pattern.findall(text_lower)[:5]  # Limit matches
                for match in matches:
                    if isinstance(match, tuple):
                        rule_text = ' -> '.join(match)
                    else:
                        rule_text = match

                    # Check which concepts are mentioned
                    mentioned = [c for c, p in concept_patterns.items() if p.search(rule_text)]

                    if mentioned:
                        rules.append({
                            'text': rule_text.strip()[:200],
                            'type': rule_type,
                            'concepts': mentioned,
                            'source_video': video_id,
                        })

        # Deduplicate
        seen = set()
        unique_rules = []
        for rule in rules:
            key = rule['text'][:40]
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)

        self.trading_rules = unique_rules[:100]  # Limit total rules
        logger.info(f"Extracted {len(self.trading_rules)} trading rules")
        return self.trading_rules

    def train(self, transcripts: List[Dict] = None, incremental: bool = False) -> Dict:
        """Train all ML components and record per-video learnings"""
        logger.info("Starting training pipeline...")

        if transcripts is None:
            transcripts = self.load_transcripts()

        if not transcripts:
            logger.warning("No transcripts available for training")
            return {'status': 'no_data'}

        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_transcripts': len(transcripts),
            'components': {},
            'video_learnings': []
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

        # 6. Record per-video learnings to training database
        logger.info("Recording per-video learnings to database...")
        self._record_video_learnings(transcripts, rules)
        results['components']['database'] = {
            'videos_recorded': len(transcripts),
            'database_path': str(self.training_db.db_path)
        }

        # Update metadata
        self.n_transcripts_processed = len(transcripts)
        self.last_training_time = datetime.utcnow().isoformat()
        self.training_history.append(results)

        logger.info(f"Training complete. Processed {len(transcripts)} transcripts.")
        return results

    def train_with_vision(
        self,
        transcripts: List[Dict] = None,
        vision_provider: str = "local",
        max_frames_per_video: int = 0,
        extraction_mode: str = "comprehensive",
        progress_callback=None
    ) -> Dict:
        """
        Enhanced training that includes visual analysis of video frames.
        This enables the ML to understand visual patterns shown in tutorials.

        Args:
            transcripts: List of transcript dicts (loads all if None)
            vision_provider: "local" (FREE on M1/M2/M3 Mac), "anthropic", or "openai"
            max_frames_per_video: Max frames to analyze per video (0 = no limit)
            extraction_mode: How thoroughly to analyze:
                - "comprehensive": Every 3s like a dedicated student (DEFAULT)
                - "thorough": Every 5s with keyword boosting
                - "balanced": Every 10-15s with keyword boosting
                - "selective": Only at demonstrative moments
            progress_callback: Optional callback(current, total, message)

        Returns:
            Training results including vision analysis
        """
        if not VISION_AVAILABLE:
            logger.warning("Vision training not available. Falling back to text-only training.")
            return self.train(transcripts)

        logger.info(f"Starting multimodal training pipeline (text + vision) - {extraction_mode} mode...")

        if transcripts is None:
            transcripts = self.load_transcripts()

        if not transcripts:
            logger.warning("No transcripts available for training")
            return {'status': 'no_data'}

        # Initialize vision trainer
        self.vision_trainer = VideoVisionTrainer(
            str(self.data_dir),
            vision_provider=vision_provider
        )

        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_transcripts': len(transcripts),
            'training_mode': 'multimodal',
            'extraction_mode': extraction_mode,
            'components': {},
            'vision_analysis': {},
            'video_learnings': []
        }

        total_steps = len(transcripts) + 5  # +5 for other training steps
        current_step = 0

        # 1. Process videos with vision analysis
        logger.info(f"Starting vision analysis of video frames ({extraction_mode} mode)...")
        vision_summaries = []

        for i, transcript in enumerate(transcripts):
            video_id = transcript.get('video_id', '')
            if not video_id:
                continue

            if progress_callback:
                progress_callback(
                    current_step,
                    total_steps,
                    f"Analyzing visual content: {transcript.get('title', video_id)[:40]}..."
                )

            try:
                summary = self.vision_trainer.process_video(
                    video_id,
                    max_frames=max_frames_per_video,
                    extraction_mode=extraction_mode
                )
                if summary:
                    vision_summaries.append(summary)

                    # Enhance transcript with visual data
                    transcripts[i] = self.vision_trainer.enhance_transcript_with_vision(
                        video_id,
                        transcript
                    )
            except Exception as e:
                logger.warning(f"Vision analysis failed for {video_id}: {e}")

            current_step += 1

        results['vision_analysis'] = {
            'videos_analyzed': len(vision_summaries),
            'total_frames_analyzed': sum(s.total_frames_analyzed for s in vision_summaries),
            'chart_frames': sum(s.chart_frames for s in vision_summaries),
        }

        # Aggregate visual knowledge
        self.vision_knowledge = self.vision_trainer.get_visual_knowledge()
        results['vision_analysis']['visual_patterns'] = self.vision_knowledge.get('pattern_frequency', {})
        results['vision_analysis']['visual_concepts'] = len(self.vision_knowledge.get('visual_concepts', []))

        if progress_callback:
            progress_callback(current_step, total_steps, "Training concept classifier...")

        # 2. Train concept classifier (now with enhanced transcripts)
        logger.info("Training concept classifier with vision-enhanced data...")
        clf_result = self.concept_classifier.fit(transcripts)
        results['components']['classifier'] = clf_result
        current_step += 1

        if progress_callback:
            progress_callback(current_step, total_steps, "Building concept embeddings...")

        # 3. Build concept embeddings
        logger.info("Building concept embeddings...")
        self.concept_embeddings.fit(transcripts)
        results['components']['embeddings'] = {'status': 'trained', 'dim': self.concept_embeddings.embedding_dim}
        current_step += 1

        if progress_callback:
            progress_callback(current_step, total_steps, "Analyzing concept sequences...")

        # 4. Build sequence analyzer
        logger.info("Analyzing concept sequences...")
        self.sequence_analyzer.build_transition_matrix(transcripts)
        results['components']['sequence_analyzer'] = {'status': 'trained'}
        current_step += 1

        if progress_callback:
            progress_callback(current_step, total_steps, "Extracting concept definitions...")

        # 5. Extract concept definitions
        logger.info("Extracting concept definitions...")
        definitions = self.extract_concept_definitions(transcripts)
        results['components']['definitions'] = {'n_concepts': len(definitions)}

        # 6. Extract trading rules
        logger.info("Extracting trading rules...")
        rules = self.extract_trading_rules(transcripts)
        results['components']['rules'] = {'n_rules': len(rules)}
        current_step += 1

        if progress_callback:
            progress_callback(current_step, total_steps, "Recording to training database...")

        # 7. Record per-video learnings
        logger.info("Recording per-video learnings to database...")
        self._record_video_learnings(transcripts, rules)
        results['components']['database'] = {
            'videos_recorded': len(transcripts),
            'database_path': str(self.training_db.db_path)
        }

        # Update metadata
        self.n_transcripts_processed = len(transcripts)
        self.last_training_time = datetime.utcnow().isoformat()
        self.training_history.append(results)

        if progress_callback:
            progress_callback(total_steps, total_steps, "Multimodal training complete!")

        logger.info(f"Multimodal training complete. Processed {len(transcripts)} transcripts with vision analysis.")
        return results

    def get_visual_pattern_examples(self, pattern_type: str) -> List[Dict]:
        """
        Get visual examples of a specific pattern type from analyzed videos.

        Args:
            pattern_type: e.g., "FVG", "Order Block", "Breaker"

        Returns:
            List of examples with timestamps and frame paths
        """
        if not self.vision_knowledge:
            return []

        patterns_by_type = self.vision_knowledge.get('patterns_by_type', {})
        return patterns_by_type.get(pattern_type, [])

    def get_teaching_moments(self, concept: str = None) -> List[Dict]:
        """
        Get key teaching moments from videos, optionally filtered by concept.
        """
        if not self.vision_knowledge:
            return []

        moments = self.vision_knowledge.get('key_teaching_moments', [])

        if concept:
            concept_lower = concept.lower()
            moments = [
                m for m in moments
                if concept_lower in m.get('teaching_point', '').lower()
                or any(concept_lower in p.get('type', '').lower() for p in m.get('patterns', []))
            ]

        return moments

    def train_from_videos(self, video_ids: List[str], incremental: bool = False) -> Dict:
        """
        Train only from specific video transcripts.
        Used for selective playlist-based training.
        """
        logger.info(f"Training from {len(video_ids)} specific videos...")

        # Load only the specified transcripts
        transcripts = []
        for vid in video_ids:
            transcript_file = self.transcripts_dir / f"{vid}.json"
            if transcript_file.exists():
                try:
                    with open(transcript_file) as f:
                        transcript = json.load(f)
                        if transcript.get('full_text'):
                            transcripts.append(transcript)
                except Exception as e:
                    logger.warning(f"Error loading transcript {vid}: {e}")

        if not transcripts:
            logger.warning("No valid transcripts found for specified videos")
            return {'status': 'no_data', 'n_transcripts': 0}

        logger.info(f"Loaded {len(transcripts)} transcripts from specified videos")

        # Train with these transcripts
        return self.train(transcripts=transcripts, incremental=incremental)

    def _record_video_learnings(self, transcripts: List[Dict], all_rules: List[Dict]):
        """Record learnings for each video to the training database"""
        import re

        # Pre-compile concept patterns for detection
        concept_patterns = {}
        for concept, terms in SMART_MONEY_VOCABULARY.items():
            term_pattern = '|'.join(re.escape(t) for t in terms)
            concept_patterns[concept] = re.compile(rf'\b({term_pattern})\b', re.IGNORECASE)

        # Group rules by source video
        rules_by_video = defaultdict(list)
        for rule in all_rules:
            vid = rule.get('source_video', '')
            if vid:
                rules_by_video[vid].append(rule)

        # Track videos by playlist for playlist-level recording
        videos_by_playlist = defaultdict(list)

        for transcript in transcripts:
            video_id = transcript.get('video_id', '')
            if not video_id:
                continue

            text = transcript.get('full_text', '')

            # Detect concepts in this video
            video_concepts = []
            for concept, pattern in concept_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    video_concepts.extend([concept] * len(matches))

            # Get video's rules
            video_rules = rules_by_video.get(video_id, [])

            # Get playlist info
            pinfo = self.playlist_meta.get(f"video:{video_id}", {})

            # Add title from playlist meta if not in transcript
            if pinfo.get('video_title') and not transcript.get('title'):
                transcript['title'] = pinfo['video_title']

            # Record to database
            self.training_db.record_video_training(
                video_id=video_id,
                transcript=transcript,
                concepts_detected=video_concepts,
                rules_extracted=video_rules,
                playlist_info={
                    'playlist_id': pinfo.get('playlist_id', ''),
                    'title': pinfo.get('playlist_title', ''),
                }
            )

            # Track for playlist recording
            if pinfo.get('playlist_id'):
                videos_by_playlist[pinfo['playlist_id']].append(video_id)

        # Record playlist-level training
        for playlist_id, video_ids in videos_by_playlist.items():
            pdata = self.playlist_meta.get(playlist_id, {})
            self.training_db.record_playlist_training(
                playlist_id=playlist_id,
                videos_trained=video_ids,
                playlist_meta=pdata
            )

        # Save the database
        self.training_db.save()

        # Export markdown report
        report_path = self.data_dir / "TRAINING_DATABASE_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(self.training_db.export_to_markdown())
        logger.info(f"Training report exported to {report_path}")

    def query_concept(self, concept_name: str) -> Dict:
        """Get comprehensive information about a concept"""
        if concept_name not in SMART_MONEY_VOCABULARY:
            return {'error': f'Unknown concept: {concept_name}'}

        result = {
            'name': concept_name,
            'terms': SMART_MONEY_VOCABULARY[concept_name],
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
        """Predict Smart Money concepts in new text"""
        if not self.concept_classifier.is_fitted:
            return {'error': 'Model not trained. Call train() first.'}

        predictions = self.concept_classifier.predict([text])
        return predictions[0] if predictions else {}

    def get_learning_progress(self) -> Dict:
        """Get summary of learning progress"""
        classifier_trend = self.concept_classifier.get_performance_trend()

        return {
            'n_transcripts_processed': self.n_transcripts_processed,
            'n_trained_videos': len(self.trained_video_ids) if self.trained_video_ids else 0,
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

    def get_training_database_report(self) -> Dict:
        """Get comprehensive report from the training database"""
        return self.training_db.get_training_report()

    def get_video_training_summary(self, video_id: str) -> Optional[Dict]:
        """Get training summary for a specific video"""
        return self.training_db.get_video_summary(video_id)

    def get_playlist_training_summary(self, playlist_id: str) -> Optional[Dict]:
        """Get training summary for a specific playlist"""
        return self.training_db.get_playlist_summary(playlist_id)

    def get_all_trained_videos(self) -> List[Dict]:
        """Get list of all trained videos"""
        return self.training_db.get_all_trained_videos()

    def get_all_playlists(self) -> List[Dict]:
        """Get list of all playlists"""
        return self.training_db.get_all_playlists()

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
                'trained_video_ids': getattr(self, 'trained_video_ids', []),
                'vision_knowledge': getattr(self, 'vision_knowledge', {}),
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
                self.trained_video_ids = data.get('trained_video_ids', [])
        except:
            logger.warning("Could not load knowledge base")

        return self


class SignalPredictionModel:
    """
    Predicts trading signals based on learned Smart Money concepts.
    Combines concept presence with market context for predictions.
    """

    def __init__(self, knowledge_base: SmartMoneyKnowledgeBase):
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
    print("Smart Money TRADING ML TRAINING PIPELINE")
    print("=" * 60)

    kb = SmartMoneyKnowledgeBase(data_dir)

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
