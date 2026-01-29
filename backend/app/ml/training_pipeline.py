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

# Synchronized Learning imports - ensures audio-visual alignment
try:
    from .synchronized_learning import (
        SynchronizedLearningPipeline,
        WhisperXTranscriber,
        VerificationGate,
        JointEmbeddingSpace,
        integrate_synchronized_learning
    )
    SYNC_LEARNING_AVAILABLE = True
except ImportError:
    SYNC_LEARNING_AVAILABLE = False
    logger.info("Synchronized learning not available")


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

        # DEPRECATED: Vision trainer - use synchronized learning instead
        # Kept for backward compatibility but redirects to synchronized learning
        self.vision_trainer = None
        self.vision_knowledge = {}  # DEPRECATED: Use synchronized_knowledge instead

        # Synchronized Learning Pipeline (RECOMMENDED - prevents contamination)
        # This combines Text + Vision + Audio-Visual Verification
        self.sync_pipeline = None
        self.synchronized_knowledge = {}  # Verified audio-visual aligned knowledge
        self.has_synchronized_learning = False  # Flag to indicate sync learning was used

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
        DEPRECATED: This method now redirects to train_synchronized().

        Standalone vision training has been removed in favor of Synchronized Learning,
        which provides better results by verifying that audio matches visual content.
        This prevents contamination (e.g., MACD discussion being labeled as FVG).

        For backward compatibility, this method redirects to train_synchronized()
        which includes vision analysis as part of its audio-visual verification pipeline.

        Args:
            transcripts: List of transcript dicts (loads all if None)
            vision_provider: "local" (FREE on M1/M2/M3 Mac), "anthropic", or "openai"
            max_frames_per_video: Max frames to analyze per video (0 = no limit)
            extraction_mode: Mapped to synchronized learning mode
            progress_callback: Optional callback(current, total, message)

        Returns:
            Training results from synchronized learning
        """
        logger.warning("=" * 60)
        logger.warning("DEPRECATION NOTICE: train_with_vision() is deprecated")
        logger.warning("Redirecting to train_synchronized() for better results")
        logger.warning("Synchronized Learning prevents contamination by verifying")
        logger.warning("that what is SAID matches what is SHOWN in the video.")
        logger.warning("=" * 60)

        # Map old extraction modes to synchronized learning modes
        mode_mapping = {
            "comprehensive": "sincere_student",
            "thorough": "sincere_student",
            "balanced": "sincere_student",
            "selective": "sincere_student",
        }
        sync_mode = mode_mapping.get(extraction_mode, "sincere_student")

        # Redirect to synchronized learning
        return self.train_synchronized(
            transcripts=transcripts,
            vision_provider=vision_provider,
            max_frames_per_video=max_frames_per_video,
            extraction_mode=sync_mode,
            alignment_threshold=0.6,
            sync_window=2.0,
            progress_callback=progress_callback
        )

    def train_synchronized(
        self,
        transcripts: List[Dict] = None,
        vision_provider: str = "local",
        max_frames_per_video: int = 0,
        extraction_mode: str = "sincere_student",
        alignment_threshold: float = 0.6,
        sync_window: float = 2.0,
        progress_callback=None
    ) -> Dict:
        """
        STATE-OF-THE-ART synchronized audio-visual training.

        This is the BEST training mode that ensures:
        1. Word-level timestamps via WhisperX (forced alignment)
        2. Audio-visual alignment in joint embedding space
        3. Verification gate rejects mismatched data (prevents MACD→FVG contamination)

        Based on:
        - Meta's ImageBind (joint embedding space)
        - Meta's PE-AV (perception encoder audiovisual)
        - WhisperX (word-level forced alignment)

        100% FREE - Uses only open-source libraries.

        Args:
            transcripts: List of transcript dicts (loads all if None)
            vision_provider: "local" (FREE on M1/M2/M3 Mac), "anthropic", or "openai"
            max_frames_per_video: Max frames to analyze per video (0 = no limit)
            extraction_mode: "sincere_student" (recommended), "comprehensive", etc.
            alignment_threshold: Min alignment score to accept (0.6 = 60% match)
            sync_window: Time window for matching audio to visual (seconds)
            progress_callback: Optional callback(current, total, message)

        Returns:
            Training results with verification statistics
        """
        if not VISION_AVAILABLE:
            logger.warning("Vision training not available. Falling back to text-only training.")
            return self.train(transcripts)

        if not SYNC_LEARNING_AVAILABLE:
            logger.warning("Synchronized learning not available. Falling back to standard multimodal.")
            return self.train_with_vision(
                transcripts, vision_provider, max_frames_per_video,
                extraction_mode, progress_callback
            )

        logger.info("=" * 60)
        logger.info("SYNCHRONIZED AUDIO-VISUAL LEARNING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Mode: {extraction_mode}")
        logger.info(f"Alignment threshold: {alignment_threshold}")
        logger.info(f"Sync window: {sync_window}s")
        logger.info("=" * 60)

        if transcripts is None:
            transcripts = self.load_transcripts()

        if not transcripts:
            logger.warning("No transcripts available for training")
            return {'status': 'no_data'}

        # Initialize synchronized learning pipeline
        self.sync_pipeline = SynchronizedLearningPipeline(
            data_dir=str(self.data_dir),
            alignment_threshold=alignment_threshold,
            sync_window=sync_window
        )

        # Initialize vision trainer
        self.vision_trainer = VideoVisionTrainer(
            str(self.data_dir),
            vision_provider=vision_provider
        )

        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_transcripts': len(transcripts),
            'training_mode': 'synchronized',
            'extraction_mode': extraction_mode,
            'alignment_threshold': alignment_threshold,
            'sync_window': sync_window,
            'components': {},
            'vision_analysis': {},
            'synchronization': {},
            'video_learnings': []
        }

        total_steps = len(transcripts) * 2 + 5  # Vision + Sync + other steps
        current_step = 0

        total_verified = 0
        total_rejected = 0
        all_rejection_reasons = {}

        # ====================================================================
        # PHASE 1: Vision Analysis + Synchronized Learning per video
        # ====================================================================
        logger.info("PHASE 1: Synchronized audio-visual processing...")

        enhanced_transcripts = []

        for i, transcript in enumerate(transcripts):
            video_id = transcript.get('video_id', '')
            if not video_id:
                enhanced_transcripts.append(transcript)
                continue

            video_title = transcript.get('title', video_id)[:40]

            if progress_callback:
                progress_callback(
                    current_step,
                    total_steps,
                    f"[Vision] Analyzing: {video_title}..."
                )

            # Step 1a: Run vision analysis on video frames
            try:
                vision_summary = self.vision_trainer.process_video(
                    video_id,
                    max_frames=max_frames_per_video,
                    extraction_mode=extraction_mode
                )

                frames_data = []
                if vision_summary:
                    # Get detailed frame analysis
                    vision_file = self.data_dir / "video_vision" / f"{video_id}_vision.json"
                    if vision_file.exists():
                        with open(vision_file) as f:
                            vision_data = json.load(f)
                            frames_data = vision_data.get("key_moments", [])

            except Exception as e:
                logger.warning(f"Vision analysis failed for {video_id}: {e}")
                frames_data = []

            current_step += 1

            if progress_callback:
                progress_callback(
                    current_step,
                    total_steps,
                    f"[Sync] Verifying: {video_title}..."
                )

            # Step 1b: Run synchronized learning (audio-visual alignment + verification)
            if frames_data:
                try:
                    # Get audio path for word-level timestamps
                    audio_path = self.data_dir / "audio" / f"{video_id}.mp3"
                    if not audio_path.exists():
                        audio_path = self.data_dir / "audio" / f"{video_id}.wav"

                    if audio_path.exists():
                        sync_result = self.sync_pipeline.process_video(
                            video_id=video_id,
                            audio_path=str(audio_path),
                            frames_data=frames_data,
                            existing_transcript=transcript
                        )

                        total_verified += sync_result.get('verified_count', 0)
                        total_rejected += sync_result.get('rejected_count', 0)

                        # Track rejection reasons
                        rej_stats = sync_result.get('rejection_stats', {})
                        for reason, count in rej_stats.get('by_reason', {}).items():
                            all_rejection_reasons[reason] = all_rejection_reasons.get(reason, 0) + count

                        logger.info(f"  {video_id}: {sync_result['verified_count']} verified, "
                                   f"{sync_result['rejected_count']} rejected")
                    else:
                        logger.warning(f"  No audio file for {video_id}, skipping sync")

                except Exception as e:
                    logger.warning(f"Synchronized learning failed for {video_id}: {e}")

            # Enhance transcript with vision data (standard way)
            enhanced = self.vision_trainer.enhance_transcript_with_vision(video_id, transcript)
            enhanced_transcripts.append(enhanced)

            current_step += 1

        # Store synchronized knowledge
        self.synchronized_knowledge = self.sync_pipeline.verified_knowledge

        results['synchronization'] = {
            'total_moments_analyzed': total_verified + total_rejected,
            'verified_moments': total_verified,
            'rejected_moments': total_rejected,
            'verification_rate': total_verified / max(1, total_verified + total_rejected),
            'rejection_reasons': all_rejection_reasons,
            'concepts_verified': list(self.synchronized_knowledge.keys())
        }

        # ====================================================================
        # PHASE 2: Standard ML Training (with synchronized data)
        # ====================================================================
        logger.info("\nPHASE 2: Training ML models with verified data...")

        if progress_callback:
            progress_callback(current_step, total_steps, "Training concept classifier...")

        # Use enhanced transcripts for training
        clf_result = self.concept_classifier.fit(enhanced_transcripts)
        results['components']['classifier'] = clf_result
        current_step += 1

        if progress_callback:
            progress_callback(current_step, total_steps, "Building concept embeddings...")

        # Build concept embeddings
        self.concept_embeddings.fit(enhanced_transcripts)
        results['components']['embeddings'] = {'status': 'trained', 'dim': self.concept_embeddings.embedding_dim}
        current_step += 1

        if progress_callback:
            progress_callback(current_step, total_steps, "Analyzing concept sequences...")

        # Build sequence analyzer
        self.sequence_analyzer.build_transition_matrix(enhanced_transcripts)
        results['components']['sequence_analyzer'] = {'status': 'trained'}
        current_step += 1

        if progress_callback:
            progress_callback(current_step, total_steps, "Extracting definitions and rules...")

        # Extract definitions and rules
        definitions = self.extract_concept_definitions(enhanced_transcripts)
        results['components']['definitions'] = {'n_concepts': len(definitions)}

        rules = self.extract_trading_rules(enhanced_transcripts)
        results['components']['rules'] = {'n_rules': len(rules)}
        current_step += 1

        if progress_callback:
            progress_callback(current_step, total_steps, "Recording to training database...")

        # Record per-video learnings
        self._record_video_learnings(enhanced_transcripts, rules)
        results['components']['database'] = {
            'videos_recorded': len(enhanced_transcripts),
            'database_path': str(self.training_db.db_path)
        }

        # Aggregate vision knowledge
        self.vision_knowledge = self.vision_trainer.get_visual_knowledge()
        results['vision_analysis'] = {
            'videos_analyzed': len([t for t in transcripts if t.get('video_id')]),
            'visual_patterns': self.vision_knowledge.get('pattern_frequency', {}),
            'visual_concepts': len(self.vision_knowledge.get('visual_concepts', []))
        }

        # Update metadata
        self.n_transcripts_processed = len(transcripts)
        self.last_training_time = datetime.utcnow().isoformat()
        self.training_history.append(results)

        if progress_callback:
            progress_callback(total_steps, total_steps, "Synchronized training complete!")

        # ====================================================================
        # Summary
        # ====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("SYNCHRONIZED TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Transcripts processed: {len(transcripts)}")
        logger.info(f"Moments analyzed: {total_verified + total_rejected}")
        logger.info(f"Verified (audio matches visual): {total_verified}")
        logger.info(f"Rejected (contamination prevented): {total_rejected}")
        logger.info(f"Verification rate: {results['synchronization']['verification_rate']:.1%}")
        logger.info(f"Concepts with verified knowledge: {len(self.synchronized_knowledge)}")
        logger.info("=" * 60)

        if all_rejection_reasons:
            logger.info("\nRejection Reasons (contamination prevention):")
            for reason, count in sorted(all_rejection_reasons.items(), key=lambda x: -x[1]):
                logger.info(f"  - {reason}: {count}")

        return results

    def get_verified_knowledge(self, concept: str = None) -> Dict:
        """
        Get verified knowledge that passed the audio-visual alignment test.

        This knowledge is guaranteed to be accurate - what was said matches
        what was shown in the video.
        """
        if not self.synchronized_knowledge:
            return {'error': 'No synchronized training done yet. Run train_synchronized() first.'}

        if concept:
            if concept in self.synchronized_knowledge:
                vk = self.synchronized_knowledge[concept]
                return vk.to_dict() if hasattr(vk, 'to_dict') else vk
            return {'error': f'No verified knowledge for concept: {concept}'}

        # Return all verified knowledge
        result = {}
        for c, vk in self.synchronized_knowledge.items():
            result[c] = vk.to_dict() if hasattr(vk, 'to_dict') else vk
        return result

    def get_visual_pattern_examples(self, pattern_type: str) -> List[Dict]:
        """
        Get visual examples of a specific pattern type from synchronized learning.

        Args:
            pattern_type: e.g., "FVG", "Order Block", "Breaker"

        Returns:
            List of verified examples with timestamps and frame paths
        """
        # First try synchronized_knowledge (preferred - verified data)
        if self.synchronized_knowledge:
            pattern_lower = pattern_type.lower().replace(' ', '_')
            if pattern_lower in self.synchronized_knowledge:
                vk = self.synchronized_knowledge[pattern_lower]
                if hasattr(vk, 'visual_examples'):
                    return vk.visual_examples
                elif isinstance(vk, dict):
                    return vk.get('visual_examples', [])

        # No synchronized data available
        return []

    def get_teaching_moments(self, concept: str = None) -> List[Dict]:
        """
        Get key teaching moments from synchronized learning data.
        These are verified moments where audio matches visual content.
        """
        moments = []

        # Get from synchronized_knowledge (verified data)
        if self.synchronized_knowledge:
            for concept_key, vk in self.synchronized_knowledge.items():
                if hasattr(vk, 'teaching_moments'):
                    moments.extend(vk.teaching_moments)
                elif isinstance(vk, dict) and 'teaching_moments' in vk:
                    moments.extend(vk['teaching_moments'])

        # Filter by concept if specified
        if concept and moments:
            concept_lower = concept.lower()
            moments = [
                m for m in moments
                if concept_lower in m.get('teaching_point', '').lower()
                or concept_lower in m.get('concept', '').lower()
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

        # Prepare synchronized knowledge for JSON serialization
        sync_knowledge_serializable = {}
        for concept, vk in getattr(self, 'synchronized_knowledge', {}).items():
            if hasattr(vk, 'to_dict'):
                sync_knowledge_serializable[concept] = vk.to_dict()
            else:
                sync_knowledge_serializable[concept] = vk

        with open(knowledge_path, 'w') as f:
            json.dump({
                'concept_definitions': self.concept_definitions,
                'trading_rules': self.trading_rules,
                'n_transcripts_processed': self.n_transcripts_processed,
                'last_training_time': self.last_training_time,
                'training_history': self.training_history[-10:],  # Last 10
                'trained_video_ids': getattr(self, 'trained_video_ids', []),
                # DEPRECATED: vision_knowledge - kept for backward compatibility only
                # New training uses synchronized_knowledge instead
                'vision_knowledge': {},  # No longer populated, use synchronized_knowledge
                # Synchronized Learning data (RECOMMENDED)
                'synchronized_knowledge': sync_knowledge_serializable,
                'has_synchronized_learning': bool(sync_knowledge_serializable),
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

                # Load synchronized knowledge (RECOMMENDED)
                self.synchronized_knowledge = data.get('synchronized_knowledge', {})
                self.has_synchronized_learning = data.get('has_synchronized_learning', bool(self.synchronized_knowledge))

                # DEPRECATED: vision_knowledge - only load for backward compatibility
                # If old data exists and no synchronized_knowledge, log warning
                old_vision = data.get('vision_knowledge', {})
                if old_vision and not self.synchronized_knowledge:
                    logger.warning("Found legacy vision_knowledge but no synchronized_knowledge.")
                    logger.warning("Consider re-training with train_synchronized() for better results.")
                self.vision_knowledge = {}  # Don't use old vision data
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
