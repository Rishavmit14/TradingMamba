"""
Playlist Registry - Maps playlist_id to isolated ML stacks.

Each playlist gets its own:
- MLPatternEngine (loaded from only that playlist's videos)
- VideoKnowledgeIndex (loaded from only that playlist's videos)
- SmartMoneyAnalyzer (using the playlist-specific MLPatternEngine)
- SignalGenerator (using the playlist-specific MLPatternEngine)

Instances are cached so repeated requests for the same playlist
don't reload from disk.

The special playlist_id "all" loads everything (backward compatible).
"""

import contextvars
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Context variable for deep-stack playlist scoping
# Set this before calling signal_fusion / feature_engineering
# so they pick up the right VideoKnowledgeIndex
playlist_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    'playlist_id', default='all'
)


def _get_data_dir() -> Path:
    """Resolve the project data directory."""
    return Path(__file__).parent.parent.parent.parent / "data"


class PlaylistRegistry:
    """
    Caches per-playlist ML engine + video knowledge instances.

    Usage:
        engine = PlaylistRegistry.get_ml_engine("PLVgHx4Z63paaRnabpBl38GoMkxF1FiXCF")
        # Returns MLPatternEngine loaded with only that playlist's video knowledge
    """

    # Cache: playlist_id -> {ml_engine, video_knowledge, analyzer, signal_generator}
    _instances: Dict[str, dict] = {}

    @classmethod
    def get_video_ids_for_playlist(cls, playlist_id: str) -> Optional[List[str]]:
        """
        Read playlist JSON and return its video IDs.

        Args:
            playlist_id: Playlist ID, or "all" for no filtering.

        Returns:
            List of video_id strings, or None for "all" (= load everything).
        """
        if playlist_id == "all":
            return None

        playlist_file = _get_data_dir() / "playlists" / f"{playlist_id}.json"
        if not playlist_file.exists():
            logger.warning(f"Playlist file not found: {playlist_file}")
            return None

        try:
            with open(playlist_file) as f:
                data = json.load(f)
            video_ids = [v['video_id'] for v in data.get('videos', [])]
            logger.info(f"Playlist {playlist_id}: {len(video_ids)} video IDs")
            return video_ids
        except Exception as e:
            logger.error(f"Failed to read playlist {playlist_id}: {e}")
            return None

    @classmethod
    def get_ml_engine(cls, playlist_id: str = "all"):
        """Get or create cached MLPatternEngine filtered to playlist's videos."""
        if playlist_id in cls._instances and 'ml_engine' in cls._instances[playlist_id]:
            return cls._instances[playlist_id]['ml_engine']

        from .ml_pattern_engine import MLPatternEngine

        video_ids = cls.get_video_ids_for_playlist(playlist_id)
        engine = MLPatternEngine(video_ids=video_ids)

        if playlist_id not in cls._instances:
            cls._instances[playlist_id] = {}
        cls._instances[playlist_id]['ml_engine'] = engine

        logger.info(f"Created MLPatternEngine for playlist={playlist_id} "
                     f"(video_ids={'all' if video_ids is None else len(video_ids)})")
        return engine

    @classmethod
    def get_video_knowledge(cls, playlist_id: str = "all"):
        """Get or create cached VideoKnowledgeIndex filtered to playlist's videos."""
        if playlist_id in cls._instances and 'video_knowledge' in cls._instances[playlist_id]:
            return cls._instances[playlist_id]['video_knowledge']

        from .video_knowledge import VideoKnowledgeIndex

        video_ids = cls.get_video_ids_for_playlist(playlist_id)
        vk = VideoKnowledgeIndex(video_ids=set(video_ids) if video_ids else None)

        if playlist_id not in cls._instances:
            cls._instances[playlist_id] = {}
        cls._instances[playlist_id]['video_knowledge'] = vk

        logger.info(f"Created VideoKnowledgeIndex for playlist={playlist_id} "
                     f"(video_ids={'all' if video_ids is None else len(video_ids)})")
        return vk

    @classmethod
    def get_analyzer(cls, playlist_id: str = "all"):
        """Get SmartMoneyAnalyzer using playlist-specific ML engine."""
        if playlist_id in cls._instances and 'analyzer' in cls._instances[playlist_id]:
            return cls._instances[playlist_id]['analyzer']

        from ..services.smart_money_analyzer import SmartMoneyAnalyzer

        ml_engine = cls.get_ml_engine(playlist_id)
        analyzer = SmartMoneyAnalyzer(use_ml=True, ml_engine=ml_engine)

        if playlist_id not in cls._instances:
            cls._instances[playlist_id] = {}
        cls._instances[playlist_id]['analyzer'] = analyzer

        return analyzer

    @classmethod
    def get_signal_generator(cls, playlist_id: str = "all"):
        """Get SignalGenerator using playlist-specific ML engine."""
        if playlist_id in cls._instances and 'signal_generator' in cls._instances[playlist_id]:
            return cls._instances[playlist_id]['signal_generator']

        from ..services.signal_generator import SignalGenerator

        ml_engine = cls.get_ml_engine(playlist_id)
        sig_gen = SignalGenerator(ml_engine=ml_engine)

        if playlist_id not in cls._instances:
            cls._instances[playlist_id] = {}
        cls._instances[playlist_id]['signal_generator'] = sig_gen

        return sig_gen

    @classmethod
    def get_available_playlists(cls) -> List[dict]:
        """
        List all playlists with training stats for the dropdown.

        Returns list of dicts:
        [{id, title, channel, video_count, trained_video_count, concepts_learned}]
        """
        playlists_dir = _get_data_dir() / "playlists"
        training_dir = _get_data_dir() / "audio_first_training"
        result = []

        if not playlists_dir.exists():
            return result

        # Get set of trained video IDs (have knowledge base files)
        trained_video_ids = set()
        if training_dir.exists():
            for kb_file in training_dir.glob("*_knowledge_base.json"):
                vid = kb_file.stem.replace('_knowledge_base', '')
                trained_video_ids.add(vid)

        for pf in sorted(playlists_dir.glob("*.json")):
            try:
                with open(pf) as f:
                    data = json.load(f)

                playlist_id = data.get('playlist_id', pf.stem)
                videos = data.get('videos', [])
                video_ids = [v['video_id'] for v in videos]

                # Count how many are trained
                trained_count = sum(1 for vid in video_ids if vid in trained_video_ids)

                # Get concepts learned for this playlist's videos
                concepts = set()
                for vid in video_ids:
                    kb_path = training_dir / f"{vid}_knowledge_base.json"
                    if kb_path.exists():
                        try:
                            with open(kb_path) as kf:
                                kb_data = json.load(kf)
                            for concept_name in kb_data.get('concepts', {}):
                                concepts.add(concept_name.lower().replace(' ', '_'))
                        except Exception:
                            pass

                result.append({
                    'id': playlist_id,
                    'title': data.get('title', 'Unknown'),
                    'channel': data.get('channel', 'Unknown'),
                    'video_count': len(videos),
                    'trained_video_count': trained_count,
                    'concepts_learned': sorted(concepts),
                    'concepts_count': len(concepts),
                })

            except Exception as e:
                logger.error(f"Failed to read playlist {pf.name}: {e}")

        return result

    @classmethod
    def invalidate(cls, playlist_id: str = None):
        """
        Clear cache for a playlist (or all) when training completes.

        Args:
            playlist_id: Specific playlist to invalidate, or None for all.
        """
        if playlist_id is None:
            cls._instances.clear()
            logger.info("PlaylistRegistry: cleared all cached instances")
        elif playlist_id in cls._instances:
            del cls._instances[playlist_id]
            logger.info(f"PlaylistRegistry: cleared cache for {playlist_id}")
