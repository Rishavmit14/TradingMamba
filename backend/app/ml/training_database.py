"""
Video Training Database

Tracks all videos the ML has been trained on, including:
- Training history per video
- Key learnings extracted
- Concept summaries
- Training dates and counts

100% FREE - Uses JSON file storage (no database required)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoTrainingRecord:
    """Record for a single video's training history"""

    def __init__(self, video_id: str, data: Dict = None):
        self.video_id = video_id
        data = data or {}

        # Basic info
        self.title = data.get('title', '')
        self.playlist_id = data.get('playlist_id', '')
        self.playlist_name = data.get('playlist_name', '')
        self.duration = data.get('duration', 0)
        self.word_count = data.get('word_count', 0)

        # Training history
        self.training_count = data.get('training_count', 0)
        self.first_trained = data.get('first_trained')
        self.last_trained = data.get('last_trained')
        self.training_dates = data.get('training_dates', [])

        # Key learnings
        self.concepts_detected = data.get('concepts_detected', [])
        self.concept_frequencies = data.get('concept_frequencies', {})
        self.trading_rules_extracted = data.get('trading_rules_extracted', [])
        self.key_points = data.get('key_points', [])
        self.summary = data.get('summary', '')

        # Quality metrics
        self.content_quality_score = data.get('content_quality_score', 0.0)
        self.concept_density = data.get('concept_density', 0.0)

    def to_dict(self) -> Dict:
        return {
            'video_id': self.video_id,
            'title': self.title,
            'playlist_id': self.playlist_id,
            'playlist_name': self.playlist_name,
            'duration': self.duration,
            'word_count': self.word_count,
            'training_count': self.training_count,
            'first_trained': self.first_trained,
            'last_trained': self.last_trained,
            'training_dates': self.training_dates,
            'concepts_detected': self.concepts_detected,
            'concept_frequencies': self.concept_frequencies,
            'trading_rules_extracted': self.trading_rules_extracted,
            'key_points': self.key_points,
            'summary': self.summary,
            'content_quality_score': self.content_quality_score,
            'concept_density': self.concept_density,
        }


class PlaylistTrainingRecord:
    """Record for a playlist's training history"""

    def __init__(self, playlist_id: str, data: Dict = None):
        self.playlist_id = playlist_id
        data = data or {}

        self.title = data.get('title', '')
        self.tier = data.get('tier', 1)
        self.total_videos = data.get('total_videos', 0)
        self.videos_trained = data.get('videos_trained', 0)
        self.total_training_runs = data.get('total_training_runs', 0)
        self.first_trained = data.get('first_trained')
        self.last_trained = data.get('last_trained')

        # Aggregated learnings
        self.top_concepts = data.get('top_concepts', [])
        self.total_rules_extracted = data.get('total_rules_extracted', 0)
        self.playlist_summary = data.get('playlist_summary', '')

        # Video IDs in this playlist
        self.video_ids = data.get('video_ids', [])

    def to_dict(self) -> Dict:
        return {
            'playlist_id': self.playlist_id,
            'title': self.title,
            'tier': self.tier,
            'total_videos': self.total_videos,
            'videos_trained': self.videos_trained,
            'total_training_runs': self.total_training_runs,
            'first_trained': self.first_trained,
            'last_trained': self.last_trained,
            'top_concepts': self.top_concepts,
            'total_rules_extracted': self.total_rules_extracted,
            'playlist_summary': self.playlist_summary,
            'video_ids': self.video_ids,
        }


class TrainingDatabase:
    """
    Central database for tracking all video training history.
    Stores per-video learnings, concept extractions, and summaries.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.db_path = self.data_dir / "training_database.json"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.playlists_dir = self.data_dir / "playlists"

        # In-memory storage
        self.videos: Dict[str, VideoTrainingRecord] = {}
        self.playlists: Dict[str, PlaylistTrainingRecord] = {}
        self.global_stats = {
            'total_videos_trained': 0,
            'total_training_runs': 0,
            'total_words_processed': 0,
            'total_rules_extracted': 0,
            'last_training_date': None,
            'concept_global_frequencies': {},
        }

        # Load existing database
        self.load()

    def load(self):
        """Load database from disk"""
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    data = json.load(f)

                # Load videos
                for vid, vdata in data.get('videos', {}).items():
                    self.videos[vid] = VideoTrainingRecord(vid, vdata)

                # Load playlists
                for pid, pdata in data.get('playlists', {}).items():
                    self.playlists[pid] = PlaylistTrainingRecord(pid, pdata)

                # Load global stats
                self.global_stats = data.get('global_stats', self.global_stats)

                logger.info(f"Loaded training database: {len(self.videos)} videos, {len(self.playlists)} playlists")
            except Exception as e:
                logger.warning(f"Error loading database: {e}")

    def save(self):
        """Save database to disk"""
        data = {
            'videos': {vid: v.to_dict() for vid, v in self.videos.items()},
            'playlists': {pid: p.to_dict() for pid, p in self.playlists.items()},
            'global_stats': self.global_stats,
            'last_saved': datetime.utcnow().isoformat(),
        }

        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved training database to {self.db_path}")

    def _extract_key_points(self, text: str, concepts: List[str]) -> List[str]:
        """Extract key learning points from transcript text"""
        import re

        key_points = []

        # Look for teaching patterns
        patterns = [
            r"(?:the key is|important thing is|remember that|the main thing is)\s+([^.]{20,150})",
            r"(?:this means|this indicates|this tells us)\s+([^.]{20,150})",
            r"(?:we're looking for|look for)\s+([^.]{20,150})",
            r"(?:when you see|if you see)\s+([^.]{20,100})",
            r"(?:that's (?:why|how|what))\s+([^.]{20,100})",
        ]

        text_lower = text.lower()

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches[:3]:  # Limit per pattern
                if any(c.replace('_', ' ') in match or c.replace('_', '') in match for c in concepts):
                    key_points.append(match.strip().capitalize())

        # Deduplicate and limit
        seen = set()
        unique_points = []
        for point in key_points:
            key = point[:30].lower()
            if key not in seen:
                seen.add(key)
                unique_points.append(point)

        return unique_points[:10]

    def _generate_summary(self, video_id: str, title: str, concepts: List[str],
                          rules: List[Dict], key_points: List[str]) -> str:
        """Generate a summary of what was learned from this video"""
        if not concepts:
            return "No significant Smart Money concepts detected in this video."

        # Count concept categories
        concept_counts = defaultdict(int)
        for c in concepts:
            concept_counts[c] += 1

        top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:5]

        # Build summary
        summary_parts = []

        # Main concepts
        concept_str = ", ".join(c[0].replace('_', ' ') for c in top_concepts[:3])
        summary_parts.append(f"Primary concepts covered: {concept_str}")

        # Rules count
        if rules:
            summary_parts.append(f"Extracted {len(rules)} trading rules")

        # Key points count
        if key_points:
            summary_parts.append(f"Identified {len(key_points)} key learning points")

        return ". ".join(summary_parts) + "."

    def record_video_training(self, video_id: str, transcript: Dict,
                               concepts_detected: List[str],
                               rules_extracted: List[Dict],
                               playlist_info: Dict = None) -> VideoTrainingRecord:
        """
        Record a video training session with all learnings.

        Args:
            video_id: YouTube video ID
            transcript: Full transcript data
            concepts_detected: List of Smart Money concepts found
            rules_extracted: Trading rules extracted from this video
            playlist_info: Optional playlist metadata
        """
        now = datetime.utcnow().isoformat()

        # Get or create record
        if video_id in self.videos:
            record = self.videos[video_id]
        else:
            record = VideoTrainingRecord(video_id)

        # Update basic info
        record.title = transcript.get('title', record.title)
        record.word_count = transcript.get('word_count', len(transcript.get('full_text', '').split()))

        if playlist_info:
            record.playlist_id = playlist_info.get('playlist_id', record.playlist_id)
            record.playlist_name = playlist_info.get('title', record.playlist_name)

        # Update training history
        record.training_count += 1
        if not record.first_trained:
            record.first_trained = now
        record.last_trained = now
        record.training_dates.append(now)

        # Keep only last 20 training dates
        record.training_dates = record.training_dates[-20:]

        # Update concept data
        record.concepts_detected = list(set(record.concepts_detected + concepts_detected))

        # Update concept frequencies
        for concept in concepts_detected:
            record.concept_frequencies[concept] = record.concept_frequencies.get(concept, 0) + 1

        # Update rules
        existing_rules = {r.get('text', '')[:50] for r in record.trading_rules_extracted}
        for rule in rules_extracted:
            if rule.get('text', '')[:50] not in existing_rules:
                record.trading_rules_extracted.append(rule)
        record.trading_rules_extracted = record.trading_rules_extracted[:20]  # Limit

        # Extract key points
        full_text = transcript.get('full_text', '')
        new_points = self._extract_key_points(full_text, concepts_detected)
        existing_points = set(p[:30].lower() for p in record.key_points)
        for point in new_points:
            if point[:30].lower() not in existing_points:
                record.key_points.append(point)
        record.key_points = record.key_points[:15]  # Limit

        # Generate summary
        record.summary = self._generate_summary(
            video_id, record.title, record.concepts_detected,
            record.trading_rules_extracted, record.key_points
        )

        # Calculate quality metrics
        if record.word_count > 0:
            record.concept_density = len(concepts_detected) / (record.word_count / 1000)
        record.content_quality_score = min(1.0,
            (len(record.concepts_detected) * 0.1 +
             len(record.trading_rules_extracted) * 0.15 +
             len(record.key_points) * 0.1 +
             record.concept_density * 0.1))

        # Save record
        self.videos[video_id] = record

        # Update global stats
        self._update_global_stats()

        return record

    def record_playlist_training(self, playlist_id: str, videos_trained: List[str],
                                  playlist_meta: Dict = None) -> PlaylistTrainingRecord:
        """Record training for a playlist"""
        now = datetime.utcnow().isoformat()

        # Get or create record
        if playlist_id in self.playlists:
            record = self.playlists[playlist_id]
        else:
            record = PlaylistTrainingRecord(playlist_id)

        # Update from metadata
        if playlist_meta:
            record.title = playlist_meta.get('title', record.title)
            record.tier = playlist_meta.get('tier', record.tier)
            record.total_videos = playlist_meta.get('video_count', record.total_videos)

        # Update training history
        record.videos_trained = len(set(record.video_ids + videos_trained))
        record.total_training_runs += 1
        if not record.first_trained:
            record.first_trained = now
        record.last_trained = now
        record.video_ids = list(set(record.video_ids + videos_trained))

        # Aggregate concept data from videos
        concept_counts = defaultdict(int)
        total_rules = 0

        for vid in record.video_ids:
            if vid in self.videos:
                v = self.videos[vid]
                for concept, count in v.concept_frequencies.items():
                    concept_counts[concept] += count
                total_rules += len(v.trading_rules_extracted)

        record.top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:10]
        record.total_rules_extracted = total_rules

        # Generate playlist summary
        if record.top_concepts:
            top_3 = [c[0].replace('_', ' ') for c in record.top_concepts[:3]]
            record.playlist_summary = (
                f"Playlist covers {len(record.video_ids)} videos focusing on: {', '.join(top_3)}. "
                f"Extracted {total_rules} trading rules total."
            )

        self.playlists[playlist_id] = record
        return record

    def _update_global_stats(self):
        """Update global statistics"""
        self.global_stats['total_videos_trained'] = len(self.videos)
        self.global_stats['total_training_runs'] = sum(v.training_count for v in self.videos.values())
        self.global_stats['total_words_processed'] = sum(v.word_count for v in self.videos.values())
        self.global_stats['total_rules_extracted'] = sum(
            len(v.trading_rules_extracted) for v in self.videos.values()
        )

        # Last training date
        all_dates = [v.last_trained for v in self.videos.values() if v.last_trained]
        if all_dates:
            self.global_stats['last_training_date'] = max(all_dates)

        # Global concept frequencies
        concept_freq = defaultdict(int)
        for v in self.videos.values():
            for concept, count in v.concept_frequencies.items():
                concept_freq[concept] += count
        self.global_stats['concept_global_frequencies'] = dict(concept_freq)

    def get_video_summary(self, video_id: str) -> Optional[Dict]:
        """Get training summary for a specific video"""
        if video_id not in self.videos:
            return None

        v = self.videos[video_id]
        return {
            'video_id': video_id,
            'title': v.title,
            'playlist': v.playlist_name,
            'training_count': v.training_count,
            'first_trained': v.first_trained,
            'last_trained': v.last_trained,
            'concepts': v.concepts_detected,
            'rules_count': len(v.trading_rules_extracted),
            'key_points': v.key_points,
            'summary': v.summary,
            'quality_score': round(v.content_quality_score, 2),
        }

    def get_playlist_summary(self, playlist_id: str) -> Optional[Dict]:
        """Get training summary for a playlist"""
        if playlist_id not in self.playlists:
            return None

        p = self.playlists[playlist_id]
        return {
            'playlist_id': playlist_id,
            'title': p.title,
            'tier': p.tier,
            'videos_trained': p.videos_trained,
            'total_videos': p.total_videos,
            'training_runs': p.total_training_runs,
            'top_concepts': p.top_concepts,
            'rules_extracted': p.total_rules_extracted,
            'summary': p.playlist_summary,
            'video_ids': p.video_ids,
        }

    def get_all_trained_videos(self) -> List[Dict]:
        """Get list of all trained videos with summaries"""
        return [self.get_video_summary(vid) for vid in sorted(self.videos.keys())]

    def get_all_playlists(self) -> List[Dict]:
        """Get list of all playlists with summaries"""
        return [self.get_playlist_summary(pid) for pid in sorted(self.playlists.keys())]

    def get_training_report(self) -> Dict:
        """Generate comprehensive training report"""
        # Videos by playlist
        videos_by_playlist = defaultdict(list)
        for vid, v in self.videos.items():
            videos_by_playlist[v.playlist_id or 'unknown'].append({
                'video_id': vid,
                'title': v.title,
                'training_count': v.training_count,
                'concepts': len(v.concepts_detected),
                'rules': len(v.trading_rules_extracted),
            })

        # Top concepts overall
        concept_freq = self.global_stats.get('concept_global_frequencies', {})
        top_concepts = sorted(concept_freq.items(), key=lambda x: -x[1])[:10]

        return {
            'generated_at': datetime.utcnow().isoformat(),
            'global_stats': self.global_stats,
            'playlists': {
                pid: {
                    'title': self.playlists[pid].title if pid in self.playlists else 'Unknown',
                    'videos': videos
                }
                for pid, videos in videos_by_playlist.items()
            },
            'top_concepts': top_concepts,
            'total_playlists': len(self.playlists),
            'total_videos': len(self.videos),
        }

    def export_to_markdown(self) -> str:
        """Export training database to markdown format"""
        lines = []
        lines.append("# TradingMamba Training Database Report")
        lines.append(f"\n**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

        # Global Stats
        lines.append("## Global Statistics\n")
        lines.append(f"- **Total Videos Trained:** {self.global_stats['total_videos_trained']}")
        lines.append(f"- **Total Training Runs:** {self.global_stats['total_training_runs']}")
        lines.append(f"- **Total Words Processed:** {self.global_stats['total_words_processed']:,}")
        lines.append(f"- **Total Rules Extracted:** {self.global_stats['total_rules_extracted']}")
        lines.append(f"- **Last Training:** {self.global_stats['last_training_date']}\n")

        # Top Concepts
        lines.append("## Top Smart Money Concepts Learned\n")
        concept_freq = self.global_stats.get('concept_global_frequencies', {})
        top_concepts = sorted(concept_freq.items(), key=lambda x: -x[1])[:10]
        for concept, freq in top_concepts:
            lines.append(f"- **{concept.replace('_', ' ').title()}:** {freq} mentions")
        lines.append("")

        # Playlists
        lines.append("## Playlists\n")
        for pid, p in sorted(self.playlists.items(), key=lambda x: x[1].tier):
            lines.append(f"### {p.title or pid}")
            lines.append(f"- **Tier:** {p.tier}")
            lines.append(f"- **Videos Trained:** {p.videos_trained}/{p.total_videos}")
            lines.append(f"- **Training Runs:** {p.total_training_runs}")
            lines.append(f"- **Rules Extracted:** {p.total_rules_extracted}")
            if p.top_concepts:
                concepts_str = ", ".join(f"{c[0].replace('_', ' ')}" for c in p.top_concepts[:5])
                lines.append(f"- **Top Concepts:** {concepts_str}")
            lines.append(f"- **Summary:** {p.playlist_summary}")
            lines.append("")

            # Videos in playlist
            lines.append("#### Videos\n")
            lines.append("| # | Video Title | Trained | Concepts | Rules | Key Points |")
            lines.append("|---|-------------|---------|----------|-------|------------|")

            for i, vid in enumerate(p.video_ids, 1):
                if vid in self.videos:
                    v = self.videos[vid]
                    lines.append(
                        f"| {i} | {v.title[:40]}{'...' if len(v.title) > 40 else ''} | "
                        f"{v.training_count}x | {len(v.concepts_detected)} | "
                        f"{len(v.trading_rules_extracted)} | {len(v.key_points)} |"
                    )
            lines.append("")

        # Detailed Video Learnings
        lines.append("## Detailed Video Learnings\n")
        for vid, v in sorted(self.videos.items(), key=lambda x: x[1].playlist_name):
            lines.append(f"### {v.title or vid}")
            lines.append(f"- **Video ID:** `{vid}`")
            lines.append(f"- **Playlist:** {v.playlist_name}")
            lines.append(f"- **Word Count:** {v.word_count:,}")
            lines.append(f"- **Times Trained:** {v.training_count}")
            lines.append(f"- **First Trained:** {v.first_trained}")
            lines.append(f"- **Last Trained:** {v.last_trained}")
            lines.append(f"- **Quality Score:** {v.content_quality_score:.2f}")
            lines.append("")

            if v.concepts_detected:
                lines.append("**Concepts Detected:**")
                for concept in v.concepts_detected[:10]:
                    freq = v.concept_frequencies.get(concept, 0)
                    lines.append(f"- {concept.replace('_', ' ').title()} ({freq}x)")
                lines.append("")

            if v.key_points:
                lines.append("**Key Learning Points:**")
                for point in v.key_points[:5]:
                    lines.append(f"- {point}")
                lines.append("")

            if v.trading_rules_extracted:
                lines.append("**Trading Rules Extracted:**")
                for rule in v.trading_rules_extracted[:5]:
                    lines.append(f"- [{rule['type']}] {rule['text'][:100]}")
                lines.append("")

            lines.append(f"**Summary:** {v.summary}\n")
            lines.append("---\n")

        return "\n".join(lines)


def migrate_existing_data(db: TrainingDatabase) -> int:
    """Migrate existing transcripts to the training database"""
    from .feature_extractor import SMART_MONEY_VOCABULARY
    import re

    migrated = 0

    # Load playlist metadata
    playlist_meta = {}
    for pfile in db.playlists_dir.glob("*.json"):
        try:
            with open(pfile) as f:
                pdata = json.load(f)
                playlist_meta[pdata['playlist_id']] = pdata
                # Map video IDs to playlist
                for video in pdata.get('videos', []):
                    playlist_meta[video['video_id']] = {
                        'playlist_id': pdata['playlist_id'],
                        'title': pdata.get('title', ''),
                        'video_title': video.get('title', ''),
                    }
        except Exception as e:
            logger.warning(f"Error loading playlist {pfile}: {e}")

    # Pre-compile concept patterns
    concept_patterns = {}
    for concept, terms in SMART_MONEY_VOCABULARY.items():
        term_pattern = '|'.join(re.escape(t) for t in terms)
        concept_patterns[concept] = re.compile(rf'\b({term_pattern})\b', re.IGNORECASE)

    # Process each transcript
    for tfile in db.transcripts_dir.glob("*.json"):
        try:
            with open(tfile) as f:
                transcript = json.load(f)

            video_id = transcript.get('video_id', tfile.stem)

            # Detect concepts
            text = transcript.get('full_text', '')
            concepts = []
            for concept, pattern in concept_patterns.items():
                if pattern.search(text):
                    concepts.append(concept)

            # Get playlist info
            pinfo = playlist_meta.get(video_id, {})

            # Update transcript with title if available
            if pinfo.get('video_title'):
                transcript['title'] = pinfo['video_title']

            # Record training (with empty rules for migration)
            db.record_video_training(
                video_id=video_id,
                transcript=transcript,
                concepts_detected=concepts,
                rules_extracted=[],
                playlist_info={
                    'playlist_id': pinfo.get('playlist_id', ''),
                    'title': pinfo.get('title', ''),
                }
            )

            migrated += 1

        except Exception as e:
            logger.warning(f"Error migrating {tfile}: {e}")

    # Update playlist records
    videos_by_playlist = defaultdict(list)
    for vid, v in db.videos.items():
        if v.playlist_id:
            videos_by_playlist[v.playlist_id].append(vid)

    for pid, vids in videos_by_playlist.items():
        if pid in playlist_meta:
            db.record_playlist_training(
                playlist_id=pid,
                videos_trained=vids,
                playlist_meta=playlist_meta[pid]
            )

    db.save()
    logger.info(f"Migrated {migrated} transcripts to training database")
    return migrated


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING DATABASE MIGRATION")
    print("=" * 60)

    db = TrainingDatabase()

    # Migrate existing data
    count = migrate_existing_data(db)
    print(f"\nMigrated {count} videos")

    # Generate report
    report = db.get_training_report()
    print(f"\nTotal Videos: {report['total_videos']}")
    print(f"Total Playlists: {report['total_playlists']}")
    print(f"Top Concepts: {report['top_concepts'][:5]}")

    # Export to markdown
    md_path = db.data_dir / "TRAINING_DATABASE_REPORT.md"
    with open(md_path, 'w') as f:
        f.write(db.export_to_markdown())
    print(f"\nExported report to: {md_path}")
