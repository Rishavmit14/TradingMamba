"""Video and Transcript models for Smart Money video processing"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field
from uuid import uuid4


class VideoStatus(Enum):
    """Video processing status"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    EXTRACTING_CONCEPTS = "extracting_concepts"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Video:
    """Represents a YouTube video from Smart Money playlists"""

    id: str = field(default_factory=lambda: str(uuid4()))
    youtube_id: str = ""
    title: str = ""
    description: str = ""
    duration_seconds: int = 0
    published_at: Optional[datetime] = None
    playlist_id: str = ""
    playlist_name: str = ""
    playlist_order: int = 0  # Order within playlist for sequential learning

    # Processing status
    status: VideoStatus = VideoStatus.PENDING
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Quality metrics
    transcript_quality_score: float = 0.0
    concept_extraction_score: float = 0.0

    # Paths
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "youtube_id": self.youtube_id,
            "title": self.title,
            "description": self.description,
            "duration_seconds": self.duration_seconds,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "playlist_id": self.playlist_id,
            "playlist_name": self.playlist_name,
            "playlist_order": self.playlist_order,
            "status": self.status.value,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "transcript_quality_score": self.transcript_quality_score,
            "concept_extraction_score": self.concept_extraction_score,
        }


@dataclass
class TranscriptSegment:
    """A segment of a video transcript with timestamps"""

    id: str = field(default_factory=lambda: str(uuid4()))
    video_id: str = ""
    start_time: float = 0.0  # seconds
    end_time: float = 0.0
    text: str = ""
    confidence: float = 0.0

    # Word-level timestamps for precise concept linking
    words: List[dict] = field(default_factory=list)

    # Embedding for semantic search
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class Transcript:
    """Complete transcript for a video"""

    id: str = field(default_factory=lambda: str(uuid4()))
    video_id: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    language: str = "en"

    # Quality metrics
    avg_confidence: float = 0.0
    word_count: int = 0

    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_text_at_time(self, time_seconds: float) -> Optional[TranscriptSegment]:
        """Get transcript segment at a specific time"""
        for segment in self.segments:
            if segment.start_time <= time_seconds <= segment.end_time:
                return segment
        return None

    def get_text_range(self, start: float, end: float) -> str:
        """Get transcript text within a time range"""
        texts = []
        for segment in self.segments:
            if segment.start_time >= start and segment.end_time <= end:
                texts.append(segment.text)
        return " ".join(texts)


@dataclass
class Playlist:
    """Represents an Smart Money YouTube playlist"""

    id: str = field(default_factory=lambda: str(uuid4()))
    youtube_playlist_id: str = ""
    name: str = ""
    description: str = ""
    url: str = ""

    # Learning tier (1=Foundation, 2=Core, 3=Advanced, 4=Mastery)
    learning_tier: int = 1
    processing_order: int = 0  # Order to process playlists

    # Stats
    total_videos: int = 0
    processed_videos: int = 0

    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def progress_percentage(self) -> float:
        if self.total_videos == 0:
            return 0.0
        return (self.processed_videos / self.total_videos) * 100
