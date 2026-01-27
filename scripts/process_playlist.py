#!/usr/bin/env python3
"""
Script to process ICT YouTube playlists

This script processes videos from ICT playlists in order,
starting with foundational content and progressing to advanced.

Usage:
    python process_playlist.py --playlist 1 --max-videos 5
    python process_playlist.py --url "https://youtube.com/playlist?list=..."
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.services.video_processor import (
    VideoProcessor,
    PlaylistProcessor,
    ICT_PLAYLISTS
)
from backend.app.services.concept_extractor import ConceptExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def list_playlists():
    """Display available playlists"""
    print("\n📚 Available ICT Playlists:\n")
    print("-" * 80)

    for i, playlist in enumerate(ICT_PLAYLISTS, 1):
        tier_emoji = ["🌱", "🌿", "🌳", "🏆"][playlist['learning_tier'] - 1]
        print(f"{i}. {tier_emoji} [{playlist['learning_tier']}] {playlist['name']}")
        print(f"   📝 {playlist['description']}")
        print(f"   🔗 {playlist['url']}")
        print()

    print("-" * 80)
    print("\nLearning Tiers:")
    print("  🌱 Tier 1: Foundation - Basic concepts")
    print("  🌿 Tier 2: Core - Main ICT methodology")
    print("  🌳 Tier 3: Advanced - Complex strategies")
    print("  🏆 Tier 4: Mastery - Expert level")


def process_single_playlist(
    playlist_url: str,
    learning_tier: int = 1,
    max_videos: int = None,
    extract_concepts: bool = True
):
    """Process a single playlist"""

    # Initialize processors
    video_processor = VideoProcessor()
    playlist_processor = PlaylistProcessor(video_processor)
    concept_extractor = ConceptExtractor(use_ai=True)

    # Get playlist info
    logger.info(f"Fetching playlist info: {playlist_url}")
    playlist_info = playlist_processor.get_playlist_info(playlist_url)

    if not playlist_info:
        logger.error("Failed to get playlist info")
        return

    print(f"\n📋 Playlist: {playlist_info.title}")
    print(f"📊 Videos: {playlist_info.video_count}")
    print(f"🎯 Learning Tier: {learning_tier}")
    print("-" * 60)

    # Process videos
    results = playlist_processor.process_playlist(
        playlist_url,
        learning_tier=learning_tier,
        max_videos=max_videos
    )

    # Summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n✅ Successfully processed: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")

    # Extract concepts from successful transcripts
    if extract_concepts and successful:
        print("\n🧠 Extracting ICT concepts...")

        all_concepts = []
        all_rules = []

        for result in successful:
            if result.transcript:
                extraction = concept_extractor.extract_from_transcript(result.transcript)
                all_concepts.extend(extraction.concepts_found)
                all_rules.extend(extraction.rules_extracted)

                print(f"  📝 {result.video_id}: {len(extraction.concepts_found)} concepts, {len(extraction.rules_extracted)} rules")

        # Concept summary
        summary = concept_extractor.get_concept_summary(all_concepts)

        print(f"\n📊 Concept Extraction Summary:")
        print(f"  Total mentions: {summary['total_mentions']}")
        print(f"  Unique concepts: {summary['unique_concepts']}")
        print(f"  Rules extracted: {len(all_rules)}")

        print(f"\n  Top concepts found:")
        for concept, count in list(summary['by_concept'].items())[:10]:
            print(f"    - {concept}: {count} mentions")

        # Save extraction results
        output_path = Path("data/extractions")
        output_path.mkdir(parents=True, exist_ok=True)

        extraction_file = output_path / f"extraction_{playlist_info.playlist_id}.json"
        with open(extraction_file, 'w') as f:
            json.dump({
                'playlist_id': playlist_info.playlist_id,
                'playlist_title': playlist_info.title,
                'processed_at': datetime.utcnow().isoformat(),
                'videos_processed': len(successful),
                'summary': summary,
                'concepts': [
                    {
                        'concept_id': c.concept_id,
                        'video_id': c.video_id,
                        'start_time': c.start_time,
                        'context': c.context_text[:200] if c.context_text else '',
                        'confidence': c.confidence_score,
                    }
                    for c in all_concepts
                ],
                'rules': [
                    {
                        'concept_id': r.concept_id,
                        'rule_type': r.rule_type,
                        'rule_text': r.rule_text,
                    }
                    for r in all_rules
                ]
            }, f, indent=2)

        print(f"\n💾 Saved extraction results to: {extraction_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Process ICT YouTube playlists for ML training'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available playlists'
    )
    parser.add_argument(
        '--playlist',
        type=int,
        help='Playlist number to process (1-10)'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Custom playlist URL to process'
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to process'
    )
    parser.add_argument(
        '--tier',
        type=int,
        default=1,
        help='Learning tier (1-4)'
    )
    parser.add_argument(
        '--no-concepts',
        action='store_true',
        help='Skip concept extraction'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all playlists in order'
    )

    args = parser.parse_args()

    if args.list:
        list_playlists()
        return

    if args.all:
        print("🚀 Processing all playlists in learning order...\n")
        for playlist in ICT_PLAYLISTS:
            print(f"\n{'='*60}")
            print(f"Processing: {playlist['name']}")
            print(f"{'='*60}")
            process_single_playlist(
                playlist['url'],
                learning_tier=playlist['learning_tier'],
                max_videos=args.max_videos,
                extract_concepts=not args.no_concepts
            )
        return

    if args.playlist:
        if 1 <= args.playlist <= len(ICT_PLAYLISTS):
            playlist = ICT_PLAYLISTS[args.playlist - 1]
            process_single_playlist(
                playlist['url'],
                learning_tier=playlist['learning_tier'],
                max_videos=args.max_videos,
                extract_concepts=not args.no_concepts
            )
        else:
            print(f"❌ Invalid playlist number. Choose 1-{len(ICT_PLAYLISTS)}")
            list_playlists()
        return

    if args.url:
        process_single_playlist(
            args.url,
            learning_tier=args.tier,
            max_videos=args.max_videos,
            extract_concepts=not args.no_concepts
        )
        return

    # Default: show help
    parser.print_help()
    print("\n")
    list_playlists()


if __name__ == "__main__":
    main()
