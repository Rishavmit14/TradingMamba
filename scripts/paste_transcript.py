#!/usr/bin/env python3
"""
Paste YouTube transcript manually.

Usage: python3 scripts/paste_transcript.py

This script will prompt you for:
1. Video ID (or you can select from the 5 missing ones)
2. Paste the transcript text from YouTube

It will then save it in the correct format.
"""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

MISSING_VIDEOS = [
    ("t6y637OyiDY", "OTE Pattern Recognition Series - Vol. 05"),
    ("ifFDif82Bmg", "OTE Pattern Recognition Series - Vol. 08"),
    ("6D3sKabkozk", "OTE Pattern Recognition Series - Vol. 12"),
    ("cYzS_kFxv7M", "OTE Pattern Recognition Series - Vol. 14"),
    ("FNwqw7IiByo", "OTE Pattern Recognition Series - Vol.17"),
]

def save_transcript(video_id: str, title: str, text: str):
    """Save transcript in the standard format"""
    
    # Clean up the text - remove timestamps if present
    lines = text.strip().split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        # Skip timestamp lines (like "0:00", "1:23", etc.)
        if line and not (len(line) < 10 and ':' in line and line.replace(':', '').replace(' ', '').isdigit()):
            clean_lines.append(line)
    
    full_text = ' '.join(clean_lines)
    full_text = ' '.join(full_text.split())  # Normalize whitespace
    word_count = len(full_text.split())
    
    transcript_data = {
        'video_id': video_id,
        'title': title,
        'full_text': full_text,
        'segments': [],  # No timing info from manual paste
        'word_count': word_count,
        'segment_count': 0,
        'method': 'manual_paste',
        'transcribed_at': datetime.utcnow().isoformat(),
    }
    
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"
    with open(transcript_path, 'w') as f:
        json.dump(transcript_data, f, indent=2)
    
    return word_count, transcript_path

def main():
    print("=" * 60)
    print("Manual Transcript Paster")
    print("=" * 60)
    print("\nMissing videos:")
    for i, (vid, title) in enumerate(MISSING_VIDEOS, 1):
        # Check if already exists
        exists = (TRANSCRIPTS_DIR / f"{vid}.json").exists()
        status = "✓ DONE" if exists else ""
        print(f"  {i}. {title} {status}")
        print(f"     https://www.youtube.com/watch?v={vid}")
    
    print("\n" + "-" * 60)
    choice = input("\nEnter number (1-5) or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        print("Bye!")
        return
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(MISSING_VIDEOS):
            print("Invalid choice")
            return
    except ValueError:
        print("Invalid input")
        return
    
    video_id, title = MISSING_VIDEOS[idx]
    
    print(f"\nSelected: {title}")
    print(f"URL: https://www.youtube.com/watch?v={video_id}")
    print("\nInstructions:")
    print("1. Open the URL in your browser")
    print("2. Click '...more' below the video")
    print("3. Click 'Show transcript'")
    print("4. Select all text (Cmd+A) and copy (Cmd+C)")
    print("5. Paste below and press Enter twice when done:\n")
    
    lines = []
    print("Paste transcript (press Enter twice when done):")
    empty_count = 0
    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break
    
    text = '\n'.join(lines)
    
    if not text.strip():
        print("No text entered. Aborting.")
        return
    
    word_count, path = save_transcript(video_id, title, text)
    
    print(f"\n✓ Saved {word_count} words to {path.name}")
    print("\nRun this script again to add more transcripts.")

if __name__ == "__main__":
    main()
