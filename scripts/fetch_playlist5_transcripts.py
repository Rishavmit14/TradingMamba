import sys
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')
import json
import os

from youtube_transcript_api import YouTubeTranscriptApi

# Playlist 5 videos
videos = [
    {"video_id": "mjVHmE1gVMg", "title": "ICT Mentorship Core Content - Month 02   Growing Small Accounts"},
    {"video_id": "Zsg8IeBtfu0", "title": "ICT Mentorship - Core Content -  Month 02 - Framing Low Risk Trade Setups"},
    {"video_id": "pctqB3UD6dk", "title": "ICT Mentorship - Core Content - Month 02 - How Traders Make 10% Per Month"},
    {"video_id": "pFdW8wdR9sQ", "title": "ICT Mentorship Core Content - Month 02 - No Fear Of Losing"},
    {"video_id": "vWDElb65YHg", "title": "ICT Mentorship Core Content - Month 02 - How To Mitigate Losing Trades Effectively"},
    {"video_id": "bftKgceXqYo", "title": "ICT Mentorship Core Content - Month 02 - The Secrets To Selecting High Reward Setups"},
    {"video_id": "cRbPS3uxkj4", "title": "ICT Mentorship Core Content - Month 02 - Market Maker Trap False Flag"},
    {"video_id": "pv2-R-STviA", "title": "ICT Mentorship Core Content - Month 02 - Market Maker Trap False Breakouts"},
]

transcripts_dir = "/tmp/TradingMamba/data/transcripts"
ytt_api = YouTubeTranscriptApi()

success_count = 0
fail_count = 0
results = []

for video in videos:
    video_id = video["video_id"]
    title = video["title"]
    transcript_path = os.path.join(transcripts_dir, f"{video_id}.json")

    # Check if transcript already exists
    if os.path.exists(transcript_path):
        print(f"SKIP: {video_id} - Already exists")
        results.append({"video_id": video_id, "status": "skipped", "reason": "already exists"})
        continue

    try:
        # Fetch transcript
        transcript_data = ytt_api.fetch(video_id)

        # Build the segments and full text
        segments = []
        full_text_parts = []

        for snippet in transcript_data.snippets:
            segments.append({
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration
            })
            full_text_parts.append(snippet.text)

        full_text = " ".join(full_text_parts)
        word_count = len(full_text.split())

        # Create output structure
        output = {
            "video_id": video_id,
            "title": title,
            "full_text": full_text,
            "segments": segments,
            "word_count": word_count,
            "method": "youtube_transcript_api"
        }

        # Save to file
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"SUCCESS: {video_id} - {title[:50]}... ({word_count} words)")
        success_count += 1
        results.append({"video_id": video_id, "status": "success", "word_count": word_count})

    except Exception as e:
        print(f"FAILED: {video_id} - {title[:50]}... - Error: {str(e)}")
        fail_count += 1
        results.append({"video_id": video_id, "status": "failed", "error": str(e)})

print("\n" + "="*60)
print(f"SUMMARY: {success_count} successful, {fail_count} failed out of {len(videos)} videos")
print("="*60)

for r in results:
    if r["status"] == "success":
        print(f"  [OK] {r['video_id']} ({r['word_count']} words)")
    elif r["status"] == "failed":
        print(f"  [FAIL] {r['video_id']}: {r['error'][:60]}")
    else:
        print(f"  [SKIP] {r['video_id']}: {r['reason']}")
