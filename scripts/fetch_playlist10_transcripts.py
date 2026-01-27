import sys
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')
import json

from youtube_transcript_api import YouTubeTranscriptApi

video_ids = [
    "6oQVb0zVxMM",
    "npL3ZXJ5zOU",
    "PIYh0CxoY9c",
    "FOUzW0QmsfI",
    "UrS-mtGHtAA",
    "oALYX0HCSYw",
    "X5pQjfkAUCI",
    "glu98jAH8vE",
    "shPGUz9pU-A",
    "HTQgH11W37o",
    "Gnw54f9v6SA",
    "FgacYSN9QEo",
    "Xae0VrbkyFk",
    "owq30ATPU5s"
]

ytt_api = YouTubeTranscriptApi()

success_count = 0
fail_count = 0
results = []

for video_id in video_ids:
    try:
        result = ytt_api.fetch(video_id)
        # Convert to list of dicts
        transcript_data = []
        for snippet in result:
            transcript_data.append({
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration
            })

        # Save to file
        output_path = f"/tmp/TradingMamba/data/transcripts/{video_id}.json"
        with open(output_path, 'w') as f:
            json.dump(transcript_data, f, indent=2)

        success_count += 1
        results.append(f"SUCCESS: {video_id}")
        print(f"SUCCESS: {video_id} - {len(transcript_data)} segments")
    except Exception as e:
        fail_count += 1
        results.append(f"FAILED: {video_id} - {str(e)}")
        print(f"FAILED: {video_id} - {str(e)}")

print("\n" + "="*50)
print(f"SUMMARY: {success_count} succeeded, {fail_count} failed")
print("="*50)
