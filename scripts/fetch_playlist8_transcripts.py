import sys
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')
import json

from youtube_transcript_api import YouTubeTranscriptApi

# Video IDs from playlist 8
video_ids = [
    "UF5J0nEBc0E",
    "PQkcFbr61FI",
    "t2VNzJZ4hK4",
    "9kabTfUEVKg",
    "GuycI8XubgE",
    "MJwWUd_FM-k",
    "o8NfSK-pUlE",
    "w8lbrvZXUVY"
]

ytt_api = YouTubeTranscriptApi()
success_count = 0
fail_count = 0
results = []

for video_id in video_ids:
    try:
        result = ytt_api.fetch(video_id)
        # Convert to serializable format
        transcript_data = []
        for entry in result:
            transcript_data.append({
                "text": entry.text,
                "start": entry.start,
                "duration": entry.duration
            })

        # Save to file
        output_path = f"/tmp/TradingMamba/data/transcripts/{video_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        success_count += 1
        results.append(f"SUCCESS: {video_id} - {len(transcript_data)} entries")
        print(f"SUCCESS: {video_id} - {len(transcript_data)} entries saved")
    except Exception as e:
        fail_count += 1
        results.append(f"FAILED: {video_id} - {str(e)}")
        print(f"FAILED: {video_id} - {str(e)}")

print(f"\n=== SUMMARY ===")
print(f"Total videos: {len(video_ids)}")
print(f"Successful: {success_count}")
print(f"Failed: {fail_count}")
