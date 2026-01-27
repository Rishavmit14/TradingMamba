import json
import os

# Load playlist
with open('/tmp/TradingMamba/data/playlists/PLVgHx4Z63paYBN404Q2QZ7D4mOJz1IHAk.json', 'r') as f:
    playlist = json.load(f)

print(f"Playlist: {playlist['title']}")
print(f"Total videos: {playlist['video_count']}")
print()

# Check which videos have transcripts
transcript_dir = '/tmp/TradingMamba/data/transcripts/'
existing_transcripts = set()
for f in os.listdir(transcript_dir):
    if f.endswith('.json'):
        existing_transcripts.add(f.replace('.json', ''))

missing = []
existing = []

for video in playlist['videos']:
    video_id = video['video_id']
    if video_id in existing_transcripts:
        existing.append((video_id, video['title']))
    else:
        missing.append((video_id, video['title']))

print(f"Videos with transcripts: {len(existing)}")
print(f"Videos missing transcripts: {len(missing)}")
print()

if missing:
    print("Missing videos:")
    for vid, title in missing:
        print(f"  - {vid}: {title}")
else:
    print("All videos have transcripts!")
