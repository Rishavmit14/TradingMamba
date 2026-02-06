# Claude Code Training Workflow

Primary method for training ML knowledge bases using Claude Code Max plan.

## Overview

Claude Code reads video transcripts + views key frames, then writes expert-quality
`_knowledge_base.json` and `_knowledge_summary.md` files that the backend loads directly.

**Benefits over MLX-VLM fallback:**
- Expert-quality vision analysis (vs generic/repetitive output)
- Expert-quality knowledge summaries with proper ICT terminology
- $0 cost (included in Claude Code Max plan)
- 10x faster (minutes vs hours)
- No paid API keys required

## Quick Start (Recommended)

### Option A: Using Python API

```python
from backend.app.ml.audio_first_learning import prepare_for_claude_code

# Single video
result = prepare_for_claude_code('https://youtube.com/watch?v=VIDEO_ID')
print(result['claude_code_instructions']['prompt'])

# Playlist
from backend.app.ml.audio_first_learning import prepare_playlist_for_claude_code
results = prepare_playlist_for_claude_code('https://youtube.com/playlist?list=PLxxx')
```

### Option B: Using CLI

```bash
# Single video
python -m backend.app.ml.audio_first_learning 'https://youtube.com/watch?v=VIDEO_ID' --claude-code

# Playlist
python -m backend.app.ml.audio_first_learning 'https://youtube.com/playlist?list=PLxxx' --playlist --claude-code
```

This runs Phase 0-3 automatically:
- Downloads audio from YouTube
- Extracts frames at regular intervals
- Transcribes audio (faster-whisper)
- Detects teaching units and selects key frames

Then prints instructions for Claude Code to complete training.

## Training a New Playlist

### Step 1: Prepare videos (automated)

```bash
python -m backend.app.ml.audio_first_learning 'PLAYLIST_URL' --playlist --claude-code
```

This creates for each video:
- `data/transcripts/{video_id}.json` - Full transcript with timestamps
- `data/video_frames/{video_id}/` - Extracted video frames
- `data/audio_first_training/{video_id}_teaching_units.json` - Detected teaching segments
- `data/audio_first_training/{video_id}_selected_frames.json` - Key frames for vision analysis
- `data/audio_first_training/{video_id}_claude_code_pending.json` - Pending marker

### Step 2: Ask Claude Code to complete training

For each video, tell Claude Code:

```
Complete the training for video: {video_id}

1. Read the transcript at data/transcripts/{video_id}.json
2. View these key frames: data/video_frames/{video_id}/frame_0000.jpg, frame_0010.jpg, ...
3. Extract all ICT/SMC concepts taught in this video
4. Write expert knowledge_base.json to data/audio_first_training/{video_id}_knowledge_base.json
5. Write markdown summary to data/audio_first_training/{video_id}_knowledge_summary.md
```

Claude Code will:
- Read the transcript to understand what's being taught
- View the chart images to understand the visual examples
- Extract ICT/SMC concepts with expert understanding
- Write structured knowledge bases with proper terminology

### Step 3: Verify

```bash
# Restart backend
kill -9 $(lsof -ti:8000) 2>/dev/null; cd backend && nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Check patterns loaded
curl -s http://localhost:8000/api/ml/pattern-knowledge | python3 -m json.tool
```

## Knowledge Base JSON Format

```json
{
  "video_id": "VIDEO_ID",
  "generated_at": "2026-02-05T15:45:00.000000",
  "generation_method": "Claude Code expert analysis",
  "total_generation_time_seconds": 0,
  "processing_stats": {
    "teaching_units": 34,
    "vision_analyses": 49,
    "concepts_extracted": 5,
    "total_audio_duration": 721.6,
    "total_words": 1287
  },
  "concepts": {
    "concept_name": {
      "llm_summary": "### Concept Name -- Subtitle\n\n#### Definition\n...\n\n#### Rules\n...",
      "generation_time_seconds": 0,
      "statistics": {
        "teaching_duration_seconds": 300.0,
        "word_count": 600,
        "teaching_units": 16,
        "frames_analyzed": 20,
        "deictic_references": 18
      },
      "teaching_types": {
        "explanation": 6,
        "example": 10
      }
    }
  }
}
```

## Vision Analysis with Claude Code

Claude Code can view video frames directly and provide expert ICT/SMC analysis.

When viewing a chart frame, Claude Code identifies:
- **Market Structure**: Swing highs/lows, BOS, CHoCH
- **Liquidity Pools**: Equal highs/lows, inducement levels
- **Order Blocks**: Valid vs invalid, mitigation status
- **Fair Value Gaps**: Open vs filled, tradeable vs non-tradeable
- **Premium/Discount Zones**: Current price position relative to range

This produces much higher quality vision analysis than MLX-VLM's generic output.

## Backend Fields Used

The ML pattern engine (`ml_pattern_engine.py`) reads these fields:

| Field | Usage |
|-------|-------|
| `concepts.{name}.llm_summary` | Stored as learned_traits, used for reasoning display |
| `concepts.{name}.statistics.teaching_units` | Frequency count + confidence boost (>=3: +0.15) |
| `concepts.{name}.statistics.frames_analyzed` | Confidence boost (>=5: +0.10) |
| `concepts.{name}.statistics.word_count` | Confidence boost (>=200: +0.05) |
| `concepts.{name}.statistics.deictic_references` | Confidence boost (>=5: +0.05) |
| `concepts.{name}.teaching_types` | Stored in learned_traits |
| `generation_method` | "Claude Code expert analysis" triggers +0.10 confidence bonus |

## Confidence Formula

```
base = 0.50
+ 0.15 if teaching_units >= 3
+ 0.10 if frames_analyzed >= 5
+ 0.05 if word_count >= 200
+ 0.05 if deictic_references >= 5
+ 0.10 if generation_method == "Claude Code expert analysis"
max = 0.95
```

## Quality Checklist

- [ ] `generation_method` is set to `"Claude Code expert analysis"`
- [ ] Each concept has a structured `llm_summary` with Definition, Rules, Identification sections
- [ ] Statistics reflect actual teaching content (not zeroed out)
- [ ] Concept names use lowercase with underscores for multi-word
- [ ] No false positive concepts (only include what the video actually teaches)
- [ ] Video ID matches the YouTube video ID exactly

## Checking Pending Videos

To see which videos are pending Claude Code training:

```python
from backend.app.ml.audio_first_learning import AudioFirstTrainer
trainer = AudioFirstTrainer()
pending = trainer.get_pending_claude_code_videos()
for v in pending:
    print(f"Pending: {v['video_id']}")
    print(v['instructions']['prompt'])
```

## Fallback: MLX-VLM Training

If Claude Code is not available, the automated MLX-VLM path still works:

```bash
python -m backend.app.ml.audio_first_learning 'VIDEO_URL'  # Without --claude-code
```

This uses MLX-VLM for vision analysis and knowledge synthesis. Quality is lower
but it runs fully automatically. Knowledge bases generated this way have
`generation_method: "LLM per concept (MLX-VLM)"` and don't get the expert bonus.

## Architecture Notes

### Paid APIs Disabled

The following paid API calls have been disabled by default to maximize Claude Code Max plan usage:

- `concept_extractor.py`: Claude API for concept extraction (set `ENABLE_PAID_CONCEPT_API=true` to enable)
- `video_vision_analyzer.py`: Claude/OpenAI vision APIs (set `ENABLE_PAID_VISION_API=true` to enable)

All concept extraction and vision analysis is now pre-computed by Claude Code and stored in knowledge bases.

### Training Pipeline Phases

| Phase | Description | Handler |
|-------|-------------|---------|
| 0 | Download, extract frames, transcribe | `VideoPreprocessor` (automated) |
| 1-3 | Teaching unit detection, frame selection | `AudioFirstTrainer.train()` (automated) |
| 4 | Vision analysis | Claude Code (expert) or MLX-VLM (fallback) |
| 5 | Knowledge synthesis | Claude Code (expert) or MLX-VLM (fallback) |

The `--claude-code` flag runs Phases 0-3 automatically and generates instructions for Claude Code to complete Phases 4-5.
