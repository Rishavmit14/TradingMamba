# TradingMamba - Claude Code Project Instructions

## Project Purpose

Build a **new trading system** from scratch, powered by domain knowledge extracted from 23 Hindi SMC/ICT YouTube videos (Forex Minions "Complete Logical SMC" playlist). The system will apply Smart Money Concepts against live market data to detect patterns and generate trading signals.

## Knowledge Foundation (COMPLETED)

All 23 videos have been trained using Claude Code Expert Analysis. The knowledge is stored as structured English knowledge bases derived from Hindi audio source material.

### Playlist: Complete Logical SMC (PLLxESps7ndeWtSZowIo2v0fMmYj4Yiy-a)

| # | Video ID | Title | Status |
|---|----------|-------|--------|
| 01 | UIRBfCT1kI4 | Structure Mapping | TRAINED |
| 02 | DkOv2qG9Cn0 | Liquidity & Inducement | TRAINED |
| 03 | Ovd5QzZutsw | Pullback & Valid Inducement | TRAINED |
| 04 | PYHUHpvKIYo | Inducement Shift & Traps | TRAINED |
| 05 | fTf3pO7F5T8 | Break of Structure | TRAINED |
| 06 | dknxgbZrN3g | BOS vs Liquidity Sweep | TRAINED |
| 07 | VH882sl1pl8 | CHoCH & Structure Mapping | TRAINED |
| 08 | lIbO4JrL_2I | High Prob Inducement | TRAINED |
| 09 | u-LM2DjDd7M | Fake CHoCH | TRAINED |
| 10 | SaTvah4OSpA | CHoCH Confirmation | TRAINED |
| 11 | 3-NsLSV6huY | Price Cycle Theory | TRAINED |
| 12 | anpETX7ahKo | Premium & Discount Zones | TRAINED |
| 13 | 5c14eElh42k | Fair Value Gap | TRAINED |
| 14 | -20GrDt-Aws | Valid Order Blocks | TRAINED |
| 15 | cCugrbID3wI | Million Dollar Setup | TRAINED |
| 16 | 9BfwzGhkOFE | Candlestick & Sessions | TRAINED |
| 17 | exF1jB8qCxo | Complete SMC Guideline (Entry) | TRAINED |
| 18 | PxShy_00yR8 | QML & POI Zones | TRAINED |
| 19 | Kc74F908K-8 | SCOB Entry Technique | TRAINED |
| 20 | JaYMVwyauKA | Multi-Timeframe Analysis | TRAINED |
| 21 | sRXIoNPq85Q | Counter-Trend Trading | TRAINED |
| 22 | newJA7qOzpE | Liquidity Sweep Module (Wyckoff) | TRAINED |
| 23 | vuZqh_oKBp8 | Putting All Together — SMC The End | TRAINED |

### Knowledge Base Location

- **Knowledge bases**: `data/audio_first_training/{video_id}_knowledge_base.json` (23 files)
- **Summaries**: `data/audio_first_training/{video_id}_knowledge_summary.md` (23 files)
- **Transcripts**: `data/transcripts/{video_id}.json` (23 files)
- **Video frames**: `data/video_frames/{video_id}/` (23 directories)
- **Audio files**: `data/audio/` (downloaded .wav files)
- **Playlist data**: `data/playlists/`

### Knowledge Base Schema

Each `*_knowledge_base.json` contains:
- `generation_method`: `"Claude Code expert analysis"`
- `metadata.training_type`: `"claude_code_expert"`
- `metadata.source_language`: `"hi"`
- `concepts`: Array of concepts, each with:
  - `llm_summary`: Structured English explanation (Definition, Rules, Identification)
  - `statistics`: Teaching time, word count, frame count
  - `teaching_types`: Array of teaching methods used
  - `key_rules`: Array of actionable trading rules
  - `visual_evidence`: References to specific analyzed frames

### Core SMC Concepts Covered

The 23 videos progressively build a complete SMC trading system:
1. **Structure**: HH/HL/LH/LL, swing points, trend identification (V01)
2. **Liquidity**: Buy/sell liquidity pools, inducement, traps (V02-V04, V08)
3. **BOS**: Break of Structure rules, BOS vs liquidity sweep (V05-V06)
4. **CHoCH**: Change of Character, fake CHoCH, confirmation (V07, V09-V10)
5. **Market Context**: Price cycle theory, premium/discount zones (V11-V12)
6. **Zones**: FVG, Order Blocks, IOF identification (V13-V14)
7. **Setups**: Million Dollar Setup, sessions/kill zones (V15-V16)
8. **Entries**: MSS/SBC/CHoCH entries, QML, SCOB technique (V17-V19)
9. **Multi-TF**: Higher TF direction + lower TF entry (V20)
10. **Advanced**: Counter-trend rules, Wyckoff/SBC module, full integration (V21-V23)

## Project Structure

```
TradingMamba/
├── CLAUDE.md                          # This file
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── data/
│   ├── audio_first_training/          # 23 KBs + 23 summaries (THE KNOWLEDGE)
│   ├── transcripts/                   # 23 Hindi transcripts
│   ├── video_frames/                  # 23 directories of extracted frames
│   ├── audio/                         # Downloaded audio files
│   └── playlists/                     # Playlist metadata
├── scripts/                           # Transcription & processing utilities
└── models/                            # Whisper model files
```

## Building the New System

When building the new codebase, Claude should:
1. **Read all 23 knowledge bases** to understand the complete SMC framework
2. **Use the key_rules arrays** as the algorithmic foundation for pattern detection
3. **Reference the summaries** for concept relationships and dependencies
4. **Apply the Master Trading Checklist** (from V23) as the top-level system flow
5. **Preserve ICT methodology constants** (Fibonacci levels, ATR multiples, session times)

## Transcription Tools (Preserved)

- **whisper.cpp**: `/opt/homebrew/bin/whisper-cli` with Metal GPU acceleration
- **Model**: `models/ggml-large-v3-turbo.bin`
- **Script**: `scripts/transcribe_local.py --video VIDEO_ID`
- **CRITICAL**: Never run 2+ whisper.cpp instances simultaneously on Metal GPU
