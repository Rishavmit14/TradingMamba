# TradingMamba - Claude Code Project Instructions

## MANDATORY: Video Training Workflow

**When the user asks to train ANY video (Forex Minions or any playlist), ALWAYS follow the Claude Code Expert Training path. NEVER use the MLX-VLM fallback.**

### The Correct Training Flow (How Video 1 DabKey96qmE Was Trained)

**Phase 0-3 (Automated):** Run `prepare_for_claude_code()` to download, extract frames, transcribe, and detect teaching units.

```python
from backend.app.ml.audio_first_learning import AudioFirstTrainer
trainer = AudioFirstTrainer(data_dir='data')
result = trainer.prepare_for_claude_code(video_id)
```

**Phase 4-5 (Claude Code Expert):** YOU (Claude Code) must:
1. **Read** the transcript at `data/transcripts/{video_id}.json`
2. **View** key frames at `data/video_frames/{video_id}/` (use the Read tool on .jpg files)
3. **Analyze** what ICT/SMC concepts are being taught with expert-level vision analysis
4. **Write** the knowledge base to `data/audio_first_training/{video_id}_knowledge_base.json`
5. **Write** the summary to `data/audio_first_training/{video_id}_knowledge_summary.md`

### Knowledge Base Format

Must match the schema used by Video 1 (`DabKey96qmE_knowledge_base.json`):
- `generation_method`: MUST be `"Claude Code expert analysis"`
- `training_type` in metadata: MUST be `"claude_code_expert"`
- Each concept needs: `llm_summary`, `statistics`, `teaching_types`, `key_rules`, `visual_evidence`
- Include `visual_evidence` with specific frame references from your vision analysis

### NEVER Do This
- NEVER use `train_from_url()` — this uses MLX-VLM (low quality)
- NEVER use `train_remaining.py` without `--claude-code` mode
- NEVER skip the vision analysis step (viewing frames)
- NEVER generate knowledge bases without reading the transcript AND viewing frames

### Forex Minions Playlist Videos

Playlist ID: `PLLxESps7ndeVQ1yoXC1QSEvCHLlTafB_d`

| # | Video ID | Title | Status |
|---|----------|-------|--------|
| 01 | DabKey96qmE | Structure Mapping | TRAINED |
| 02 | BtCIrLqNKe4 | Liquidity & Inducement | PENDING |
| 03 | f9rg4BDaaXE | Pullback & Valid Inducement | PENDING |
| 04 | E1AgOEk-lfM | Inducement Shift & Traps | PENDING |
| 05 | GunkTVpUccM | Break of Structure | PENDING |
| 06 | Yq-Tw3PEU5U | BOS vs Liquidity Sweep | PENDING |
| 07 | NbhVSLd18YM | CHoCH & Structure Mapping | PENDING |
| 08 | evng_upluR0 | High Prob Inducement | PENDING |
| 09 | eoL_y_6ODLk | Fake CHoCH | PENDING |
| 10 | HEq0YzT19kI | CHoCH Confirmation | PENDING |
| 11 | G-pD_Ts4UEE | Price Cycle Theory | PENDING |
| 12 | gSyIFHd3HeE | Premium & Discount Zones | PENDING |
| 13 | hMb-cEAVKcQ | Fair Value Gap | PENDING |
| 14 | -zPGWtuuWdU | Valid Order Blocks | PENDING |
| 15 | hdnldU2yQMw | Millions Dollar Setup | PENDING |
| 16 | hRuUCLE7i6U | Candlestick & Sessions | PENDING |

### Quality Checklist (Before Marking a Video as Trained)
- [ ] `generation_method` = `"Claude Code expert analysis"`
- [ ] Transcript was read and analyzed
- [ ] Key frames were viewed (vision analysis done by Claude Code)
- [ ] All ICT/SMC concepts from the video are captured
- [ ] Each concept has structured `llm_summary` with Definition, Rules, Identification
- [ ] `visual_evidence` array references actual frames viewed
- [ ] `statistics` reflect real teaching content
- [ ] Knowledge base JSON is valid and follows the schema
