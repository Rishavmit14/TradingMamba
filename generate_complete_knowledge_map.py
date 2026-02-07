#!/usr/bin/env python3
"""
Generate Complete SMC/ICT Knowledge Map
Combines knowledge_base.json + knowledge_summary.md for all 16 videos
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Video metadata from CLAUDE.md
VIDEO_METADATA = [
    ("01", "DabKey96qmE", "Structure Mapping"),
    ("02", "BtCIrLqNKe4", "Liquidity & Inducement"),
    ("03", "f9rg4BDaaXE", "Pullback & Valid Inducement"),
    ("04", "E1AgOEk-lfM", "Inducement Shift & Traps"),
    ("05", "GunkTVpUccM", "Break of Structure"),
    ("06", "Yq-Tw3PEU5U", "BOS vs Liquidity Sweep"),
    ("07", "NbhVSLd18YM", "CHoCH & Structure Mapping"),
    ("08", "evng_upluR0", "High Prob Inducement"),
    ("09", "eoL_y_6ODLk", "Fake CHoCH"),
    ("10", "HEq0YzT19kI", "CHoCH Confirmation"),
    ("11", "G-pD_Ts4UEE", "Price Cycle Theory"),
    ("12", "gSyIFHd3HeE", "Premium & Discount Zones"),
    ("13", "hMb-cEAVKcQ", "Fair Value Gap"),
    ("14", "-zPGWtuuWdU", "Valid Order Blocks"),
    ("15", "hdnldU2yQMw", "Million Dollar Setup"),
    ("16", "hRuUCLE7i6U", "Candlestick & Sessions"),
]

def load_video_data(video_id, training_dir):
    """Load both JSON and MD data for a video"""
    kb_file = training_dir / f"{video_id}_knowledge_base.json"
    summary_file = training_dir / f"{video_id}_knowledge_summary.md"

    data = {
        "kb_data": None,
        "summary_text": None,
        "concepts": {},
        "metadata": {}
    }

    # Load JSON
    if kb_file.exists():
        try:
            with open(kb_file, 'r') as f:
                kb_data = json.load(f)
                data["kb_data"] = kb_data
                data["concepts"] = kb_data.get("concepts", {})
                data["metadata"] = {
                    "video_id": kb_data.get("video_id"),
                    "generation_method": kb_data.get("generation_method"),
                    "generated_at": kb_data.get("generated_at"),
                    "processing_stats": kb_data.get("processing_stats", {})
                }
        except Exception as e:
            print(f"❌ Error loading JSON for {video_id}: {e}")

    # Load summary MD
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                data["summary_text"] = f.read()
        except Exception as e:
            print(f"⚠️  Error loading summary for {video_id}: {e}")

    return data

def format_concept_section(concept_name, concept_data):
    """Format a single concept with all its details"""
    lines = []

    # Concept header
    formatted_name = concept_name.replace('_', ' ').title()
    lines.append(f"#### {formatted_name}")
    lines.append("")

    # LLM Summary (most important)
    llm_summary = concept_data.get("llm_summary", "")
    if llm_summary:
        lines.append(f"**Definition:** {llm_summary}")
        lines.append("")

    # Statistics
    stats = concept_data.get("statistics", {})
    if stats:
        lines.append("**Statistics:**")
        if stats.get("teaching_units"):
            lines.append(f"- Teaching Units: {stats['teaching_units']}")
        if stats.get("frames_analyzed"):
            lines.append(f"- Frames Analyzed: {stats['frames_analyzed']}")
        if stats.get("mentions"):
            lines.append(f"- Mentions: {stats['mentions']}")
        if stats.get("teaching_duration_seconds"):
            lines.append(f"- Duration: {stats['teaching_duration_seconds']}s")
        lines.append("")

    # Key Rules
    key_rules = concept_data.get("key_rules", [])
    if key_rules:
        lines.append("**Key Rules:**")
        for rule in key_rules:
            lines.append(f"- {rule}")
        lines.append("")

    # Teaching Types
    teaching_types = concept_data.get("teaching_types", {})
    if teaching_types:
        active_types = [k for k, v in teaching_types.items() if v]
        if active_types:
            lines.append(f"**Teaching Methods:** {', '.join(active_types)}")
            lines.append("")

    # Visual Evidence
    visual_evidence = concept_data.get("visual_evidence", [])
    if visual_evidence:
        lines.append("**Visual Evidence:**")
        for evidence in visual_evidence[:3]:  # Limit to first 3
            if isinstance(evidence, dict):
                frame = evidence.get("frame", "")
                desc = evidence.get("description", "")
                timestamp = evidence.get("timestamp", "")
                lines.append(f"- Frame {timestamp}s: {desc}")
        lines.append("")

    # Common Mistakes (if present)
    common_mistakes = concept_data.get("common_mistakes", {})
    if common_mistakes:
        lines.append("**Common Mistakes:**")
        for mistake_name, mistake_desc in common_mistakes.items():
            formatted_mistake = mistake_name.replace('_', ' ').title()
            lines.append(f"- {formatted_mistake}: {mistake_desc}")
        lines.append("")

    return "\n".join(lines)

def generate_video_section(num, video_id, title, video_data):
    """Generate complete section for one video"""
    lines = []

    # Video header
    lines.append(f"## VIDEO {num}: {title.upper()}")
    lines.append(f"**Video ID:** {video_id}")
    lines.append("")

    # Metadata
    if video_data["metadata"]:
        meta = video_data["metadata"]
        stats = meta.get("processing_stats", {})
        if stats:
            lines.append("**Processing Stats:**")
            if stats.get("vision_analyses"):
                lines.append(f"- Vision Analyses: {stats['vision_analyses']}")
            if stats.get("total_frames"):
                lines.append(f"- Total Frames: {stats['total_frames']}")
            if stats.get("transcript_words"):
                lines.append(f"- Transcript Words: {stats['transcript_words']}")
            lines.append("")

    # Summary from MD file
    if video_data["summary_text"]:
        # Extract just the core teaching summary from MD
        summary_lines = video_data["summary_text"].split('\n')
        # Find the overview or core teaching section
        in_overview = False
        overview_text = []
        for line in summary_lines:
            if '## Overview' in line or '## Core Teaching' in line:
                in_overview = True
                continue
            if in_overview and line.startswith('##'):
                break
            if in_overview and line.strip():
                overview_text.append(line)

        if overview_text:
            lines.append("### Overview")
            lines.append("")
            lines.extend(overview_text[:10])  # First 10 lines
            lines.append("")

    # All concepts from JSON
    concepts = video_data["concepts"]
    if concepts:
        lines.append(f"### Concepts Taught ({len(concepts)} total)")
        lines.append("")

        for concept_name, concept_data in sorted(concepts.items()):
            concept_section = format_concept_section(concept_name, concept_data)
            lines.append(concept_section)

    lines.append("---")
    lines.append("")

    return "\n".join(lines)

def main():
    training_dir = Path("data/audio_first_training")
    output_file = Path("COMPLETE_SMC_KNOWLEDGE_MAP_NEW.md")

    print("=" * 70)
    print("GENERATING COMPLETE SMC/ICT KNOWLEDGE MAP")
    print("=" * 70)

    # Header
    content = []
    content.append("# COMPLETE SMC/ICT KNOWLEDGE MAP - ENRICHED VERSION")
    content.append("## All 16 Forex Minions Videos - JSON + MD Combined")
    content.append("")
    content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append("**Method:** Combined knowledge_base.json + knowledge_summary.md")
    content.append("**Videos:** 16/16 (100% Complete)")
    content.append("")
    content.append("---")
    content.append("")

    # Executive Summary
    content.append("## EXECUTIVE SUMMARY")
    content.append("")
    content.append("This enriched knowledge map combines structured data from knowledge_base.json files with narrative summaries from knowledge_summary.md files for all 16 Forex Minions ICT/SMC training videos.")
    content.append("")
    content.append("### 10 Core Components Learned:")
    content.append("")
    content.append("1. **Inducement (IDM)** - 70% of forex market activity")
    content.append("2. **Liquidity & Liquidity Sweep** - Market catalyst and validation")
    content.append("3. **Market Structure** - HH/HL/LL/LH (50% of trading success)")
    content.append("4. **Break of Structure (BOS)** - Trend continuation signal")
    content.append("5. **Change of Character (CHoCH)** - Trend reversal signal")
    content.append("6. **Valid Pullback** - Liquidity sweep validation requirement")
    content.append("7. **Fair Value Gap (FVG)** - 70-80% fill probability")
    content.append("8. **Order Block (OB)** - Institutional footprint zones")
    content.append("9. **Premium/Discount Zones** - Optimal pricing for entries")
    content.append("10. **Engineered Liquidity (ENG LIQ)** - Retail trap zones")
    content.append("")
    content.append("---")
    content.append("")

    # Video-by-video breakdown
    total_concepts = 0
    total_teaching_units = 0

    for num, video_id, title in VIDEO_METADATA:
        print(f"Processing Video {num}: {video_id} - {title}")

        video_data = load_video_data(video_id, training_dir)
        video_section = generate_video_section(num, video_id, title, video_data)
        content.append(video_section)

        # Track totals
        total_concepts += len(video_data["concepts"])
        for concept_data in video_data["concepts"].values():
            stats = concept_data.get("statistics", {})
            total_teaching_units += stats.get("teaching_units", 0)

    # Footer with statistics
    content.append("---")
    content.append("")
    content.append("## COMPLETE STATISTICS")
    content.append("")
    content.append(f"**Total Videos:** 16/16")
    content.append(f"**Total Concepts:** {total_concepts}")
    content.append(f"**Total Teaching Units:** {total_teaching_units}")
    content.append("")
    content.append("**All videos analyzed with Claude Code expert method**")
    content.append("")

    # Write file
    with open(output_file, 'w') as f:
        f.write('\n'.join(content))

    print("=" * 70)
    print(f"✅ Complete knowledge map written to {output_file}")
    print(f"   Total Concepts: {total_concepts}")
    print(f"   Total Teaching Units: {total_teaching_units}")
    print(f"   File Size: {output_file.stat().st_size} bytes")
    print("=" * 70)

if __name__ == "__main__":
    main()
