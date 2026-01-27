#!/usr/bin/env python3
"""
Extract Smart Money Concepts from Transcripts (FREE)

Uses keyword matching and pattern recognition to extract
Smart Money concepts from video transcripts without requiring AI APIs.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CONCEPTS_DIR = DATA_DIR / "concepts"
CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)


# Smart Money Concept Definitions with keywords and patterns
SMART_MONEY_CONCEPTS = {
    # Market Structure
    "break_of_structure": {
        "name": "Break of Structure (BOS)",
        "category": "market_structure",
        "keywords": ["break of structure", "bos", "structure break", "broke structure", "breaking structure"],
        "patterns": [r"break(?:ing|s)?\s+(?:of\s+)?structure", r"\bbos\b"],
        "description": "Price breaking above/below a previous swing point confirming trend continuation"
    },
    "change_of_character": {
        "name": "Change of Character (CHoCH)",
        "category": "market_structure",
        "keywords": ["change of character", "choch", "character change", "shift in character"],
        "patterns": [r"change\s+(?:of|in)\s+character", r"\bchoch\b"],
        "description": "First sign of potential trend reversal"
    },
    "market_structure": {
        "name": "Market Structure",
        "category": "market_structure",
        "keywords": ["market structure", "structure", "higher high", "higher low", "lower high", "lower low"],
        "patterns": [r"market\s+structure", r"higher\s+high", r"higher\s+low", r"lower\s+high", r"lower\s+low"],
        "description": "The pattern of swing highs and lows that define trend"
    },

    # Order Blocks
    "order_block": {
        "name": "Order Block",
        "category": "key_levels",
        "keywords": ["order block", "orderblock", "ob", "bullish order block", "bearish order block"],
        "patterns": [r"order\s*block", r"\bob\b", r"bullish\s+ob", r"bearish\s+ob"],
        "description": "The last candle before a significant price move representing institutional orders"
    },
    "breaker_block": {
        "name": "Breaker Block",
        "category": "key_levels",
        "keywords": ["breaker block", "breaker", "failed order block"],
        "patterns": [r"breaker\s+block", r"breaker"],
        "description": "An order block that has been mitigated and now acts as opposite zone"
    },
    "mitigation_block": {
        "name": "Mitigation Block",
        "category": "key_levels",
        "keywords": ["mitigation block", "mitigation", "mitigate", "mitigated"],
        "patterns": [r"mitigation\s+block", r"mitigat(?:e|ed|ion)"],
        "description": "A zone where price returns to fill orders"
    },

    # Fair Value Gap
    "fair_value_gap": {
        "name": "Fair Value Gap (FVG)",
        "category": "key_levels",
        "keywords": ["fair value gap", "fvg", "imbalance", "inefficiency", "gap"],
        "patterns": [r"fair\s+value\s+gap", r"\bfvg\b", r"imbalance", r"inefficiency"],
        "description": "A gap in price representing imbalance"
    },

    # Liquidity
    "liquidity": {
        "name": "Liquidity",
        "category": "liquidity",
        "keywords": ["liquidity", "stops", "stop loss", "stop hunt", "liquidity pool"],
        "patterns": [r"liquidity", r"stop\s+(?:loss|hunt)", r"liquidity\s+pool"],
        "description": "Areas where stop losses are clustered"
    },
    "buy_side_liquidity": {
        "name": "Buy Side Liquidity",
        "category": "liquidity",
        "keywords": ["buy side liquidity", "bsl", "buy stops", "liquidity above"],
        "patterns": [r"buy\s+side\s+liquidity", r"\bbsl\b", r"buy\s+stops"],
        "description": "Stop losses above swing highs"
    },
    "sell_side_liquidity": {
        "name": "Sell Side Liquidity",
        "category": "liquidity",
        "keywords": ["sell side liquidity", "ssl", "sell stops", "liquidity below"],
        "patterns": [r"sell\s+side\s+liquidity", r"\bssl\b", r"sell\s+stops"],
        "description": "Stop losses below swing lows"
    },
    "liquidity_sweep": {
        "name": "Liquidity Sweep",
        "category": "liquidity",
        "keywords": ["liquidity sweep", "sweep", "raid", "stop raid", "liquidity grab"],
        "patterns": [r"liquidity\s+sweep", r"sweep", r"stop\s+raid", r"liquidity\s+grab"],
        "description": "When price takes out stop losses before reversing"
    },
    "equal_highs_lows": {
        "name": "Equal Highs/Lows",
        "category": "liquidity",
        "keywords": ["equal highs", "equal lows", "double top", "double bottom", "triple top"],
        "patterns": [r"equal\s+highs?", r"equal\s+lows?", r"double\s+(?:top|bottom)"],
        "description": "Multiple swing points at similar levels creating liquidity"
    },

    # Premium/Discount
    "premium_discount": {
        "name": "Premium/Discount",
        "category": "institutional",
        "keywords": ["premium", "discount", "premium zone", "discount zone", "equilibrium"],
        "patterns": [r"premium(?:\s+zone)?", r"discount(?:\s+zone)?", r"equilibrium"],
        "description": "Price zones relative to the 50% of a range"
    },

    # Entry Models
    "optimal_trade_entry": {
        "name": "Optimal Trade Entry (OTE)",
        "category": "entry_models",
        "keywords": ["optimal trade entry", "ote", "fib", "fibonacci", "62%", "79%", "retracement"],
        "patterns": [r"optimal\s+trade\s+entry", r"\bote\b", r"fibonacci", r"(?:62|79)\s*%"],
        "description": "Entry at 62-79% Fibonacci retracement"
    },
    "power_of_three": {
        "name": "Power of Three (AMD)",
        "category": "entry_models",
        "keywords": ["power of three", "amd", "accumulation", "manipulation", "distribution"],
        "patterns": [r"power\s+of\s+three", r"\bamd\b", r"accumulation.*manipulation.*distribution"],
        "description": "Three-phase market cycle: Accumulation, Manipulation, Distribution"
    },
    "judas_swing": {
        "name": "Judas Swing",
        "category": "entry_models",
        "keywords": ["judas swing", "judas", "fake move", "false break", "false breakout"],
        "patterns": [r"judas\s+swing", r"judas", r"fake\s+(?:move|out)", r"false\s+break"],
        "description": "False move to trap traders before the real move"
    },
    "silver_bullet": {
        "name": "Silver Bullet",
        "category": "entry_models",
        "keywords": ["silver bullet", "10am", "11am"],
        "patterns": [r"silver\s+bullet"],
        "description": "Time-based entry model during NY session"
    },

    # Time-Based
    "kill_zone": {
        "name": "Kill Zone",
        "category": "time_based",
        "keywords": ["kill zone", "killzone", "trading session"],
        "patterns": [r"kill\s*zone"],
        "description": "Specific time windows with highest probability"
    },
    "asian_session": {
        "name": "Asian Session/Range",
        "category": "time_based",
        "keywords": ["asian session", "asian range", "asia", "tokyo session"],
        "patterns": [r"asian\s+(?:session|range)", r"tokyo"],
        "description": "Trading session from ~7pm-2am EST"
    },
    "london_session": {
        "name": "London Session",
        "category": "time_based",
        "keywords": ["london session", "london open", "london"],
        "patterns": [r"london\s+(?:session|open|close)"],
        "description": "Trading session from ~2am-5am EST"
    },
    "new_york_session": {
        "name": "New York Session",
        "category": "time_based",
        "keywords": ["new york session", "ny session", "new york open", "ny open"],
        "patterns": [r"new\s+york\s+(?:session|open)", r"ny\s+(?:session|open)"],
        "description": "Trading session from ~7am-10am EST"
    },
    "ipda": {
        "name": "IPDA Data Range",
        "category": "time_based",
        "keywords": ["ipda", "interbank", "20 day", "40 day", "60 day"],
        "patterns": [r"\bipda\b", r"interbank", r"(?:20|40|60)\s*day"],
        "description": "Interbank Price Delivery Algorithm ranges"
    },

    # Institutional
    "smart_money": {
        "name": "Smart Money",
        "category": "institutional",
        "keywords": ["smart money", "institutional", "banks", "market makers"],
        "patterns": [r"smart\s+money", r"institutional", r"market\s+makers?"],
        "description": "Institutional trading activity"
    },
    "displacement": {
        "name": "Displacement",
        "category": "institutional",
        "keywords": ["displacement", "impulse", "aggressive move", "strong move"],
        "patterns": [r"displacement", r"impulse\s+move", r"aggressive\s+move"],
        "description": "Strong price movement showing institutional involvement"
    },
    "propulsion_block": {
        "name": "Propulsion Block",
        "category": "institutional",
        "keywords": ["propulsion block", "propulsion"],
        "patterns": [r"propulsion\s+block", r"propulsion"],
        "description": "Strong institutional candle that creates momentum"
    },

    # Risk Management
    "stop_loss": {
        "name": "Stop Loss Placement",
        "category": "risk_management",
        "keywords": ["stop loss", "stop placement", "risk", "risk management"],
        "patterns": [r"stop\s+loss", r"stop\s+placement", r"risk\s+management"],
        "description": "Where to place protective stops"
    },
    "position_sizing": {
        "name": "Position Sizing",
        "category": "risk_management",
        "keywords": ["position size", "lot size", "risk per trade", "position sizing"],
        "patterns": [r"position\s+siz(?:e|ing)", r"lot\s+size", r"risk\s+per\s+trade"],
        "description": "How to size positions based on risk"
    },
}


def find_concept_mentions(text: str, concept_id: str, concept_data: dict) -> List[dict]:
    """Find all mentions of a concept in text"""
    mentions = []
    text_lower = text.lower()

    # Search by keywords
    for keyword in concept_data.get('keywords', []):
        keyword_lower = keyword.lower()
        start = 0
        while True:
            pos = text_lower.find(keyword_lower, start)
            if pos == -1:
                break

            # Get context (100 chars before and after)
            context_start = max(0, pos - 100)
            context_end = min(len(text), pos + len(keyword) + 100)
            context = text[context_start:context_end]

            mentions.append({
                'concept_id': concept_id,
                'keyword': keyword,
                'position': pos,
                'context': f"...{context}...",
                'match_type': 'keyword'
            })

            start = pos + 1

    # Search by regex patterns
    for pattern in concept_data.get('patterns', []):
        try:
            for match in re.finditer(pattern, text_lower):
                pos = match.start()

                # Check if this position already found by keyword
                if any(abs(m['position'] - pos) < 20 for m in mentions):
                    continue

                context_start = max(0, pos - 100)
                context_end = min(len(text), match.end() + 100)
                context = text[context_start:context_end]

                mentions.append({
                    'concept_id': concept_id,
                    'keyword': match.group(),
                    'position': pos,
                    'context': f"...{context}...",
                    'match_type': 'pattern'
                })
        except re.error:
            pass

    return mentions


def extract_concepts_from_transcript(transcript: dict) -> dict:
    """Extract all Smart Money concepts from a transcript"""
    text = transcript.get('full_text', '')
    segments = transcript.get('segments', [])

    all_mentions = []
    concept_counts = defaultdict(int)
    category_counts = defaultdict(int)

    # Find mentions for each concept
    for concept_id, concept_data in SMART_MONEY_CONCEPTS.items():
        mentions = find_concept_mentions(text, concept_id, concept_data)

        if mentions:
            concept_counts[concept_id] = len(mentions)
            category_counts[concept_data['category']] += len(mentions)

            # Add segment timing for mentions
            for mention in mentions:
                # Find which segment this mention is in
                char_pos = mention['position']
                current_pos = 0
                mention['start_time'] = 0
                mention['end_time'] = 0

                for seg in segments:
                    seg_len = len(seg.get('text', '')) + 1
                    if current_pos <= char_pos < current_pos + seg_len:
                        mention['start_time'] = seg.get('start_time', 0)
                        mention['end_time'] = seg.get('end_time', 0)
                        break
                    current_pos += seg_len

                all_mentions.append(mention)

    # Sort concepts by frequency
    top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:15]

    return {
        'video_id': transcript.get('video_id'),
        'title': transcript.get('title'),
        'word_count': transcript.get('word_count', 0),
        'total_mentions': len(all_mentions),
        'unique_concepts': len([c for c, count in concept_counts.items() if count > 0]),
        'concept_counts': dict(concept_counts),
        'category_counts': dict(category_counts),
        'top_concepts': [
            {
                'id': c,
                'name': SMART_MONEY_CONCEPTS[c]['name'],
                'count': count,
                'category': SMART_MONEY_CONCEPTS[c]['category']
            }
            for c, count in top_concepts
        ],
        'mentions': all_mentions[:100],  # Limit to 100 sample mentions
        'extracted_at': datetime.utcnow().isoformat()
    }


def process_all_transcripts():
    """Process all available transcripts"""
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))

    if not transcript_files:
        print("No transcripts found!")
        return

    print(f"\n{'='*60}")
    print(f"SMART MONEY CONCEPT EXTRACTION")
    print(f"{'='*60}")
    print(f"Transcripts: {len(transcript_files)}")
    print(f"Concepts to find: {len(SMART_MONEY_CONCEPTS)}")
    print(f"{'='*60}\n")

    all_results = []
    grand_totals = {
        'total_mentions': 0,
        'total_words': 0,
        'concept_counts': defaultdict(int),
        'category_counts': defaultdict(int)
    }

    for i, tf in enumerate(transcript_files, 1):
        with open(tf) as f:
            transcript = json.load(f)

        title = transcript.get('title', 'Unknown')[:45]
        print(f"[{i}/{len(transcript_files)}] {title}...")

        result = extract_concepts_from_transcript(transcript)

        # Save individual result
        output_path = CONCEPTS_DIR / f"{transcript['video_id']}_concepts.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Update totals
        grand_totals['total_mentions'] += result['total_mentions']
        grand_totals['total_words'] += result.get('word_count', 0)
        for c, count in result['concept_counts'].items():
            grand_totals['concept_counts'][c] += count
        for cat, count in result['category_counts'].items():
            grand_totals['category_counts'][cat] += count

        all_results.append(result)
        print(f"  Found {result['total_mentions']} mentions of {result['unique_concepts']} concepts")

    # Summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Videos processed: {len(all_results)}")
    print(f"Total words analyzed: {grand_totals['total_words']:,}")
    print(f"Total concept mentions: {grand_totals['total_mentions']:,}")

    # Top concepts overall
    top_overall = sorted(grand_totals['concept_counts'].items(), key=lambda x: -x[1])[:10]
    print(f"\n📊 TOP 10 SMART MONEY CONCEPTS FOUND:")
    for concept_id, count in top_overall:
        name = SMART_MONEY_CONCEPTS[concept_id]['name']
        print(f"  {count:5} - {name}")

    # By category
    print(f"\n📁 MENTIONS BY CATEGORY:")
    for cat, count in sorted(grand_totals['category_counts'].items(), key=lambda x: -x[1]):
        print(f"  {count:5} - {cat}")

    # Save summary
    summary = {
        'processed_at': datetime.utcnow().isoformat(),
        'videos_processed': len(all_results),
        'total_words': grand_totals['total_words'],
        'total_mentions': grand_totals['total_mentions'],
        'concept_counts': dict(grand_totals['concept_counts']),
        'category_counts': dict(grand_totals['category_counts']),
        'top_concepts': [
            {'id': c, 'name': SMART_MONEY_CONCEPTS[c]['name'], 'count': count}
            for c, count in top_overall
        ]
    }

    summary_path = CONCEPTS_DIR / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Summary saved: {summary_path}")

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract Smart Money concepts from transcripts')
    parser.add_argument('--video', type=str, help='Process single video by ID')
    parser.add_argument('--summary', action='store_true', help='Show extraction summary')

    args = parser.parse_args()

    if args.summary:
        summary_path = CONCEPTS_DIR / "extraction_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"\n📊 Smart Money Concept Extraction Summary")
            print(f"{'='*40}")
            print(f"Videos: {summary['videos_processed']}")
            print(f"Words: {summary['total_words']:,}")
            print(f"Mentions: {summary['total_mentions']:,}")
            print(f"\nTop Concepts:")
            for c in summary.get('top_concepts', [])[:10]:
                print(f"  {c['count']:5} - {c['name']}")
        else:
            print("No summary found. Run extraction first.")
        return

    if args.video:
        transcript_path = TRANSCRIPTS_DIR / f"{args.video}.json"
        if transcript_path.exists():
            with open(transcript_path) as f:
                transcript = json.load(f)
            result = extract_concepts_from_transcript(transcript)
            print(f"\nConcepts found: {result['unique_concepts']}")
            print(f"Total mentions: {result['total_mentions']}")
            for c in result.get('top_concepts', [])[:5]:
                print(f"  {c['count']:3} - {c['name']}")
        else:
            print(f"Transcript not found: {args.video}")
        return

    # Process all
    process_all_transcripts()


if __name__ == "__main__":
    main()
