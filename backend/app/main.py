"""
Smart Money AI Trading System - FastAPI Application

Main entry point for the backend API.
"""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Set
from datetime import datetime
import json
import asyncio
import os
from pathlib import Path

from .config import settings

# Create FastAPI app
app = FastAPI(
    title="TradingMamba - Smart Money AI Trading System",
    description="AI-powered trading signal system based on Smart Money methodology",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "TradingMamba API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Playlists & Videos
# ============================================================================

@app.get("/api/playlists")
async def get_playlists():
    """Get all Smart Money playlists"""
    playlists = []

    if not PLAYLISTS_DIR.exists():
        return {"playlists": [], "total": 0}

    for playlist_file in sorted(PLAYLISTS_DIR.glob("*.json")):
        with open(playlist_file) as f:
            data = json.load(f)
            playlists.append({
                "id": data.get("playlist_id"),
                "title": data.get("title"),
                "video_count": data.get("video_count"),
                "tier": data.get("tier", 1),
            })

    return {
        "playlists": playlists,
        "total": len(playlists)
    }


@app.get("/api/playlists/{playlist_id}")
async def get_playlist(playlist_id: str):
    """Get a specific playlist with videos"""
    playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not playlist_file.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    with open(playlist_file) as f:
        return json.load(f)


@app.get("/api/playlists/{playlist_id}/videos")
async def get_playlist_videos(playlist_id: str):
    """Get videos in a playlist"""
    playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not playlist_file.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    with open(playlist_file) as f:
        data = json.load(f)
        return {
            "playlist_id": playlist_id,
            "videos": data.get("videos", [])
        }


# ============================================================================
# Transcripts
# ============================================================================

@app.get("/api/transcripts")
async def get_transcripts():
    """Get all available transcripts"""
    transcripts = []

    if not TRANSCRIPTS_DIR.exists():
        return {"transcripts": [], "total": 0}

    for transcript_file in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        with open(transcript_file) as f:
            data = json.load(f)
            transcripts.append({
                "video_id": data.get("video_id"),
                "title": data.get("title"),
                "word_count": data.get("word_count", 0),
                "method": data.get("method"),
                "transcribed_at": data.get("transcribed_at"),
            })

    return {
        "transcripts": transcripts,
        "total": len(transcripts)
    }


@app.get("/api/transcripts/grouped")
async def get_transcripts_grouped_by_playlist():
    """Get all transcripts grouped by their playlist"""
    if not TRANSCRIPTS_DIR.exists():
        return {"playlists": [], "ungrouped": [], "total": 0}

    # Load all playlists to map video_id -> playlist
    playlist_videos = {}  # video_id -> playlist_info
    playlist_info = {}    # playlist_id -> playlist details

    if PLAYLISTS_DIR.exists():
        for playlist_file in sorted(PLAYLISTS_DIR.glob("*.json")):
            with open(playlist_file) as f:
                data = json.load(f)
                playlist_id = data.get("playlist_id")
                playlist_title = data.get("title", "Unknown Playlist")

                playlist_info[playlist_id] = {
                    "id": playlist_id,
                    "title": playlist_title,
                    "tier": data.get("tier", 3),
                    "video_count": data.get("video_count", 0),
                    "transcripts": []
                }

                for video in data.get("videos", []):
                    vid_id = video.get("video_id")
                    if vid_id:
                        playlist_videos[vid_id] = playlist_id

    # Load all transcripts and group them
    ungrouped = []

    for transcript_file in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        with open(transcript_file) as f:
            data = json.load(f)
            video_id = data.get("video_id")

            transcript_info = {
                "video_id": video_id,
                "title": data.get("title"),
                "word_count": data.get("word_count", 0),
                "method": data.get("method"),
                "transcribed_at": data.get("transcribed_at"),
            }

            # Find which playlist this belongs to
            if video_id in playlist_videos:
                playlist_id = playlist_videos[video_id]
                if playlist_id in playlist_info:
                    playlist_info[playlist_id]["transcripts"].append(transcript_info)
            else:
                ungrouped.append(transcript_info)

    # Convert to list and sort by tier, then by title
    playlists = sorted(
        [p for p in playlist_info.values() if p["transcripts"]],
        key=lambda x: (x["tier"], x["title"])
    )

    # Add transcript counts
    for p in playlists:
        p["transcript_count"] = len(p["transcripts"])
        # Sort transcripts by date
        p["transcripts"] = sorted(
            p["transcripts"],
            key=lambda x: x.get("transcribed_at") or "",
            reverse=True
        )

    total_transcripts = sum(len(p["transcripts"]) for p in playlists) + len(ungrouped)

    return {
        "playlists": playlists,
        "ungrouped": ungrouped,
        "total": total_transcripts
    }


@app.get("/api/transcripts/{video_id}")
async def get_transcript(video_id: str):
    """Get transcript for a specific video"""
    transcript_file = TRANSCRIPTS_DIR / f"{video_id}.json"

    if not transcript_file.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    with open(transcript_file) as f:
        return json.load(f)


@app.get("/api/transcripts/{video_id}/search")
async def search_transcript(
    video_id: str,
    q: str = Query(..., description="Search query"),
    context: int = Query(50, description="Characters of context around match")
):
    """Search within a transcript"""
    transcript_file = TRANSCRIPTS_DIR / f"{video_id}.json"

    if not transcript_file.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    with open(transcript_file) as f:
        data = json.load(f)

    full_text = data.get("full_text", "").lower()
    query = q.lower()

    matches = []
    start = 0

    while True:
        pos = full_text.find(query, start)
        if pos == -1:
            break

        # Get context
        context_start = max(0, pos - context)
        context_end = min(len(full_text), pos + len(query) + context)
        context_text = full_text[context_start:context_end]

        # Find timestamp
        timestamp = None
        for seg in data.get("segments", []):
            seg_text = seg.get("text", "").lower()
            if query in seg_text:
                timestamp = seg.get("start_time")
                break

        matches.append({
            "position": pos,
            "context": f"...{context_text}...",
            "timestamp": timestamp
        })

        start = pos + 1

    return {
        "video_id": video_id,
        "query": q,
        "matches": matches[:20],  # Limit to 20 matches
        "total_matches": len(matches)
    }


# ============================================================================
# Smart Money Concepts
# ============================================================================

@app.get("/api/concepts")
async def get_concepts():
    """Get all Smart Money concepts from taxonomy"""
    from .models.concept import SMART_MONEY_CONCEPT_TAXONOMY

    concepts = []
    for category, data in SMART_MONEY_CONCEPT_TAXONOMY.items():
        for concept in data.get("concepts", []):
            concepts.append({
                "category": category,
                "name": concept["name"],
                "short_name": concept.get("short_name", ""),
                "description": concept.get("description", ""),
                "keywords": concept.get("keywords", []),
            })

    return {
        "concepts": concepts,
        "total": len(concepts),
        "categories": list(SMART_MONEY_CONCEPT_TAXONOMY.keys())
    }


@app.get("/api/concepts/{category}")
async def get_concepts_by_category(category: str):
    """Get concepts for a specific category"""
    from .models.concept import SMART_MONEY_CONCEPT_TAXONOMY

    if category not in SMART_MONEY_CONCEPT_TAXONOMY:
        raise HTTPException(status_code=404, detail="Category not found")

    data = SMART_MONEY_CONCEPT_TAXONOMY[category]
    return {
        "category": category,
        "name": data.get("name"),
        "concepts": data.get("concepts", [])
    }


@app.get("/api/concepts/search")
async def search_concepts(q: str = Query(..., description="Search query")):
    """Search for concepts by name or keyword"""
    from .models.concept import SMART_MONEY_CONCEPT_TAXONOMY

    query = q.lower()
    matches = []

    for category, data in SMART_MONEY_CONCEPT_TAXONOMY.items():
        for concept in data.get("concepts", []):
            # Check name
            if query in concept["name"].lower():
                matches.append({**concept, "category": category, "match_type": "name"})
                continue

            # Check short name
            if query in concept.get("short_name", "").lower():
                matches.append({**concept, "category": category, "match_type": "short_name"})
                continue

            # Check keywords
            for keyword in concept.get("keywords", []):
                if query in keyword.lower():
                    matches.append({**concept, "category": category, "match_type": "keyword"})
                    break

    return {
        "query": q,
        "matches": matches,
        "total": len(matches)
    }


# ============================================================================
# ML & Signals
# ============================================================================

# Global ML components (lazy loaded)
_knowledge_base = None
_signal_generator = None
_analyzer = None


def get_knowledge_base():
    """Lazy load knowledge base"""
    global _knowledge_base
    if _knowledge_base is None:
        try:
            from .ml.training_pipeline import SmartMoneyKnowledgeBase
            _knowledge_base = SmartMoneyKnowledgeBase(str(DATA_DIR))
            _knowledge_base.load()
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {e}")
    return _knowledge_base


def get_signal_generator():
    """Lazy load signal generator (legacy - for backwards compatibility)"""
    global _signal_generator
    if _signal_generator is None:
        try:
            from .ml.signal_fusion import SignalGenerator
            _signal_generator = SignalGenerator(str(DATA_DIR))
            _signal_generator.load()
        except Exception as e:
            print(f"Warning: Could not load signal generator: {e}")
    return _signal_generator


def get_analyzer():
    """Lazy load technical analyzer (legacy - for backwards compatibility)"""
    global _analyzer
    if _analyzer is None:
        try:
            from .ml.technical_analysis import FullSmartMoneyAnalysis
            _analyzer = FullSmartMoneyAnalysis()
        except Exception as e:
            print(f"Warning: Could not load analyzer: {e}")
    return _analyzer


# ============================================================================
# NEW ML-Powered Analyzers - Use ONLY trained knowledge from videos
# ============================================================================

_ml_signal_generator = None
_ml_analyzer = None


def get_ml_signal_generator():
    """Get the ML-powered signal generator that uses ONLY trained knowledge"""
    global _ml_signal_generator
    if _ml_signal_generator is None:
        try:
            from .services.signal_generator import SignalGenerator
            _ml_signal_generator = SignalGenerator()
            print("ML-powered SignalGenerator initialized")
        except Exception as e:
            print(f"Warning: Could not load ML signal generator: {e}")
    return _ml_signal_generator


def get_ml_analyzer():
    """Get the ML-powered analyzer that uses ONLY trained knowledge"""
    global _ml_analyzer
    if _ml_analyzer is None:
        try:
            from .services.smart_money_analyzer import SmartMoneyAnalyzer
            _ml_analyzer = SmartMoneyAnalyzer(use_ml=True)
            print("ML-powered SmartMoneyAnalyzer initialized")
        except Exception as e:
            print(f"Warning: Could not load ML analyzer: {e}")
    return _ml_analyzer


@app.get("/api/signals/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    timeframes: str = Query("H1,H4,D1", description="Comma-separated timeframes")
):
    """
    Analyze a symbol using ML-POWERED Smart Money methodology.

    IMPORTANT: This endpoint uses ONLY patterns the ML has learned from video training.
    Patterns not learned will not be detected.

    Returns:
    - ML patterns detected (from training)
    - ML patterns not yet learned (needs training)
    - Confidence scores based on ML training frequency
    """
    try:
        from .services.free_market_data import FreeMarketDataService
        from .models.signal import Timeframe
        from .ml.ml_pattern_engine import get_ml_engine

        market_service = FreeMarketDataService()
        ml_engine = get_ml_engine()
        ml_knowledge = ml_engine.get_knowledge_summary()

        tf_list = [tf.strip() for tf in timeframes.split(",")]

        # Fetch market data
        market_data = {}
        for tf in tf_list:
            df = market_service.get_ohlcv(symbol, tf, limit=200)
            if df is not None and not df.empty:
                market_data[tf] = df

        if not market_data:
            raise HTTPException(status_code=404, detail=f"No market data available for {symbol}")

        # Check ML knowledge status
        if ml_knowledge.get('status') != 'trained' or not ml_knowledge.get('patterns_learned'):
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'timeframes': list(market_data.keys()),
                'ml_patterns_detected': [],
                'ml_patterns_not_learned': ml_knowledge.get('patterns_not_learned', []),
                'analyses': {},
                'signal': None,
                'patterns': [],  # Empty patterns array for frontend compatibility
                'ml_status': 'not_trained',
                'ml_knowledge_status': '⚠️ ML has NO trained knowledge. Train videos using Vision Training to enable pattern detection.',
                'message': 'Train videos using Vision Training to enable ML-powered analysis.'
            }

        # Use ML-powered analyzer
        ml_analyzer = get_ml_analyzer()
        ml_sig_gen = get_ml_signal_generator()

        # Run ML-powered Smart Money analysis for each timeframe
        analyses = {}
        all_ml_patterns_used = []
        all_ml_patterns_not_learned = []
        all_ml_confidence = {}
        all_patterns = []  # Collect actual pattern objects for chart visualization

        for tf, df in market_data.items():
            if ml_analyzer:
                analysis = ml_analyzer.analyze(df)

                # Collect ML knowledge info
                all_ml_patterns_used.extend(analysis.ml_patterns_used)
                all_ml_patterns_not_learned.extend(analysis.ml_patterns_not_learned)
                all_ml_confidence.update(analysis.ml_confidence_scores)

                # Collect actual pattern objects for frontend visualization
                for ob in analysis.order_blocks:
                    all_patterns.append({
                        'pattern_type': f'{ob.type}_order_block',  # 'bullish_order_block' or 'bearish_order_block'
                        'high': ob.high,
                        'low': ob.low,
                        'start_index': ob.start_index,
                        'timeframe': tf,
                        'strength': ob.strength,
                        'mitigated': ob.mitigated,
                    })

                for fvg in analysis.fair_value_gaps:
                    all_patterns.append({
                        'pattern_type': f'{fvg.type}_fvg',  # 'bullish_fvg' or 'bearish_fvg'
                        'high': fvg.high,
                        'low': fvg.low,
                        'start_index': fvg.index,
                        'timeframe': tf,
                        'filled': fvg.filled,
                        'fill_percentage': fvg.fill_percentage,
                    })

                # Also add market structure events (BOS/CHoCH)
                for event in analysis.structure_events:
                    if event.type in ['bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish']:
                        all_patterns.append({
                            'pattern_type': event.type,
                            'price': event.level,
                            'timeframe': tf,
                            'description': event.description,
                        })

                # Add liquidity levels (equal highs/lows)
                if analysis.liquidity_levels.get('equal_highs'):
                    for eq in analysis.liquidity_levels['equal_highs']:
                        all_patterns.append({
                            'pattern_type': 'equal_highs',
                            'price': eq['level'],
                            'timeframe': tf,
                        })
                if analysis.liquidity_levels.get('equal_lows'):
                    for eq in analysis.liquidity_levels['equal_lows']:
                        all_patterns.append({
                            'pattern_type': 'equal_lows',
                            'price': eq['level'],
                            'timeframe': tf,
                        })

                analyses[tf] = {
                    'bias': analysis.bias.value,
                    'bias_confidence': analysis.bias_confidence,
                    'zone': analysis.premium_discount.get('zone', 'neutral'),
                    'order_blocks': len(analysis.order_blocks),
                    'fvgs': len(analysis.fair_value_gaps),
                    'market_structure': analysis.market_structure.value,
                    'reasoning': analysis.bias_reasoning,
                    'ml_patterns_used': analysis.ml_patterns_used,
                    'ml_patterns_not_learned': analysis.ml_patterns_not_learned,
                    'ml_confidence': analysis.ml_confidence_scores,
                }

        # Deduplicate
        all_ml_patterns_used = list(set(all_ml_patterns_used))
        all_ml_patterns_not_learned = list(set(all_ml_patterns_not_learned))

        # Get ML Engine for reasoning generation
        ml_engine = get_ml_engine()

        # Generate ML-based reasoning (ONLY from learned knowledge, not generic SMC)
        ml_reasoning = ml_engine.generate_ml_reasoning(
            detected_patterns=all_ml_patterns_used,
            bias=analyses.get(tf_list[0], {}).get('bias', 'neutral'),
            zone=analyses.get(tf_list[0], {}).get('zone', 'neutral'),
        )

        # Get entry/exit reasoning from ML knowledge
        entry_exit_reasoning = ml_engine.get_entry_exit_reasoning(
            direction=analyses.get(tf_list[0], {}).get('bias', 'neutral'),
            patterns_found={'ml_patterns': all_ml_patterns_used}
        )

        # Generate ML-powered signal using primary timeframe (first in list)
        signal = None
        primary_tf = tf_list[0]
        if ml_sig_gen and primary_tf in market_data:
            try:
                tf_enum = Timeframe(primary_tf)
                signal_obj = ml_sig_gen.generate_signal(
                    symbol=symbol,
                    data=market_data[primary_tf],
                    timeframe=tf_enum
                )
                if signal_obj:
                    signal = signal_obj.to_dict()
            except Exception as e:
                print(f"Signal generation error: {e}")

        # Get full list of not-learned patterns from ML knowledge base (not just from analysis)
        ml_not_learned_full = ml_knowledge.get('patterns_not_learned', [])

        # Generate ML knowledge status message
        if all_ml_patterns_used and not ml_not_learned_full:
            ml_status_msg = f"✅ ML-powered analysis. Patterns detected: {', '.join(all_ml_patterns_used)}"
        elif all_ml_patterns_used and ml_not_learned_full:
            ml_status_msg = f"⚡ Partial ML coverage. Detected: {', '.join(all_ml_patterns_used)}. Needs training: {', '.join(ml_not_learned_full)}"
        else:
            ml_status_msg = "⚠️ No ML patterns detected in this data. Train more videos to improve detection."

        # Get all learned patterns with their confidences
        all_learned_patterns = {}
        for p in ml_knowledge.get('patterns_learned', []):
            all_learned_patterns[p['type']] = p['confidence']

        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'timeframes': list(market_data.keys()),
            'ml_patterns_detected': all_ml_patterns_used,  # Patterns actively detected in this analysis
            'ml_patterns_not_learned': ml_not_learned_full,  # Full list from ML knowledge base
            'ml_confidence_scores': all_learned_patterns,  # All learned patterns with confidences
            'analyses': analyses,
            'signal': signal,
            'patterns': all_patterns,  # Full pattern objects for chart visualization (OBs, FVGs, BOS, CHoCH, etc.)
            'ml_status': 'trained',
            'ml_knowledge_status': ml_status_msg,
            # NEW: ML-based reasoning (from learned knowledge ONLY, not generic SMC)
            'ml_reasoning': ml_reasoning,  # Why ML made this analysis decision
            'entry_exit_reasoning': entry_exit_reasoning,  # Why entry/SL/TP are placed here
            'training_info': {
                'videos_trained': ml_knowledge.get('total_videos', 0),
                'frames_analyzed': ml_knowledge.get('total_frames', 0),
                'patterns_learned': [p['type'] for p in ml_knowledge.get('patterns_learned', [])],
            }
        }

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals/quick/{symbol}")
async def quick_signal(symbol: str):
    """
    Get a quick ML-powered signal for a symbol (single timeframe analysis).

    Uses ONLY patterns the ML has learned from video training.
    Returns ML knowledge status showing what patterns were/weren't detected.
    """
    try:
        from .services.free_market_data import FreeMarketDataService
        from .ml.ml_pattern_engine import get_ml_engine

        market_service = FreeMarketDataService()
        ml_engine = get_ml_engine()
        ml_knowledge = ml_engine.get_knowledge_summary()

        # Get H1 data
        df = market_service.get_ohlcv(symbol, 'H1', limit=100)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        current_price = float(df['close'].iloc[-1])

        # Check ML knowledge status
        is_ml_trained = ml_knowledge.get('status') == 'trained' and ml_knowledge.get('patterns_learned')

        if not is_ml_trained:
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'bias': 'neutral',
                'zone': 'neutral',
                'ml_patterns_detected': [],
                'ml_patterns_not_learned': ml_knowledge.get('patterns_not_learned', []),
                'patterns': [],  # Empty for consistency
                'summary': '⚠️ ML not trained - use Vision Training on videos to enable pattern detection',
                'current_price': current_price,
                'ml_status': 'not_trained',
                'ml_knowledge_status': 'No trained knowledge. Train videos first.',
            }

        # Use ML-powered analyzer
        ml_analyzer = get_ml_analyzer()

        if ml_analyzer:
            analysis = ml_analyzer.analyze(df)

            # Generate summary based on ML patterns
            if analysis.ml_patterns_used:
                patterns_str = ', '.join(analysis.ml_patterns_used)
                if analysis.bias.value == 'bullish':
                    summary = f"✅ BULLISH - ML detected: {patterns_str}"
                elif analysis.bias.value == 'bearish':
                    summary = f"✅ BEARISH - ML detected: {patterns_str}"
                else:
                    summary = f"⚡ NEUTRAL - ML detected: {patterns_str}"
            else:
                summary = f"⚠️ No ML patterns found in current data"

            # Build patterns array for frontend visualization
            patterns = []
            for ob in analysis.order_blocks:
                patterns.append({
                    'pattern_type': f'{ob.type}_order_block',
                    'high': ob.high,
                    'low': ob.low,
                    'start_index': ob.start_index,
                    'timeframe': 'H1',
                    'strength': ob.strength,
                })
            for fvg in analysis.fair_value_gaps:
                patterns.append({
                    'pattern_type': f'{fvg.type}_fvg',
                    'high': fvg.high,
                    'low': fvg.low,
                    'start_index': fvg.index,
                    'timeframe': 'H1',
                })

            # Determine kill zone status
            now = datetime.utcnow()
            hour = now.hour
            kill_zone_active = any([
                0 <= hour < 4,   # Asian
                7 <= hour < 10,  # London Open
                12 <= hour < 15, # NY Open
                15 <= hour < 17, # London Close
            ])

            if 0 <= hour < 4:
                kill_zone_name = "Asian Session"
            elif 7 <= hour < 10:
                kill_zone_name = "London Open (Premium)"
            elif 12 <= hour < 15:
                kill_zone_name = "New York Open (Premium)"
            elif 15 <= hour < 17:
                kill_zone_name = "London Close"
            else:
                kill_zone_name = None

            # Generate ML-based reasoning (from learned knowledge ONLY)
            ml_reasoning = ml_engine.generate_ml_reasoning(
                detected_patterns=analysis.ml_patterns_used,
                bias=analysis.bias.value,
                zone=analysis.premium_discount.get('zone', 'neutral'),
            )

            # Get entry/exit reasoning from ML knowledge
            entry_exit_reasoning = ml_engine.get_entry_exit_reasoning(
                direction=analysis.bias.value,
                patterns_found={'ml_patterns': analysis.ml_patterns_used}
            )

            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'bias': analysis.bias.value,
                'bias_confidence': analysis.bias_confidence,
                'zone': analysis.premium_discount.get('zone', 'neutral'),
                'ml_patterns_detected': analysis.ml_patterns_used,
                'ml_patterns_not_learned': analysis.ml_patterns_not_learned,
                'ml_confidence_scores': analysis.ml_confidence_scores,
                'order_blocks_found': len(analysis.order_blocks),
                'fvgs_found': len(analysis.fair_value_gaps),
                'patterns': patterns,  # Full pattern objects for chart visualization
                'summary': summary,
                'reasoning': analysis.bias_reasoning,
                'ml_reasoning': ml_reasoning,  # Detailed ML reasoning from learned knowledge
                'entry_exit_reasoning': entry_exit_reasoning,  # Why entry/SL/TP placed here
                'current_price': current_price,
                'ml_status': 'trained',
                'ml_knowledge_status': f"Trained on {ml_knowledge.get('total_videos', 0)} videos",
                'kill_zone_active': kill_zone_active,
                'kill_zone_name': kill_zone_name,
            }
        else:
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'bias': 'neutral',
                'zone': 'neutral',
                'ml_patterns_detected': [],
                'ml_patterns_not_learned': [],
                'summary': 'ML Analyzer not available',
                'current_price': current_price,
                'ml_status': 'error',
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/status")
async def get_ml_status():
    """Get ML model status and learning progress"""
    kb = get_knowledge_base()

    if kb:
        progress = kb.get_learning_progress()
        # Check if models are actually trained (not just loaded empty)
        is_trained = progress.get('n_transcripts_processed', 0) > 0 or progress.get('n_training_runs', 0) > 0
        return {
            'status': 'operational' if is_trained else 'not_trained',
            'knowledge_base_loaded': is_trained,
            **progress
        }
    else:
        return {
            'status': 'not_trained',
            'knowledge_base_loaded': False,
            'message': 'Run training pipeline to initialize ML models'
        }


@app.get("/api/ml/concepts/{concept_name}")
async def query_concept(concept_name: str):
    """Query detailed information about an Smart Money concept from the knowledge base"""
    kb = get_knowledge_base()

    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    result = kb.query_concept(concept_name)

    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])

    return result


@app.post("/api/ml/predict")
async def predict_concepts(text: str = Query(..., description="Text to analyze")):
    """Predict Smart Money concepts in given text"""
    kb = get_knowledge_base()

    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    try:
        prediction = kb.predict_concepts(text)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train")
async def trigger_training(incremental: bool = Query(True, description="Incremental training")):
    """Trigger ML model training (async)"""
    try:
        from .ml.training_pipeline import SmartMoneyKnowledgeBase

        kb = SmartMoneyKnowledgeBase(str(DATA_DIR))

        if incremental:
            try:
                kb.load()
            except:
                pass

        # Run training
        results = kb.train(incremental=incremental)
        kb.save()

        # Reload global instance
        global _knowledge_base
        _knowledge_base = kb

        return {
            'status': 'success',
            'message': 'Training complete',
            'results': {
                'n_transcripts': results.get('n_transcripts', 0),
                'classifier_f1': results.get('components', {}).get('classifier', {}).get('ensemble_f1'),
                'concepts_defined': results.get('components', {}).get('definitions', {}).get('n_concepts', 0),
                'rules_extracted': results.get('components', {}).get('rules', {}).get('n_rules', 0),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/reset")
async def reset_ml_state():
    """Reset all ML state - clears cached models and knowledge base"""
    global _knowledge_base, _signal_generator, _analyzer

    # Clear all cached instances
    _knowledge_base = None
    _signal_generator = None
    _analyzer = None

    return {
        'status': 'success',
        'message': 'ML state reset. All cached models cleared.'
    }


@app.post("/api/ml/whitewash")
async def whitewash_ml():
    """
    Complete ML whitewash - deletes all trained models and resets state.
    Transcripts are preserved but all training is wiped clean.
    """
    global _knowledge_base, _signal_generator, _analyzer
    import shutil

    # Clear all cached instances
    _knowledge_base = None
    _signal_generator = None
    _analyzer = None

    # Delete ML model files
    ml_models_dir = DATA_DIR / "ml_models"
    deleted_files = []

    if ml_models_dir.exists():
        for model_file in ml_models_dir.glob("*"):
            if model_file.is_file():
                deleted_files.append(model_file.name)
                model_file.unlink()

    return {
        'status': 'success',
        'message': 'ML whitewash complete. All models deleted, transcripts preserved.',
        'deleted_files': deleted_files
    }


# Training status for selective training
_selective_training_status = {}


@app.post("/api/ml/train/playlist/{playlist_id}")
async def train_ml_from_playlist(playlist_id: str):
    """Start ML training using only transcripts from a specific playlist"""
    import uuid

    # Verify playlist exists and has transcripts
    playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"
    if not playlist_file.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    with open(playlist_file) as f:
        playlist_data = json.load(f)

    # Get video IDs from playlist
    video_ids = [v.get("video_id") for v in playlist_data.get("videos", []) if v.get("video_id")]

    # Check which have transcripts
    available_transcripts = []
    for vid in video_ids:
        transcript_file = TRANSCRIPTS_DIR / f"{vid}.json"
        if transcript_file.exists():
            available_transcripts.append(vid)

    if not available_transcripts:
        raise HTTPException(status_code=400, detail="No transcripts available for this playlist")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    _selective_training_status[job_id] = {
        "status": "starting",
        "message": "Initializing training...",
        "playlist_id": playlist_id,
        "playlist_title": playlist_data.get("title"),
        "total_transcripts": len(available_transcripts),
        "progress": 0,
        "started_at": datetime.utcnow().isoformat()
    }

    # Start background training in a separate thread (not blocking event loop)
    import threading
    thread = threading.Thread(
        target=train_from_playlist_worker,
        args=(job_id, playlist_id, available_transcripts, playlist_data.get("title")),
        daemon=True
    )
    thread.start()

    return {
        "status": "started",
        "job_id": job_id,
        "playlist_title": playlist_data.get("title"),
        "transcript_count": len(available_transcripts)
    }


def train_from_playlist_worker(job_id: str, playlist_id: str, video_ids: list, playlist_title: str):
    """Background worker for selective playlist training with per-transcript progress"""
    global _selective_training_status, _knowledge_base

    try:
        total = len(video_ids)
        _selective_training_status[job_id]["status"] = "loading"
        _selective_training_status[job_id]["message"] = f"Loading transcripts from '{playlist_title}'..."
        _selective_training_status[job_id]["current_transcript"] = 0
        _selective_training_status[job_id]["transcripts_loaded"] = []

        from .ml.training_pipeline import SmartMoneyKnowledgeBase

        # Load transcripts one by one with progress updates
        transcripts = []
        for i, vid in enumerate(video_ids):
            transcript_file = TRANSCRIPTS_DIR / f"{vid}.json"
            if transcript_file.exists():
                try:
                    with open(transcript_file) as f:
                        transcript = json.load(f)
                        if transcript.get('full_text'):
                            transcripts.append(transcript)
                            title = transcript.get('title', vid)[:50]
                            _selective_training_status[job_id]["current_transcript"] = i + 1
                            _selective_training_status[job_id]["current_title"] = title
                            _selective_training_status[job_id]["message"] = f"Loaded: {title}..."
                            _selective_training_status[job_id]["transcripts_loaded"].append({
                                "video_id": vid,
                                "title": title,
                                "status": "loaded"
                            })
                except Exception as e:
                    _selective_training_status[job_id]["transcripts_loaded"].append({
                        "video_id": vid,
                        "title": vid,
                        "status": "error",
                        "error": str(e)
                    })

        if not transcripts:
            _selective_training_status[job_id]["status"] = "error"
            _selective_training_status[job_id]["message"] = "No valid transcripts found"
            return

        # Now train with all loaded transcripts
        _selective_training_status[job_id]["status"] = "training"
        _selective_training_status[job_id]["message"] = f"Training ML on {len(transcripts)} transcripts..."
        _selective_training_status[job_id]["training_phase"] = "initializing"

        # Load existing knowledge base to get previously trained video IDs
        kb = SmartMoneyKnowledgeBase(str(DATA_DIR))
        kb.load()  # Load saved data including trained_video_ids

        # Get previously trained transcripts and merge with new ones
        previously_trained_ids = set(kb.trained_video_ids) if kb.trained_video_ids else set()
        new_video_ids = set(t.get('video_id') for t in transcripts)

        # If there are previously trained transcripts not in current batch, load them too
        all_transcripts = list(transcripts)  # Start with new transcripts
        for vid in previously_trained_ids:
            if vid not in new_video_ids:
                transcript_file = TRANSCRIPTS_DIR / f"{vid}.json"
                if transcript_file.exists():
                    try:
                        with open(transcript_file) as f:
                            old_transcript = json.load(f)
                            if old_transcript.get('full_text'):
                                all_transcripts.append(old_transcript)
                    except:
                        pass

        _selective_training_status[job_id]["message"] = f"Training ML on {len(all_transcripts)} total transcripts ({len(transcripts)} new + {len(all_transcripts) - len(transcripts)} existing)..."

        # Update progress during training phases
        _selective_training_status[job_id]["training_phase"] = "extracting_concepts"
        _selective_training_status[job_id]["message"] = "Extracting Smart Money concepts..."

        # Train with all transcripts (new + previously trained)
        results = kb.train(transcripts=all_transcripts, incremental=False)

        # Store all trained video IDs
        kb.trained_video_ids = list(set(t.get('video_id') for t in all_transcripts))

        _selective_training_status[job_id]["training_phase"] = "saving"
        _selective_training_status[job_id]["message"] = "Saving trained models..."
        kb.save()

        # Reload global instance
        _knowledge_base = kb

        _selective_training_status[job_id]["status"] = "completed"
        _selective_training_status[job_id]["message"] = f"Training complete! Processed {len(transcripts)} transcripts."
        _selective_training_status[job_id]["training_phase"] = "done"
        _selective_training_status[job_id]["results"] = {
            "n_transcripts": results.get("n_transcripts", len(transcripts)),
            "classifier_f1": results.get("components", {}).get("classifier", {}).get("ensemble_f1"),
            "concepts_defined": results.get("components", {}).get("knowledge_base", {}).get("n_concepts"),
        }
        _selective_training_status[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}"
        print(f"[Training Error] {error_msg}")
        print(traceback.format_exc())
        _selective_training_status[job_id]["status"] = "error"
        _selective_training_status[job_id]["message"] = error_msg
        _selective_training_status[job_id]["traceback"] = traceback.format_exc()


@app.get("/api/ml/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Get status of a selective training job"""
    if job_id not in _selective_training_status:
        raise HTTPException(status_code=404, detail="Training job not found")
    return _selective_training_status[job_id]


@app.get("/api/ml/train/stream/{job_id}")
async def stream_training_status(job_id: str):
    """SSE stream for training progress"""
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        while True:
            if job_id in _selective_training_status:
                status = _selective_training_status[job_id]
                yield {"data": json.dumps(status)}

                if status.get("status") in ["completed", "error"]:
                    break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


# ============================================================================
# ML Vision Training - Multimodal video analysis
# ============================================================================

_vision_training_status = {}


@app.post("/api/ml/train/vision/{playlist_id}")
async def train_ml_with_vision(
    playlist_id: str,
    background_tasks: BackgroundTasks,
    vision_provider: str = Query("local", description="Vision AI provider: 'local' (FREE on M1/M2/M3 Mac), 'anthropic' (paid), 'openai' (paid)"),
    max_frames: int = Query(0, description="Max frames per video (0 = no limit, recommended for comprehensive)"),
    extraction_mode: str = Query("comprehensive", description="Learning mode: 'comprehensive' (every 3s - learns everything), 'thorough' (every 5s), 'balanced' (every 10-15s), 'selective' (key moments only)")
):
    """
    Start multimodal ML training that analyzes video frames along with transcripts.
    This allows the ML to understand visual patterns shown in trading tutorials.

    Vision Providers:
    - local: FREE! Uses mlx-vlm on Apple Silicon (M1/M2/M3 Mac) - no API costs
    - anthropic: Claude API (costs money)
    - openai: GPT-4V API (costs money)

    Extraction Modes:
    - comprehensive: Like a dedicated student - extracts frame every 3 seconds to miss NOTHING
    - thorough: Good coverage - extracts every 5 seconds with extra frames at key moments
    - balanced: Moderate coverage - extracts every 10-15 seconds with keyword boosting
    - selective: Fastest but may miss content - only extracts at demonstrative moments
    """
    import uuid
    import traceback

    # Create job ID
    job_id = str(uuid.uuid4())[:8]

    # Initialize status
    _vision_training_status[job_id] = {
        "job_id": job_id,
        "playlist_id": playlist_id,
        "status": "starting",
        "progress": 0,
        "total": 0,
        "current_video": None,
        "message": f"Initializing vision training ({extraction_mode} mode)...",
        "started_at": datetime.utcnow().isoformat(),
        "vision_provider": vision_provider,
        "extraction_mode": extraction_mode,
        "frames_analyzed": 0,
        "charts_detected": 0,
        "patterns_found": 0
    }

    def run_vision_training():
        """Background task for vision training"""
        try:
            from .ml.training_pipeline import SmartMoneyKnowledgeBase

            _vision_training_status[job_id]["status"] = "loading"
            _vision_training_status[job_id]["message"] = "Loading playlist..."

            # Load playlist
            playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"
            if not playlist_file.exists():
                _vision_training_status[job_id]["status"] = "error"
                _vision_training_status[job_id]["error"] = f"Playlist not found: {playlist_id}"
                return

            with open(playlist_file) as f:
                playlist_data = json.load(f)

            videos = playlist_data.get("videos", [])
            video_ids = [v.get("video_id") for v in videos if v.get("video_id")]

            # Load transcripts for these videos
            transcripts = []
            for vid in video_ids:
                transcript_file = TRANSCRIPTS_DIR / f"{vid}.json"
                if transcript_file.exists():
                    try:
                        with open(transcript_file) as f:
                            transcript = json.load(f)
                            if transcript.get('full_text'):
                                transcripts.append(transcript)
                    except Exception:
                        pass

            if not transcripts:
                _vision_training_status[job_id]["status"] = "error"
                _vision_training_status[job_id]["error"] = "No transcripts found for playlist videos"
                return

            _vision_training_status[job_id]["total"] = len(transcripts)
            _vision_training_status[job_id]["message"] = f"Found {len(transcripts)} transcripts with videos"

            # Create knowledge base and run vision training
            kb = SmartMoneyKnowledgeBase()

            def progress_callback(current, total, message):
                _vision_training_status[job_id]["progress"] = current
                _vision_training_status[job_id]["total"] = total
                _vision_training_status[job_id]["message"] = message
                _vision_training_status[job_id]["status"] = "training"

                # Extract video title from message
                if "Analyzing" in message:
                    _vision_training_status[job_id]["current_video"] = message.replace("Analyzing visual content: ", "")

            # Run multimodal training
            results = kb.train_with_vision(
                transcripts=transcripts,
                vision_provider=vision_provider,
                max_frames_per_video=max_frames,
                extraction_mode=extraction_mode,
                progress_callback=progress_callback
            )

            # Save the trained model
            kb.save()

            # Reset global knowledge base to use new model
            global _knowledge_base
            _knowledge_base = None

            # Update final status
            vision_analysis = results.get('vision_analysis', {})
            _vision_training_status[job_id].update({
                "status": "completed",
                "message": "Vision training completed successfully",
                "completed_at": datetime.utcnow().isoformat(),
                "frames_analyzed": vision_analysis.get('total_frames_analyzed', 0),
                "charts_detected": vision_analysis.get('chart_frames', 0),
                "patterns_found": len(vision_analysis.get('visual_patterns', {})),
                "visual_concepts": vision_analysis.get('visual_concepts', 0),
                "results": {
                    "videos_processed": results.get('n_transcripts', 0),
                    "concepts_learned": results['components'].get('definitions', {}).get('n_concepts', 0),
                    "rules_extracted": results['components'].get('rules', {}).get('n_rules', 0),
                    "vision_analysis": vision_analysis
                }
            })

        except Exception as e:
            _vision_training_status[job_id]["status"] = "error"
            _vision_training_status[job_id]["error"] = str(e)
            _vision_training_status[job_id]["traceback"] = traceback.format_exc()

    # Run in background
    background_tasks.add_task(run_vision_training)

    return {
        "job_id": job_id,
        "status": "started",
        "message": f"Vision training started for playlist {playlist_id}",
        "vision_provider": vision_provider,
        "max_frames_per_video": max_frames,
        "status_url": f"/api/ml/train/vision/status/{job_id}",
        "stream_url": f"/api/ml/train/vision/stream/{job_id}"
    }


@app.get("/api/ml/train/vision/status/{job_id}")
async def get_vision_training_status(job_id: str):
    """Get status of a vision training job"""
    if job_id not in _vision_training_status:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _vision_training_status[job_id]


@app.get("/api/ml/train/vision/stream/{job_id}")
async def stream_vision_training_status(job_id: str):
    """SSE stream for vision training progress"""
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        while True:
            if job_id in _vision_training_status:
                status = _vision_training_status[job_id]
                yield {"data": json.dumps(status)}

                if status.get("status") in ["completed", "error"]:
                    break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.post("/api/ml/train/vision/video/{video_id}")
async def train_single_video_with_vision(
    video_id: str,
    background_tasks: BackgroundTasks,
    vision_provider: str = Query("local", description="Vision AI provider: 'local' (FREE), 'anthropic' (paid), 'openai' (paid)"),
    max_frames: int = Query(0, description="Max frames (0 = no limit)"),
    extraction_mode: str = Query("comprehensive", description="Learning mode")
):
    """
    Start vision training for a SINGLE video.
    Perfect for testing or when you want to train on specific videos.
    """
    import uuid
    import traceback

    # Create job ID
    job_id = str(uuid.uuid4())[:8]

    # Check if transcript exists for this video
    transcript_file = TRANSCRIPTS_DIR / f"{video_id}.json"
    if not transcript_file.exists():
        raise HTTPException(status_code=404, detail=f"Transcript not found for video: {video_id}")

    # Load transcript to get title
    with open(transcript_file) as f:
        transcript_data = json.load(f)
    video_title = transcript_data.get('title', video_id)

    # Initialize status
    _vision_training_status[job_id] = {
        "job_id": job_id,
        "video_id": video_id,
        "video_title": video_title,
        "status": "starting",
        "progress": 0,
        "total": 1,
        "current_video": video_title,
        "message": f"Initializing vision training for: {video_title}",
        "started_at": datetime.utcnow().isoformat(),
        "vision_provider": vision_provider,
        "extraction_mode": extraction_mode,
        "frames_analyzed": 0,
        "charts_detected": 0,
        "patterns_found": 0
    }

    def run_single_video_vision_training():
        """Background task for single video vision training"""
        try:
            from .ml.training_pipeline import SmartMoneyKnowledgeBase

            _vision_training_status[job_id]["status"] = "loading"
            _vision_training_status[job_id]["message"] = f"Loading transcript for {video_title}..."

            # Load the transcript
            with open(transcript_file) as f:
                transcript = json.load(f)

            if not transcript.get('full_text'):
                _vision_training_status[job_id]["status"] = "error"
                _vision_training_status[job_id]["error"] = "Transcript has no text content"
                return

            _vision_training_status[job_id]["message"] = f"Starting vision analysis for: {video_title}"

            # Create knowledge base and run vision training
            kb = SmartMoneyKnowledgeBase()

            def progress_callback(current, total, message):
                _vision_training_status[job_id]["progress"] = current
                _vision_training_status[job_id]["total"] = total
                _vision_training_status[job_id]["message"] = message
                _vision_training_status[job_id]["status"] = "training"

            # Run multimodal training on single video
            results = kb.train_with_vision(
                transcripts=[transcript],
                vision_provider=vision_provider,
                max_frames_per_video=max_frames,
                extraction_mode=extraction_mode,
                progress_callback=progress_callback
            )

            # Save the trained model
            kb.save()

            # Reset global knowledge base to use new model
            global _knowledge_base
            _knowledge_base = None

            # Update final status
            vision_analysis = results.get('vision_analysis', {})
            _vision_training_status[job_id].update({
                "status": "completed",
                "message": f"Vision training completed for: {video_title}",
                "completed_at": datetime.utcnow().isoformat(),
                "frames_analyzed": vision_analysis.get('total_frames_analyzed', 0),
                "charts_detected": vision_analysis.get('chart_frames', 0),
                "patterns_found": len(vision_analysis.get('visual_patterns', {})),
                "visual_concepts": vision_analysis.get('visual_concepts', 0),
                "results": {
                    "video_title": video_title,
                    "concepts_learned": results['components'].get('definitions', {}).get('n_concepts', 0),
                    "rules_extracted": results['components'].get('rules', {}).get('n_rules', 0),
                    "vision_analysis": vision_analysis
                }
            })

        except Exception as e:
            _vision_training_status[job_id]["status"] = "error"
            _vision_training_status[job_id]["error"] = str(e)
            _vision_training_status[job_id]["traceback"] = traceback.format_exc()

    # Run in background
    background_tasks.add_task(run_single_video_vision_training)

    return {
        "job_id": job_id,
        "status": "started",
        "message": f"Vision training started for video: {video_title}",
        "video_id": video_id,
        "video_title": video_title,
        "vision_provider": vision_provider,
        "extraction_mode": extraction_mode,
        "status_url": f"/api/ml/train/vision/status/{job_id}"
    }


@app.get("/api/ml/vision/status")
async def get_vision_capabilities():
    """Check if vision training is available and its current status"""
    try:
        from .ml.training_pipeline import VISION_AVAILABLE
        from .ml.video_vision_analyzer import VideoVisionTrainer
        vision_available = VISION_AVAILABLE
    except ImportError:
        vision_available = False

    # Check if Ollama is available for local vision (FREE!)
    ollama_available = False
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_available = True
    except:
        pass

    # Vision is available if we have API keys OR local Ollama
    vision_available = vision_available or ollama_available

    kb = get_knowledge_base()
    vision_knowledge = getattr(kb, 'vision_knowledge', {}) if kb else {}

    supported_providers = []
    if ollama_available:
        supported_providers.append("local")
    if os.environ.get("ANTHROPIC_API_KEY"):
        supported_providers.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        supported_providers.append("openai")

    return {
        "vision_available": vision_available,
        "has_vision_knowledge": bool(vision_knowledge),
        "patterns_learned": len(vision_knowledge.get('pattern_frequency', {})) if vision_knowledge else 0,
        "visual_concepts": len(vision_knowledge.get('visual_concepts', [])) if vision_knowledge else 0,
        "videos_with_vision": len(vision_knowledge.get('analyzed_videos', [])) if vision_knowledge else 0,
        "supported_providers": supported_providers,
        "ollama_available": ollama_available,
        "requirements": {
            "local": "Ollama running locally (FREE!) - ollama.ai",
            "anthropic": "ANTHROPIC_API_KEY environment variable",
            "openai": "OPENAI_API_KEY environment variable"
        }
    }


@app.get("/api/ml/vision/patterns/{pattern_type}")
async def get_visual_pattern_examples(pattern_type: str):
    """
    Get visual examples of a specific Smart Money pattern type.
    Examples: FVG, OrderBlock, Breaker, Mitigation, etc.
    """
    kb = get_knowledge_base()
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    examples = kb.get_visual_pattern_examples(pattern_type)

    if not examples:
        # Check if we have any vision knowledge at all
        vision_knowledge = getattr(kb, 'vision_knowledge', {})
        available_patterns = list(vision_knowledge.get('pattern_frequency', {}).keys()) if vision_knowledge else []

        return {
            "pattern_type": pattern_type,
            "examples": [],
            "count": 0,
            "message": f"No visual examples found for '{pattern_type}'",
            "available_patterns": available_patterns
        }

    return {
        "pattern_type": pattern_type,
        "examples": examples[:20],  # Limit to 20 examples
        "count": len(examples)
    }


@app.get("/api/ml/vision/teaching-moments")
async def get_teaching_moments(
    concept: Optional[str] = Query(None, description="Filter by concept (e.g., 'FVG', 'Order Block')")
):
    """
    Get key teaching moments from analyzed videos.
    These are moments where the tutor is demonstrating or explaining a concept visually.
    """
    kb = get_knowledge_base()
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    moments = kb.get_teaching_moments(concept)

    return {
        "concept_filter": concept,
        "teaching_moments": moments[:50],  # Limit to 50 moments
        "total_count": len(moments)
    }


@app.get("/api/ml/vision/knowledge")
async def get_visual_knowledge():
    """Get comprehensive visual knowledge learned from video analysis"""
    from .ml.ml_pattern_engine import get_ml_engine

    # Get ML engine knowledge
    try:
        ml_engine = get_ml_engine()
        ml_summary = ml_engine.get_knowledge_summary()
    except Exception as e:
        ml_summary = {'status': 'error', 'patterns_learned': [], 'patterns_not_learned': []}

    # Check if we have any trained knowledge
    if ml_summary.get('status') != 'trained' or not ml_summary.get('patterns_learned'):
        return {
            "has_vision_knowledge": False,
            "message": "No vision training has been performed. Use /api/ml/train/vision/{playlist_id} to start vision training.",
            "patterns_not_learned": ml_summary.get('patterns_not_learned', [])
        }

    # Build detailed pattern info
    pattern_details = []
    for p in ml_summary.get('patterns_learned', []):
        pattern_details.append({
            'type': p.get('type'),
            'frequency': p.get('frequency', 0),
            'confidence': p.get('confidence', 0),
            'has_teaching': p.get('has_teaching', False),
            'has_visual': p.get('has_visual', False)
        })

    return {
        "has_vision_knowledge": True,
        "patterns_learned": len(pattern_details),
        "pattern_details": pattern_details,
        "patterns_not_learned": ml_summary.get('patterns_not_learned', []),
        "videos_with_vision": ml_summary.get('total_videos', 0),
        "total_frames_analyzed": ml_summary.get('total_frames', 0),
        "chart_frames": ml_summary.get('chart_frames', 0),
        "visual_concepts": len(pattern_details),  # Same as patterns learned
        "key_teaching_moments_count": sum(1 for p in pattern_details if p.get('has_teaching', False)),
        "training_sources": ml_summary.get('training_sources', []),
        "last_trained": ml_summary.get('last_trained')
    }


@app.get("/api/ml/pattern-knowledge")
async def get_ml_pattern_knowledge():
    """
    Get ML's pattern knowledge status - what it can and cannot detect for live charts.

    This endpoint shows:
    - Which patterns the ML has learned and can detect
    - Which patterns the ML hasn't learned yet (needs training)
    - Confidence levels for each learned pattern
    - Training sources (which videos contributed to learning)

    IMPORTANT: Live charts ONLY use patterns the ML has learned.
    If a pattern isn't in the 'learned' list, it won't be detected.
    """
    try:
        from .ml.ml_pattern_engine import get_ml_engine

        engine = get_ml_engine()
        summary = engine.get_knowledge_summary()

        return {
            "status": summary.get('status', 'unknown'),
            "message": summary.get('message', ''),
            "patterns_learned": summary.get('patterns_learned', []),
            "patterns_not_learned": summary.get('patterns_not_learned', []),
            "training_stats": {
                "total_videos": summary.get('total_videos', 0),
                "total_frames_analyzed": summary.get('total_frames', 0),
                "chart_frames": summary.get('chart_frames', 0),
                "last_trained": summary.get('last_trained'),
            },
            "training_sources": summary.get('training_sources', []),
            "usage_info": {
                "note": "Live charts ONLY detect patterns from 'patterns_learned' list.",
                "to_learn_more": "Train more videos using Vision Training to expand ML knowledge.",
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "patterns_learned": [],
            "patterns_not_learned": ["fvg", "order_block", "breaker_block", "market_structure"],
        }


@app.get("/api/market/price/{symbol}")
async def get_price(symbol: str):
    """Get current price for a symbol"""
    try:
        from .services.free_market_data import FreeMarketDataService

        service = FreeMarketDataService()
        price = service.get_current_price(symbol)

        if price is None:
            raise HTTPException(status_code=404, detail=f"Price not available for {symbol}")

        return {
            'symbol': symbol,
            'price': price,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe"),
    limit: int = Query(100, description="Number of candles")
):
    """Get OHLCV data for a symbol"""
    try:
        from .services.free_market_data import FreeMarketDataService

        service = FreeMarketDataService()
        df = service.get_ohlcv(symbol, timeframe, limit)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Convert to JSON-friendly format
        data = df.reset_index().to_dict(orient='records')

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'count': len(data),
            'data': data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/symbols")
async def get_symbols():
    """Get list of available symbols"""
    try:
        from .services.free_market_data import FreeMarketDataService

        service = FreeMarketDataService()
        symbols = service.get_available_symbols()

        return {
            'symbols': symbols,
            'count': len(symbols)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analysis (placeholder for future)
# ============================================================================

@app.get("/api/analysis/status")
async def get_analysis_status():
    """Get system analysis status"""
    # Count transcripts
    transcript_count = len(list(TRANSCRIPTS_DIR.glob("*.json"))) if TRANSCRIPTS_DIR.exists() else 0

    # Count videos
    total_videos = 0
    if PLAYLISTS_DIR.exists():
        for pf in PLAYLISTS_DIR.glob("*.json"):
            with open(pf) as f:
                data = json.load(f)
                total_videos += data.get("video_count", 0)

    # Check ML status
    kb = get_knowledge_base()
    ml_status = kb.get_learning_progress() if kb else {'status': 'not_trained'}

    # Count signals
    sig_gen = get_signal_generator()
    signals_generated = sig_gen.signals_generated if sig_gen else 0

    # Use n_trained_videos if available (more accurate), fallback to n_transcripts_processed
    n_trained = ml_status.get('n_trained_videos', ml_status.get('n_transcripts_processed', 0))

    return {
        "status": "operational",
        "playlists_loaded": len(list(PLAYLISTS_DIR.glob("*.json"))) if PLAYLISTS_DIR.exists() else 0,
        "total_videos": total_videos,
        "transcripts_ready": transcript_count,
        "transcription_progress": f"{transcript_count}/{total_videos}" if total_videos else "0/0",
        "concepts_loaded": 40,  # From taxonomy
        "ml_trained": n_trained > 0,
        "n_transcripts_trained": n_trained,
        "signals_generated": signals_generated,
        "last_updated": datetime.utcnow().isoformat()
    }


# ============================================================================
# Telegram Notifications
# ============================================================================

@app.get("/api/notifications/status")
async def get_notification_status():
    """Check Telegram notification status"""
    try:
        from .services.telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier()

        return {
            'configured': notifier.is_configured(),
            'service': 'telegram',
            'message': 'Ready to send notifications' if notifier.is_configured() else 'Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID'
        }
    except Exception as e:
        return {
            'configured': False,
            'error': str(e)
        }


@app.post("/api/notifications/test")
async def test_notifications():
    """Send a test notification"""
    try:
        from .services.telegram_notifier import TelegramNotifier

        notifier = TelegramNotifier()

        if not notifier.is_configured():
            raise HTTPException(
                status_code=400,
                detail="Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
            )

        result = await notifier.test_connection()

        if result['success']:
            return {
                'status': 'success',
                'message': 'Test notification sent',
                'bot_name': result.get('bot_name')
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notifications/send-signal")
async def send_signal_notification(
    symbol: str = Query(..., description="Symbol to analyze and notify")
):
    """Analyze a symbol and send signal notification via Telegram"""
    try:
        from .services.telegram_notifier import TelegramNotifier
        from .services.free_market_data import FreeMarketDataService

        notifier = TelegramNotifier()

        if not notifier.is_configured():
            raise HTTPException(
                status_code=400,
                detail="Telegram not configured"
            )

        # Get analysis
        market_service = FreeMarketDataService()
        analyzer = get_analyzer()
        sig_gen = get_signal_generator()

        market_data = {}
        for tf in ['H1', 'H4', 'D1']:
            df = market_service.get_ohlcv(symbol, tf, limit=200)
            if df is not None and not df.empty:
                market_data[tf] = df

        if not market_data:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Analyze
        detected_concepts = []
        for tf, df in market_data.items():
            if analyzer:
                analysis = analyzer.analyze(df, tf)
                detected_concepts.extend(analysis.get('detected_concepts', []))

        detected_concepts = list(set(detected_concepts))

        # Generate signal
        if sig_gen:
            signal_obj = sig_gen.generate_signal(
                symbol=symbol,
                market_data=market_data,
                detected_concepts=detected_concepts
            )

            if signal_obj:
                signal_dict = {
                    'symbol': symbol,
                    'direction': signal_obj.direction,
                    'confidence': signal_obj.confidence,
                    'strength': signal_obj.strength,
                    'confluence_score': signal_obj.confluence_score,
                    'entry_zone': signal_obj.entry_zone,
                    'stop_loss': signal_obj.stop_loss,
                    'take_profit': signal_obj.take_profit,
                    'risk_reward': signal_obj.risk_reward,
                    'concepts': signal_obj.concepts,
                    'timestamp': signal_obj.timestamp,
                }

                success = await notifier.send_signal(signal_dict)

                return {
                    'status': 'success' if success else 'failed',
                    'signal': signal_dict
                }
            else:
                return {
                    'status': 'no_signal',
                    'message': 'Insufficient confluence for signal'
                }
        else:
            raise HTTPException(status_code=500, detail="Signal generator not available")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Kill Zones & Session Analysis
# ============================================================================

@app.get("/api/session/current")
async def get_current_session():
    """Get current trading session and kill zone status"""
    try:
        from .ml.kill_zones import KillZoneAnalyzer

        analyzer = KillZoneAnalyzer()
        info = analyzer.get_session_info()
        po3 = analyzer.get_power_of_three_phase()
        in_kz, kz = analyzer.is_in_kill_zone()
        next_kz, time_to_next = analyzer.get_next_kill_zone()

        return {
            'current_session': info.current_session.value,
            'in_kill_zone': in_kz,
            'kill_zone': kz.name if kz else None,
            'kill_zone_weight': kz.bias_weight if kz else 0,
            'optimal_pairs': kz.optimal_pairs if kz else [],
            'next_kill_zone': next_kz.name if next_kz else None,
            'time_to_next_kz_minutes': int(time_to_next.total_seconds() / 60) if time_to_next else None,
            'power_of_three': {
                'phase': po3['phase'],
                'description': po3['description']
            },
            'daily_bias': info.daily_bias,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/kill-zones")
async def get_all_kill_zones():
    """Get all Smart Money kill zones"""
    try:
        from .ml.kill_zones import KillZoneAnalyzer

        analyzer = KillZoneAnalyzer()

        kill_zones = []
        for kz in analyzer.KILL_ZONES:
            kill_zones.append({
                'name': kz.name,
                'session': kz.session.value,
                'start_time_utc': kz.start_time.strftime('%H:%M'),
                'end_time_utc': kz.end_time.strftime('%H:%M'),
                'bias_weight': kz.bias_weight,
                'description': kz.description,
                'optimal_pairs': kz.optimal_pairs
            })

        return {
            'kill_zones': kill_zones,
            'current_utc': datetime.utcnow().strftime('%H:%M:%S')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Performance Tracking
# ============================================================================

@app.get("/api/performance/summary")
async def get_performance_summary(
    days: int = Query(30, description="Number of days to analyze")
):
    """Get performance summary for last N days"""
    try:
        from .ml.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(str(DATA_DIR))
        summary = tracker.get_performance_summary(days=days)

        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance/feedback")
async def get_model_feedback():
    """Get model feedback and recommendations"""
    try:
        from .ml.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(str(DATA_DIR))
        feedback = tracker.get_model_feedback()

        return feedback
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance/signals")
async def get_recent_signals(
    limit: int = Query(20, description="Number of signals to return")
):
    """Get recent signal history"""
    try:
        from .ml.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(str(DATA_DIR))
        signals = tracker.get_recent_signals(limit=limit)

        return {
            'signals': signals,
            'count': len(signals)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/performance/record-signal")
async def record_signal(signal_data: dict):
    """Record a new signal for tracking"""
    try:
        from .ml.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(str(DATA_DIR))
        signal_id = tracker.record_signal(signal_data)

        return {
            'status': 'recorded',
            'signal_id': signal_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/performance/update-outcome/{signal_id}")
async def update_signal_outcome(signal_id: str, outcome: dict):
    """Update signal outcome after trade closes"""
    try:
        from .ml.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(str(DATA_DIR))
        success = tracker.update_signal_outcome(signal_id, outcome)

        if success:
            return {'status': 'updated', 'signal_id': signal_id}
        else:
            raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Training Database
# ============================================================================

@app.get("/api/training/database-report")
async def get_training_database_report():
    """Get comprehensive training database report"""
    try:
        kb = get_knowledge_base()
        if kb:
            return kb.get_training_database_report()
        else:
            from .ml.training_database import TrainingDatabase
            db = TrainingDatabase(str(DATA_DIR))
            return db.get_training_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/videos")
async def get_all_trained_videos():
    """Get all trained videos with summaries"""
    try:
        kb = get_knowledge_base()
        if kb:
            return {'videos': kb.get_all_trained_videos()}
        else:
            from .ml.training_database import TrainingDatabase
            db = TrainingDatabase(str(DATA_DIR))
            return {'videos': db.get_all_trained_videos()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/video/{video_id}")
async def get_video_training_summary(video_id: str):
    """Get training summary for a specific video"""
    try:
        kb = get_knowledge_base()
        if kb:
            summary = kb.get_video_training_summary(video_id)
        else:
            from .ml.training_database import TrainingDatabase
            db = TrainingDatabase(str(DATA_DIR))
            summary = db.get_video_summary(video_id)

        if summary:
            return summary
        else:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/playlists")
async def get_all_playlist_summaries():
    """Get all playlist training summaries"""
    try:
        kb = get_knowledge_base()
        if kb:
            return {'playlists': kb.get_all_playlists()}
        else:
            from .ml.training_database import TrainingDatabase
            db = TrainingDatabase(str(DATA_DIR))
            return {'playlists': db.get_all_playlists()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ============================================================================
# Startup/Shutdown
# ============================================================================

# ============================================================================
# Pattern Recognition
# ============================================================================

@app.get("/api/patterns/detect/{symbol}")
async def detect_patterns(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe")
):
    """Detect Smart Money patterns in price data"""
    try:
        from .services.free_market_data import FreeMarketDataService
        from .ml.pattern_recognition import SmartMoneyPatternRecognizer

        market_service = FreeMarketDataService()
        recognizer = SmartMoneyPatternRecognizer()

        df = market_service.get_ohlcv(symbol, timeframe, limit=200)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        patterns = recognizer.detect_all_patterns(df)
        summary = recognizer.get_pattern_summary(patterns)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'patterns': [p.to_dict() for p in patterns],
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Price Prediction
# ============================================================================

@app.get("/api/predict/{symbol}")
async def predict_price(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe")
):
    """Get AI price prediction for a symbol"""
    try:
        from .services.free_market_data import FreeMarketDataService
        from .ml.price_predictor import SmartMoneyPricePredictor

        market_service = FreeMarketDataService()
        predictor = SmartMoneyPricePredictor()

        df = market_service.get_ohlcv(symbol, timeframe, limit=200)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Get analysis for context
        analyzer = get_analyzer()
        analysis = analyzer.analyze(df, timeframe) if analyzer else None

        prediction = predictor.predict(df, analysis, timeframe)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'prediction': prediction.to_dict(),
            'explanation': predictor.get_prediction_explanation(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Chart Generation
# ============================================================================

@app.get("/api/chart/{symbol}")
async def generate_chart(
    symbol: str,
    timeframe: str = Query("H1", description="Timeframe"),
    with_signal: bool = Query(True, description="Include signal levels"),
    with_patterns: bool = Query(True, description="Include pattern detection")
):
    """Generate an Smart Money annotated chart"""
    try:
        from .services.free_market_data import FreeMarketDataService
        from .services.chart_generator import ICTChartGenerator
        from .ml.pattern_recognition import SmartMoneyPatternRecognizer

        market_service = FreeMarketDataService()
        chart_gen = ICTChartGenerator()

        df = market_service.get_ohlcv(symbol, timeframe, limit=100)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Get patterns
        patterns = None
        if with_patterns:
            recognizer = SmartMoneyPatternRecognizer()
            detected = recognizer.detect_all_patterns(df)
            patterns = [p.to_dict() for p in detected]

        # Get signal
        signal = None
        if with_signal:
            analyzer = get_analyzer()
            sig_gen = get_signal_generator()
            if analyzer and sig_gen:
                analysis = analyzer.analyze(df, timeframe)
                signal_obj = sig_gen.generate_signal(
                    symbol=symbol,
                    market_data={timeframe: df},
                    detected_concepts=analysis.get('detected_concepts', [])
                )
                if signal_obj:
                    signal = {
                        'direction': signal_obj.direction,
                        'confidence': signal_obj.confidence,
                        'entry_price': float(df['close'].iloc[-1]),
                        'entry_zone': signal_obj.entry_zone,
                        'stop_loss': signal_obj.stop_loss,
                        'take_profit': signal_obj.take_profit,
                    }

        # Generate chart
        chart_base64 = chart_gen.generate_candlestick_chart(
            data=df,
            symbol=symbol,
            timeframe=timeframe,
            signal=signal,
            patterns=patterns
        )

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'chart': chart_base64,
            'signal': signal,
            'pattern_count': len(patterns) if patterns else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Live Price WebSocket
# ============================================================================

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, symbol: str):
        await websocket.accept()
        if symbol not in self.active_connections:
            self.active_connections[symbol] = set()
        self.active_connections[symbol].add(websocket)

    def disconnect(self, websocket: WebSocket, symbol: str):
        if symbol in self.active_connections:
            self.active_connections[symbol].discard(websocket)

    async def broadcast(self, symbol: str, data: dict):
        if symbol in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[symbol]:
                try:
                    await connection.send_json(data)
                except:
                    disconnected.add(connection)
            for conn in disconnected:
                self.active_connections[symbol].discard(conn)

manager = ConnectionManager()


@app.websocket("/ws/live/{symbol}")
async def websocket_live_price(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for live price updates with real-time USDT Perpetual futures data"""
    import httpx

    await manager.connect(websocket, symbol)

    # Crypto symbols that can use Binance Futures real-time data
    # Support both formats: BTC -> BTCUSDT, or direct BTCUSDT
    CRYPTO_BINANCE_MAP = {
        'BTC': 'BTCUSDT',
        'BTCUSDT': 'BTCUSDT',
        'ETH': 'ETHUSDT',
        'ETHUSDT': 'ETHUSDT',
        'SOL': 'SOLUSDT',
        'SOLUSDT': 'SOLUSDT',
        'XRP': 'XRPUSDT',
        'XRPUSDT': 'XRPUSDT',
        'DOGE': 'DOGEUSDT',
        'DOGEUSDT': 'DOGEUSDT',
        'ADA': 'ADAUSDT',
        'ADAUSDT': 'ADAUSDT',
        'AVAX': 'AVAXUSDT',
        'AVAXUSDT': 'AVAXUSDT',
        'DOT': 'DOTUSDT',
        'DOTUSDT': 'DOTUSDT',
        'LINK': 'LINKUSDT',
        'LINKUSDT': 'LINKUSDT',
        'MATIC': 'MATICUSDT',
        'MATICUSDT': 'MATICUSDT',
    }

    is_crypto = symbol.upper() in CRYPTO_BINANCE_MAP
    binance_symbol = CRYPTO_BINANCE_MAP.get(symbol.upper())

    # Track candle data
    last_price = None
    last_high = None
    last_low = None
    last_open = None
    candle_minute = None

    try:
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    if is_crypto and binance_symbol:
                        # Use Binance Futures API for real-time USDT Perpetual prices
                        # This matches TradingView's BTCUSDT.P (perpetual) chart
                        response = await client.get(
                            f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={binance_symbol}",
                            timeout=5.0
                        )
                        if response.status_code == 200:
                            data = response.json()
                            current_price = float(data['price'])

                            # Track candle data (reset every minute)
                            current_minute = datetime.utcnow().minute
                            if candle_minute != current_minute:
                                # New candle
                                candle_minute = current_minute
                                last_open = current_price
                                last_high = current_price
                                last_low = current_price
                            else:
                                # Update high/low
                                last_high = max(last_high, current_price) if last_high else current_price
                                last_low = min(last_low, current_price) if last_low else current_price

                            last_price = current_price

                            await websocket.send_json({
                                'type': 'price_update',
                                'symbol': symbol,
                                'timestamp': datetime.utcnow().isoformat(),
                                'open': last_open or current_price,
                                'high': last_high or current_price,
                                'low': last_low or current_price,
                                'close': current_price,
                                'volume': 0
                            })
                    else:
                        # For non-crypto, use Yahoo Finance (delayed)
                        from .services.free_market_data import FreeMarketDataService
                        market_service = FreeMarketDataService()
                        df = market_service.get_ohlcv(symbol, 'M1', limit=1)
                        if df is not None and not df.empty:
                            latest = df.iloc[-1]
                            await websocket.send_json({
                                'type': 'price_update',
                                'symbol': symbol,
                                'timestamp': datetime.utcnow().isoformat(),
                                'open': float(latest['open']),
                                'high': float(latest['high']),
                                'low': float(latest['low']),
                                'close': float(latest['close']),
                                'volume': float(latest['volume']) if 'volume' in latest else 0
                            })

                except Exception as e:
                    print(f"Price fetch error for {symbol}: {e}")

                # Update every 1 second for crypto, 5 seconds for others
                await asyncio.sleep(1 if is_crypto else 5)
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
    except Exception as e:
        manager.disconnect(websocket, symbol)


@app.get("/api/live/ohlcv/{symbol}")
async def get_live_ohlcv(
    symbol: str,
    timeframe: str = Query("M1", description="Timeframe"),
    limit: int = Query(100, description="Number of candles")
):
    """Get OHLCV data for live charting - uses Binance Futures for crypto"""
    import httpx

    # Crypto symbols that use Binance Futures
    CRYPTO_BINANCE_MAP = {
        'BTC': 'BTCUSDT', 'BTCUSDT': 'BTCUSDT',
        'ETH': 'ETHUSDT', 'ETHUSDT': 'ETHUSDT',
        'SOL': 'SOLUSDT', 'SOLUSDT': 'SOLUSDT',
        'XRP': 'XRPUSDT', 'XRPUSDT': 'XRPUSDT',
        'DOGE': 'DOGEUSDT', 'DOGEUSDT': 'DOGEUSDT',
        'ADA': 'ADAUSDT', 'ADAUSDT': 'ADAUSDT',
        'AVAX': 'AVAXUSDT', 'AVAXUSDT': 'AVAXUSDT',
        'DOT': 'DOTUSDT', 'DOTUSDT': 'DOTUSDT',
        'LINK': 'LINKUSDT', 'LINKUSDT': 'LINKUSDT',
        'MATIC': 'MATICUSDT', 'MATICUSDT': 'MATICUSDT',
    }

    # Binance Futures timeframe mapping
    BINANCE_TF_MAP = {
        'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m',
        'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w', 'MN': '1M'
    }

    is_crypto = symbol.upper() in CRYPTO_BINANCE_MAP
    binance_symbol = CRYPTO_BINANCE_MAP.get(symbol.upper())
    binance_interval = BINANCE_TF_MAP.get(timeframe, '1h')

    try:
        if is_crypto and binance_symbol:
            # Use Binance Futures API for crypto OHLCV data
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={
                        'symbol': binance_symbol,
                        'interval': binance_interval,
                        'limit': limit
                    },
                    timeout=10.0
                )

                if response.status_code == 200:
                    klines = response.json()
                    candles = []
                    for k in klines:
                        # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
                        candles.append({
                            'time': int(k[0] / 1000),  # Convert ms to seconds
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5])
                        })

                    return {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'candles': candles,
                        'source': 'binance_futures'
                    }
                else:
                    raise HTTPException(status_code=response.status_code, detail=f"Binance API error: {response.text}")
        else:
            # Use Yahoo Finance for non-crypto (forex, indices, stocks)
            from .services.free_market_data import FreeMarketDataService

            market_service = FreeMarketDataService()
            df = market_service.get_ohlcv(symbol, timeframe, limit=limit)

            if df is None or df.empty:
                raise HTTPException(status_code=404, detail=f"No data for {symbol}")

            candles = []
            for idx, row in df.iterrows():
                candles.append({
                    'time': int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(datetime.now().timestamp()) - (len(df) - len(candles)) * 60,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 0
                })

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'candles': candles,
                'source': 'yahoo_finance'
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Scheduler Control
# ============================================================================

@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get signal scheduler status"""
    try:
        from .services.signal_scheduler import signal_scheduler
        return signal_scheduler.get_schedule_info()
    except Exception as e:
        return {
            'status': 'not_available',
            'error': str(e)
        }


@app.post("/api/scheduler/start")
async def start_scheduler():
    """Start the signal scheduler"""
    try:
        from .services.signal_scheduler import signal_scheduler
        success = signal_scheduler.start()
        return {
            'status': 'started' if success else 'failed',
            'info': signal_scheduler.get_schedule_info()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scheduler/stop")
async def stop_scheduler():
    """Stop the signal scheduler"""
    try:
        from .services.signal_scheduler import signal_scheduler
        signal_scheduler.stop()
        return {'status': 'stopped'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scheduler/run-now")
async def run_scheduler_now(
    timeframe: str = Query("H1", description="Timeframe to check")
):
    """Manually trigger a signal check"""
    try:
        from .services.signal_scheduler import run_manual_check
        signals = await run_manual_check(timeframe)
        return {
            'status': 'completed',
            'signals_generated': len(signals),
            'signals': signals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Database Operations
# ============================================================================

@app.get("/api/db/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        from .database import db
        return {
            'performance': db.get_performance_summary(),
            'training': db.get_training_stats(),
            'status': 'connected'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@app.get("/api/db/signals")
async def get_db_signals(
    symbol: str = Query(None, description="Filter by symbol"),
    status: str = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Max results")
):
    """Get signals from database"""
    try:
        from .database import db
        signals = db.get_signals(symbol=symbol, status=status, limit=limit)
        return {
            'signals': signals,
            'count': len(signals)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Playlist Processing with Live Progress
# ============================================================================

import asyncio
import re
from fastapi.responses import StreamingResponse
import queue
import threading

# Global progress tracking
_processing_status = {}


def extract_playlist_id(url: str) -> str:
    """Extract playlist ID from URL"""
    patterns = [
        r'list=([a-zA-Z0-9_-]+)',
        r'playlist\?list=([a-zA-Z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_playlist_info_sync(playlist_url: str) -> dict:
    """Fetch playlist information from YouTube (synchronous)"""
    try:
        import yt_dlp

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'skip_download': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        videos = []
        for entry in info.get('entries', []):
            if entry:
                videos.append({
                    'video_id': entry.get('id'),
                    'title': entry.get('title', 'Unknown'),
                    'duration': entry.get('duration'),
                    'url': f"https://www.youtube.com/watch?v={entry.get('id')}"
                })

        return {
            'playlist_id': info.get('id'),
            'title': info.get('title', 'Unknown Playlist'),
            'channel': info.get('channel', info.get('uploader', 'Unknown')),
            'video_count': len(videos),
            'videos': videos,
            'fetched_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}


def transcribe_with_whisper(video_id: str, status_callback=None) -> dict:
    """Download audio and transcribe with Whisper (fallback for videos without captions)"""
    import tempfile
    import os

    # Add user's bin directory to PATH for ffmpeg
    user_bin = os.path.expanduser("~/bin")
    if user_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{user_bin}:{os.environ.get('PATH', '')}"

    try:
        import yt_dlp
        import whisper
    except ImportError as e:
        return {'error': f'Missing dependency: {e}. Install with: pip install openai-whisper yt-dlp'}

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Create temp directory for audio
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, f"{video_id}.mp3")

        # Download audio only
        if status_callback:
            status_callback('downloading_audio', 'Downloading audio...')

        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
            'outtmpl': os.path.join(temp_dir, f"{video_id}.%(ext)s"),
            # No postprocessors - Whisper can handle m4a/webm directly
            'quiet': True,
            'no_warnings': True,
            # Bypass bot detection
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            return {'error': f'Failed to download audio: {str(e)}'}

        # Find the downloaded file (extension might vary)
        audio_file = None
        for f in os.listdir(temp_dir):
            if f.startswith(video_id):
                audio_file = os.path.join(temp_dir, f)
                break

        if not audio_file or not os.path.exists(audio_file):
            return {'error': 'Audio download failed - file not found'}

        # Load Whisper model (use base for speed, large-v3 for accuracy)
        if status_callback:
            status_callback('loading_model', 'Loading Whisper model...')

        try:
            # Use 'base' model for faster processing, 'large-v3' for best accuracy
            model = whisper.load_model("base")
        except Exception as e:
            return {'error': f'Failed to load Whisper model: {str(e)}'}

        # Transcribe
        if status_callback:
            status_callback('transcribing', 'Transcribing with Whisper...')

        try:
            result = model.transcribe(audio_file, language='en')
        except Exception as e:
            return {'error': f'Transcription failed: {str(e)}'}

        # Process segments
        segments = []
        for seg in result.get('segments', []):
            segments.append({
                'start_time': round(seg['start'], 2),
                'end_time': round(seg['end'], 2),
                'text': seg['text'].strip()
            })

        full_text = result.get('text', '').strip()

        return {
            'video_id': video_id,
            'full_text': full_text,
            'segments': segments,
            'word_count': len(full_text.split()),
            'segment_count': len(segments),
            'method': 'whisper',
            'whisper_model': 'base',
            'transcribed_at': datetime.utcnow().isoformat()
        }


def get_transcript_sync(video_id: str, status_callback=None) -> dict:
    """Fetch transcript for a video - tries YouTube captions first, falls back to Whisper"""
    # First try YouTube's built-in captions (instant)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt_api = YouTubeTranscriptApi()
        result = ytt_api.fetch(video_id)

        segments = []
        for item in result:
            segments.append({
                'start_time': round(item.start, 2),
                'end_time': round(item.start + item.duration, 2),
                'text': item.text.strip()
            })

        full_text = ' '.join([s['text'] for s in segments])

        return {
            'video_id': video_id,
            'full_text': full_text,
            'segments': segments,
            'word_count': len(full_text.split()),
            'segment_count': len(segments),
            'method': 'youtube_transcript_api',
            'transcribed_at': datetime.utcnow().isoformat()
        }
    except Exception as yt_error:
        # YouTube captions not available, try Whisper fallback
        if status_callback:
            status_callback('whisper_fallback', f'No captions available, using Whisper...')

        whisper_result = transcribe_with_whisper(video_id, status_callback)

        if 'error' in whisper_result:
            # Both methods failed
            return {'error': f'YouTube: {str(yt_error)[:50]} | Whisper: {whisper_result["error"][:50]}'}

        return whisper_result


def process_playlist_worker(job_id: str, playlist_url: str, tier: int, train_after: bool):
    """Background worker to process playlist"""
    global _processing_status

    try:
        # Update status - fetching playlist
        _processing_status[job_id] = {
            'status': 'fetching_playlist',
            'message': 'Fetching playlist information...',
            'progress': 0,
            'total': 0,
            'current_video': None,
            'completed': [],
            'failed': [],
            'started_at': datetime.utcnow().isoformat()
        }

        # Fetch playlist info
        playlist_info = fetch_playlist_info_sync(playlist_url)

        if 'error' in playlist_info:
            _processing_status[job_id]['status'] = 'error'
            _processing_status[job_id]['message'] = f"Failed to fetch playlist: {playlist_info['error']}"
            return

        # Save playlist
        playlist_info['tier'] = tier
        playlist_info['description'] = f"Added via Dashboard - Tier {tier}"
        playlist_info['added_at'] = datetime.utcnow().isoformat()

        save_path = PLAYLISTS_DIR / f"{playlist_info['playlist_id']}.json"
        with open(save_path, 'w') as f:
            json.dump(playlist_info, f, indent=2)

        videos = playlist_info['videos']
        total = len(videos)

        _processing_status[job_id].update({
            'status': 'processing',
            'message': f'Processing {total} videos...',
            'playlist_title': playlist_info['title'],
            'playlist_id': playlist_info['playlist_id'],
            'total': total,
            'progress': 0
        })

        # Process each video
        for i, video in enumerate(videos):
            video_id = video['video_id']
            video_title = video['title']

            _processing_status[job_id]['current_video'] = {
                'index': i + 1,
                'video_id': video_id,
                'title': video_title[:50]
            }
            _processing_status[job_id]['message'] = f'Processing: {video_title[:40]}...'

            transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

            # Check if already exists
            if transcript_path.exists():
                _processing_status[job_id]['completed'].append({
                    'video_id': video_id,
                    'title': video_title,
                    'status': 'skipped',
                    'reason': 'Already transcribed'
                })
            else:
                # Create status callback to update UI during Whisper transcription
                def whisper_status_callback(step, message):
                    _processing_status[job_id]['whisper_step'] = step
                    _processing_status[job_id]['message'] = f'{video_title[:30]}... - {message}'

                # Get transcript (tries YouTube first, falls back to Whisper)
                transcript = get_transcript_sync(video_id, whisper_status_callback)

                # Clear whisper step after completion
                _processing_status[job_id].pop('whisper_step', None)

                if 'error' in transcript:
                    _processing_status[job_id]['failed'].append({
                        'video_id': video_id,
                        'title': video_title,
                        'error': transcript['error'][:100]
                    })
                else:
                    transcript['title'] = video_title
                    with open(transcript_path, 'w') as f:
                        json.dump(transcript, f, indent=2, ensure_ascii=False)

                    _processing_status[job_id]['completed'].append({
                        'video_id': video_id,
                        'title': video_title,
                        'status': 'success',
                        'method': transcript.get('method', 'unknown'),
                        'word_count': transcript['word_count']
                    })

            _processing_status[job_id]['progress'] = i + 1

        # Training phase
        if train_after:
            _processing_status[job_id]['status'] = 'training'
            _processing_status[job_id]['message'] = 'Training ML model with new data...'
            _processing_status[job_id]['current_video'] = None

            try:
                from .ml.training_pipeline import SmartMoneyKnowledgeBase

                kb = SmartMoneyKnowledgeBase(str(DATA_DIR))
                try:
                    kb.load()
                except:
                    pass

                results = kb.train(incremental=True)
                kb.save()

                # Reload global instance
                global _knowledge_base
                _knowledge_base = kb

                _processing_status[job_id]['training_results'] = {
                    'n_transcripts': results.get('n_transcripts', 0),
                    'classifier_f1': results.get('components', {}).get('classifier', {}).get('ensemble_f1'),
                }
            except Exception as e:
                _processing_status[job_id]['training_error'] = str(e)

        # Complete
        _processing_status[job_id]['status'] = 'completed'
        _processing_status[job_id]['message'] = 'Processing complete!'
        _processing_status[job_id]['completed_at'] = datetime.utcnow().isoformat()

    except Exception as e:
        _processing_status[job_id]['status'] = 'error'
        _processing_status[job_id]['message'] = f'Error: {str(e)}'


@app.post("/api/playlist/add")
async def add_playlist_endpoint(
    url: str = Query(..., description="YouTube playlist URL"),
    tier: int = Query(3, description="Learning tier 1-5"),
    train_after: bool = Query(True, description="Train model after processing")
):
    """
    Add a YouTube playlist and process all videos.
    Returns a job ID for tracking progress.
    """
    # Validate URL
    playlist_id = extract_playlist_id(url)
    if not playlist_id:
        raise HTTPException(status_code=400, detail="Invalid playlist URL")

    # Check if already exists
    existing = PLAYLISTS_DIR / f"{playlist_id}.json"
    if existing.exists():
        return {
            'status': 'exists',
            'message': 'Playlist already added',
            'playlist_id': playlist_id
        }

    # Create job
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{playlist_id[:8]}"

    # Start background processing
    thread = threading.Thread(
        target=process_playlist_worker,
        args=(job_id, url, tier, train_after)
    )
    thread.daemon = True
    thread.start()

    return {
        'status': 'started',
        'job_id': job_id,
        'message': 'Processing started. Use /api/playlist/status/{job_id} to track progress.'
    }


@app.get("/api/playlist/status/{job_id}")
async def get_playlist_status(job_id: str):
    """Get processing status for a playlist job"""
    if job_id not in _processing_status:
        raise HTTPException(status_code=404, detail="Job not found")

    return _processing_status[job_id]


@app.get("/api/playlist/stream/{job_id}")
async def stream_playlist_progress(job_id: str):
    """Stream processing progress using Server-Sent Events"""

    async def event_generator():
        last_hash = None

        while True:
            if job_id not in _processing_status:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            status = _processing_status[job_id]

            # Create a hash of important fields to detect any change
            current_hash = (
                status.get('progress', 0),
                status.get('status'),
                status.get('message', ''),
                status.get('whisper_step', ''),
                len(status.get('completed', [])),
                len(status.get('failed', []))
            )

            # Send update if anything changed
            if current_hash != last_hash:
                yield f"data: {json.dumps(status)}\n\n"
                last_hash = current_hash

            # Stop if completed or error
            if status.get('status') in ['completed', 'error']:
                break

            await asyncio.sleep(0.3)  # Faster updates

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/api/playlist/jobs")
async def list_processing_jobs():
    """List all processing jobs"""
    return {
        'jobs': [
            {
                'job_id': job_id,
                'status': status.get('status'),
                'playlist_title': status.get('playlist_title'),
                'progress': status.get('progress', 0),
                'total': status.get('total', 0),
                'started_at': status.get('started_at')
            }
            for job_id, status in _processing_status.items()
        ]
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("🚀 TradingMamba API starting...")
    print(f"📁 Data directory: {DATA_DIR}")

    # Initialize database
    try:
        from .database import db
        print("✅ SQLite database initialized")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLAYLISTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("👋 TradingMamba API shutting down...")
