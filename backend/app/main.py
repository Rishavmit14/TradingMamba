"""
Smart Money AI Trading System - FastAPI Application

Main entry point for the backend API.

Performance Optimizations:
- uvloop: 2-4x faster async event loop (Apple Silicon optimized)
- orjson: 6x faster JSON serialization
"""

# Enable uvloop for 2-4x faster async performance
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass  # uvloop not available, use default event loop

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Set
from datetime import datetime
from .utils import json_utils as json  # 6x faster than standard json (orjson-based)
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


@app.get("/api/playlists/available")
async def get_available_playlists():
    """
    List all playlists with training stats for the LiveChart dropdown.

    Returns list of playlists with:
    - id, title, channel, video_count
    - trained_video_count (how many videos have been ML-trained)
    - concepts_learned (what concepts this playlist teaches)
    - concepts_count
    """
    try:
        from .ml.playlist_registry import PlaylistRegistry
        return PlaylistRegistry.get_available_playlists()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    timeframes: str = Query("H1,H4,D1", description="Comma-separated timeframes"),
    playlist_id: str = Query("all", description="Playlist ID to scope ML knowledge. 'all' = combined.")
):
    """
    Analyze a symbol using ML-POWERED Smart Money methodology.

    IMPORTANT: This endpoint uses ONLY patterns the ML has learned from video training.
    Patterns not learned will not be detected.

    Args:
        playlist_id: Scope analysis to a specific playlist's knowledge.
                     'all' = use all trained knowledge (default, backward compatible).

    Returns:
    - ML patterns detected (from training)
    - ML patterns not yet learned (needs training)
    - Confidence scores based on ML training frequency
    """
    try:
        from .services.free_market_data import FreeMarketDataService
        from .models.signal import Timeframe
        from .ml.playlist_registry import PlaylistRegistry, playlist_context

        # Set playlist context for deep-stack scoping (signal_fusion, feature_engineering)
        playlist_context.set(playlist_id)

        market_service = FreeMarketDataService()
        ml_engine = PlaylistRegistry.get_ml_engine(playlist_id)
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

        # Use ML-powered analyzer (playlist-scoped)
        ml_analyzer = PlaylistRegistry.get_analyzer(playlist_id)
        ml_sig_gen = PlaylistRegistry.get_signal_generator(playlist_id)

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
                        'high': float(ob.high),
                        'low': float(ob.low),
                        'start_index': int(ob.start_index),
                        'timeframe': tf,
                        'strength': float(ob.strength) if ob.strength else 0.5,
                        'mitigated': bool(ob.mitigated),
                    })

                # Mitigated order blocks (faded visualization)
                for mob in analysis.mitigated_order_blocks:
                    all_patterns.append({
                        'pattern_type': f'{mob.type}_mitigation_block',
                        'high': float(mob.high),
                        'low': float(mob.low),
                        'start_index': int(mob.start_index),
                        'timeframe': tf,
                        'mitigated': True,
                    })

                for fvg in analysis.fair_value_gaps:
                    if fvg.filled:
                        continue  # Skip filled FVGs
                    all_patterns.append({
                        'pattern_type': f'{fvg.type}_fvg',
                        'high': float(fvg.high),
                        'low': float(fvg.low),
                        'start_index': int(fvg.index),
                        'timeframe': tf,
                        'filled': False,
                        'fill_percentage': float(fvg.fill_percentage) if fvg.fill_percentage else 0.0,
                    })

                # Also add market structure events (BOS/CHoCH + HH/HL/LH/LL)
                for event in analysis.structure_events:
                    if event.type in ['bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
                                       'higher_high', 'higher_low', 'lower_high', 'lower_low']:
                        all_patterns.append({
                            'pattern_type': event.type,
                            'price': float(event.level),
                            'timeframe': tf,
                            'description': event.description,
                        })

                # Add liquidity levels (equal highs/lows)
                if analysis.liquidity_levels.get('equal_highs'):
                    for eq in analysis.liquidity_levels['equal_highs']:
                        all_patterns.append({
                            'pattern_type': 'equal_highs',
                            'price': float(eq['level']),
                            'timeframe': tf,
                        })
                if analysis.liquidity_levels.get('equal_lows'):
                    for eq in analysis.liquidity_levels['equal_lows']:
                        all_patterns.append({
                            'pattern_type': 'equal_lows',
                            'price': float(eq['level']),
                            'timeframe': tf,
                        })

                # Add buy-side and sell-side liquidity levels
                if analysis.liquidity_levels.get('buy_side'):
                    for liq in analysis.liquidity_levels['buy_side'][:3]:  # Top 3 BSL
                        all_patterns.append({
                            'pattern_type': 'buyside_liquidity',
                            'price': float(liq.price),
                            'timeframe': tf,
                            'strength': float(liq.strength) if liq.strength else 0.5,
                        })
                if analysis.liquidity_levels.get('sell_side'):
                    for liq in analysis.liquidity_levels['sell_side'][:3]:  # Top 3 SSL
                        all_patterns.append({
                            'pattern_type': 'sellside_liquidity',
                            'price': float(liq.price),
                            'timeframe': tf,
                            'strength': float(liq.strength) if liq.strength else 0.5,
                        })

                # ============================================================
                # NEW ICT PATTERNS (from Audio-First Training)
                # ============================================================

                # Displacement candles
                for disp in analysis.displacements:
                    all_patterns.append({
                        'pattern_type': f'{disp["type"]}_displacement',
                        'high': float(disp['high']),
                        'low': float(disp['low']),
                        'price': float((disp['high'] + disp['low']) / 2),
                        'start_index': int(disp.get('index', 0)),
                        'timeframe': tf,
                        'body_ratio': float(disp.get('body_ratio', 0)),
                    })

                # OTE zones (Optimal Trade Entry / Fibonacci)
                for ote in analysis.ote_zones:
                    all_patterns.append({
                        'pattern_type': f'{ote["type"]}_ote',
                        'high': float(ote['ote_high']),
                        'low': float(ote['ote_low']),
                        'timeframe': tf,
                        'fib_62': float(ote.get('fib_62', 0)),
                        'fib_79': float(ote.get('fib_79', 0)),
                        'price_in_ote': ote.get('price_in_ote', False),
                    })

                # Breaker blocks
                for bb in analysis.breaker_blocks:
                    all_patterns.append({
                        'pattern_type': bb['type'],  # 'bullish_breaker' or 'bearish_breaker'
                        'high': float(bb['high']),
                        'low': float(bb['low']),
                        'start_index': int(bb.get('index', 0)),
                        'timeframe': tf,
                        'being_tested': bb.get('being_tested', False),
                    })

                # Buy/sell stops
                if analysis.buy_sell_stops.get('buy_stops'):
                    for bs in analysis.buy_sell_stops['buy_stops'][:3]:
                        all_patterns.append({
                            'pattern_type': 'buy_stops',
                            'price': float(bs['level']),
                            'timeframe': tf,
                            'distance_pct': float(bs.get('distance_pct', 0)),
                        })
                if analysis.buy_sell_stops.get('sell_stops'):
                    for ss in analysis.buy_sell_stops['sell_stops'][:3]:
                        all_patterns.append({
                            'pattern_type': 'sell_stops',
                            'price': float(ss['level']),
                            'timeframe': tf,
                            'distance_pct': float(ss.get('distance_pct', 0)),
                        })

                # Inducement zones
                for idm in analysis.inducements:
                    idm_entry = {
                        'pattern_type': f'{idm["type"]}_inducement',
                        'high': float(idm['high']),
                        'low': float(idm['low']),
                        'price': float(idm['price']),
                        'start_index': int(idm.get('index', 0)),
                        'timeframe': tf,
                        'taken_out': idm.get('taken_out', False),
                    }
                    # Include unix timestamp for accurate chart positioning
                    ts = idm.get('timestamp')
                    if ts is not None:
                        import pandas as pd
                        if hasattr(ts, 'timestamp'):
                            idm_entry['time'] = int(ts.timestamp())
                        elif isinstance(ts, (int, float)):
                            idm_entry['time'] = int(ts)
                    all_patterns.append(idm_entry)

                # Swing point markers (most recent 3 highs + 3 lows)
                swing_highs = [sp for sp in analysis.swing_points if sp.type == 'high'][-3:]
                swing_lows = [sp for sp in analysis.swing_points if sp.type == 'low'][-3:]
                for sp in swing_highs + swing_lows:
                    all_patterns.append({
                        'pattern_type': f'swing_{sp.type}',  # 'swing_high' or 'swing_low'
                        'price': float(sp.price),
                        'start_index': int(sp.index),
                        'timeframe': tf,
                        'strength': int(sp.strength),
                    })

                # Premium/Discount zone overlay
                pd_data = analysis.premium_discount
                if pd_data.get('range_high') and pd_data.get('range_low') and pd_data.get('equilibrium'):
                    all_patterns.append({
                        'pattern_type': 'premium_zone',
                        'high': float(pd_data['range_high']),
                        'low': float(pd_data['equilibrium']),
                        'timeframe': tf,
                    })
                    all_patterns.append({
                        'pattern_type': 'discount_zone',
                        'high': float(pd_data['equilibrium']),
                        'low': float(pd_data['range_low']),
                        'timeframe': tf,
                    })
                    all_patterns.append({
                        'pattern_type': 'equilibrium',
                        'price': float(pd_data['equilibrium']),
                        'timeframe': tf,
                    })

                analyses[tf] = {
                    'bias': analysis.bias.value,
                    'bias_confidence': float(analysis.bias_confidence),
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

        # Get ML Engine for reasoning generation (use playlist-scoped engine)
        # ml_engine already set above from PlaylistRegistry

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

        # ============================================================
        # QUANT TIER DATA (Tiers 1-5)
        # ============================================================
        quant_data = {}
        try:
            # Tier 1: Backtest performance
            from .ml.backtester import get_backtester
            bt = get_backtester()
            bt_perf = bt.get_aggregate_performance()
            if bt_perf:
                quant_data['backtest'] = bt_perf

            # Tier 3: ML classifier predictions
            try:
                from .ml.ml_models import get_classifier
                classifier = get_classifier()
                if classifier.models:
                    quant_data['ml_classifier'] = {
                        'models_trained': len(classifier.models),
                        'pattern_types': list(classifier.models.keys()),
                    }
            except Exception:
                pass

            # Tier 5: Regime detection
            try:
                from .ml.quant_engine import get_quant_engine
                engine = get_quant_engine()
                primary_df = list(market_data.values())[0]
                from .ml.quant_engine import RegimeDetector
                detector = RegimeDetector()
                regime_info = detector.detect(primary_df)
                quant_data['regime'] = regime_info
            except Exception:
                pass

        except Exception as e:
            pass  # Quant data enrichment is optional

        # ============================================================
        # VIDEO KNOWLEDGE ENRICHMENT
        # Add teaching depth and co-occurrence data for detected patterns
        # ============================================================
        video_knowledge_data = {}
        try:
            vk = PlaylistRegistry.get_video_knowledge(playlist_id)
            if vk.is_loaded():
                # Per-pattern teaching info
                pattern_teaching = {}
                for p in all_ml_patterns_used:
                    profile = vk.get_concept_profile(p)
                    if profile:
                        pattern_teaching[p] = {
                            'teaching_depth': profile.teaching_depth,
                            'video_count': profile.video_count,
                            'total_words': profile.total_words,
                            'depth_score': round(vk.get_teaching_depth_score(p), 3),
                        }
                # Co-occurrence for detected pattern pairs
                co_occurrences = []
                for i, a in enumerate(all_ml_patterns_used):
                    for b in all_ml_patterns_used[i+1:]:
                        score = vk.get_co_occurrence(a, b)
                        if score > 0.1:
                            co_occurrences.append({'a': a, 'b': b, 'score': round(score, 3)})
                co_occurrences.sort(key=lambda x: x['score'], reverse=True)

                video_knowledge_data = {
                    'active': True,
                    'concepts_loaded': len(vk.concept_profiles),
                    'videos_loaded': vk._n_videos,
                    'pattern_teaching': pattern_teaching,
                    'co_occurrences': co_occurrences[:5],
                    'features_active': 22,
                }
        except Exception:
            video_knowledge_data = {'active': False}

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
            },
            # Quant Engine Data (Tiers 1-5)
            'quant': quant_data,
            # Video Knowledge → ML Features data
            'video_knowledge': video_knowledge_data,
        }

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals/quick/{symbol}")
async def quick_signal(
    symbol: str,
    playlist_id: str = Query("all", description="Playlist ID to scope ML knowledge")
):
    """
    Get a quick ML-powered signal for a symbol (single timeframe analysis).

    Uses ONLY patterns the ML has learned from video training.
    Returns ML knowledge status showing what patterns were/weren't detected.
    """
    try:
        from .services.free_market_data import FreeMarketDataService
        from .ml.playlist_registry import PlaylistRegistry, playlist_context

        playlist_context.set(playlist_id)
        market_service = FreeMarketDataService()
        ml_engine = PlaylistRegistry.get_ml_engine(playlist_id)
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

        # Use ML-powered analyzer (playlist-scoped)
        ml_analyzer = PlaylistRegistry.get_analyzer(playlist_id)

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
                if fvg.filled:
                    continue  # Skip filled FVGs
                patterns.append({
                    'pattern_type': f'{fvg.type}_fvg',
                    'high': fvg.high,
                    'low': fvg.low,
                    'start_index': fvg.index,
                    'timeframe': 'H1',
                })

            # Add market structure events (BOS/CHoCH)
            for event in analysis.structure_events:
                if event.type in ['bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish']:
                    patterns.append({
                        'pattern_type': event.type,
                        'price': float(event.level),
                        'timeframe': 'H1',
                        'description': event.description,
                    })

            # Add equal highs/lows
            if analysis.liquidity_levels.get('equal_highs'):
                for eq in analysis.liquidity_levels['equal_highs']:
                    patterns.append({
                        'pattern_type': 'equal_highs',
                        'price': float(eq['level']),
                        'timeframe': 'H1',
                    })
            if analysis.liquidity_levels.get('equal_lows'):
                for eq in analysis.liquidity_levels['equal_lows']:
                    patterns.append({
                        'pattern_type': 'equal_lows',
                        'price': float(eq['level']),
                        'timeframe': 'H1',
                    })

            # Add buy-side and sell-side liquidity
            if analysis.liquidity_levels.get('buy_side'):
                for liq in analysis.liquidity_levels['buy_side'][:3]:
                    patterns.append({
                        'pattern_type': 'buyside_liquidity',
                        'price': float(liq.price),
                        'timeframe': 'H1',
                        'strength': float(liq.strength) if liq.strength else 0.5,
                    })
            if analysis.liquidity_levels.get('sell_side'):
                for liq in analysis.liquidity_levels['sell_side'][:3]:
                    patterns.append({
                        'pattern_type': 'sellside_liquidity',
                        'price': float(liq.price),
                        'timeframe': 'H1',
                        'strength': float(liq.strength) if liq.strength else 0.5,
                    })

            # NEW ICT PATTERNS from Audio-First Training

            # Displacement candles
            for disp in analysis.displacements:
                patterns.append({
                    'pattern_type': f'{disp["type"]}_displacement',
                    'high': float(disp['high']),
                    'low': float(disp['low']),
                    'price': float((disp['high'] + disp['low']) / 2),
                    'start_index': int(disp.get('index', 0)),
                    'timeframe': 'H1',
                    'body_ratio': float(disp.get('body_ratio', 0)),
                })

            # OTE zones (Optimal Trade Entry)
            for ote in analysis.ote_zones:
                patterns.append({
                    'pattern_type': f'{ote["type"]}_ote',
                    'high': float(ote['ote_high']),
                    'low': float(ote['ote_low']),
                    'timeframe': 'H1',
                    'fib_62': float(ote.get('fib_62', 0)),
                    'fib_79': float(ote.get('fib_79', 0)),
                    'price_in_ote': ote.get('price_in_ote', False),
                })

            # Breaker blocks
            for bb in analysis.breaker_blocks:
                patterns.append({
                    'pattern_type': bb['type'],  # 'bullish_breaker' or 'bearish_breaker'
                    'high': float(bb['high']),
                    'low': float(bb['low']),
                    'start_index': int(bb.get('index', 0)),
                    'timeframe': 'H1',
                    'being_tested': bb.get('being_tested', False),
                })

            # Buy/sell stops
            if analysis.buy_sell_stops.get('buy_stops'):
                for bs in analysis.buy_sell_stops['buy_stops'][:3]:
                    patterns.append({
                        'pattern_type': 'buy_stops',
                        'price': float(bs['level']),
                        'timeframe': 'H1',
                        'distance_pct': float(bs.get('distance_pct', 0)),
                    })
            if analysis.buy_sell_stops.get('sell_stops'):
                for ss in analysis.buy_sell_stops['sell_stops'][:3]:
                    patterns.append({
                        'pattern_type': 'sell_stops',
                        'price': float(ss['level']),
                        'timeframe': 'H1',
                        'distance_pct': float(ss.get('distance_pct', 0)),
                    })

            # Inducement zones
            for idm in analysis.inducements:
                patterns.append({
                    'pattern_type': f'{idm["type"]}_inducement',
                    'high': float(idm['high']),
                    'low': float(idm['low']),
                    'price': float(idm['price']),
                    'start_index': int(idm.get('index', 0)),
                    'timeframe': 'H1',
                    'taken_out': idm.get('taken_out', False),
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
# ============================================================================
# ML Training - DEPRECATED Vision endpoints redirect to Synchronized Learning
# ============================================================================
# NOTE: Standalone Vision Training has been deprecated.
# All /api/ml/train/vision/* endpoints now redirect to Synchronized Learning
# which combines Text + Vision + Audio-Visual Verification for better results.
# ============================================================================

# Use the synchronized training status for all training
# _vision_training_status is now an alias to _sync_training_status (defined below)


@app.post("/api/ml/train/vision/{playlist_id}")
async def train_ml_with_vision(
    playlist_id: str,
    background_tasks: BackgroundTasks,
    vision_provider: str = Query("local", description="Vision AI provider"),
    max_frames: int = Query(0, description="Max frames per video"),
    extraction_mode: str = Query("sincere_student", description="Extraction mode")
):
    """
    DEPRECATED: Use Claude Code training instead.

    This endpoint is deprecated because:
    1. Local MLX-VLM is slow and often gets stuck
    2. Claude Code provides expert-level ICT/SMC analysis
    3. Claude Code training produces higher quality knowledge

    Use /api/ml/train/claude-code/{video_id} for each video instead.
    """
    return {
        "status": "deprecated",
        "message": "Vision training via local MLX-VLM is deprecated. Use Claude Code training instead.",
        "recommended_endpoint": "/api/ml/train/claude-code/{video_id}",
        "why": [
            "Claude Code has expert ICT/SMC knowledge",
            "Produces higher quality, verified training data",
            "Doesn't require local GPU or slow local models"
        ],
        "how_to_use": [
            "1. Call POST /api/ml/train/claude-code/{video_id} to prepare video",
            "2. Use Claude Code to analyze frames and write knowledge_base.json",
            "3. Call POST /api/ml/train/claude-code/complete/{video_id} to finish"
        ]
    }


@app.get("/api/ml/train/vision/status/{job_id}")
async def get_vision_training_status(job_id: str):
    """Get status of a training job (redirects to synchronized status)"""
    return await get_synchronized_training_status(job_id)


@app.get("/api/ml/train/vision/stream/{job_id}")
async def stream_vision_training_status(job_id: str):
    """SSE stream for training progress (redirects to synchronized stream)"""
    return await stream_synchronized_training_status(job_id)


@app.post("/api/ml/train/vision/video/{video_id}")
async def train_single_video_with_vision(
    video_id: str,
    background_tasks: BackgroundTasks,
    vision_provider: str = Query("local", description="Vision AI provider"),
    max_frames: int = Query(0, description="Max frames"),
    extraction_mode: str = Query("sincere_student", description="Extraction mode")
):
    """
    DEPRECATED: Use Claude Code training instead.

    Use /api/ml/train/claude-code/{video_id} for expert-quality training.
    """
    return {
        "status": "deprecated",
        "video_id": video_id,
        "message": "Single video vision training is deprecated. Use Claude Code training instead.",
        "recommended_endpoint": f"/api/ml/train/claude-code/{video_id}",
        "why": [
            "Claude Code has expert ICT/SMC knowledge",
            "Produces higher quality, verified training data",
            "Doesn't require local GPU or slow local models"
        ],
        "how_to_use": [
            f"1. Call POST /api/ml/train/claude-code/{video_id} to prepare video",
            "2. Use Claude Code to analyze frames and write knowledge_base.json",
            f"3. Call POST /api/ml/train/claude-code/complete/{video_id} to finish"
        ]
    }


# ============================================================================
# SYNCHRONIZED LEARNING (STATE-OF-THE-ART)
# ============================================================================

_sync_training_status = {}


@app.post("/api/ml/train/synchronized/{playlist_id}")
async def train_ml_synchronized(
    playlist_id: str,
    background_tasks: BackgroundTasks,
    vision_provider: str = Query("local", description="Vision AI provider: 'local' (FREE), 'anthropic', 'openai'"),
    max_frames: int = Query(0, description="Max frames per video (0 = no limit)"),
    extraction_mode: str = Query("sincere_student", description="'sincere_student' (recommended), 'comprehensive', 'thorough', 'balanced'"),
    alignment_threshold: float = Query(0.6, description="Min audio-visual alignment score (0.6 = 60% match required)"),
    sync_window: float = Query(2.0, description="Time window for syncing audio to visual (seconds)")
):
    """
    STATE-OF-THE-ART synchronized audio-visual training.

    This is the BEST training mode that ensures ML only learns VERIFIED knowledge
    where what is heard (transcript) matches what is seen (chart patterns).

    Key Features:
    1. WhisperX word-level timestamps (forced alignment) - precise timing
    2. Joint embedding space (ImageBind-style) - compares audio vs visual
    3. Verification Gate - REJECTS mismatched data (prevents MACD→FVG contamination)

    Based on:
    - Meta's ImageBind (joint embedding across modalities)
    - Meta's PE-AV (Perception Encoder AudioVisual)
    - WhisperX (word-level forced alignment)

    100% FREE - Uses only open-source libraries!
    """
    import uuid
    import traceback

    job_id = str(uuid.uuid4())[:8]

    _sync_training_status[job_id] = {
        "job_id": job_id,
        "playlist_id": playlist_id,
        "status": "starting",
        "progress": 0,
        "total": 0,
        "current_video": None,
        "message": "Initializing synchronized learning...",
        "started_at": datetime.utcnow().isoformat(),
        "vision_provider": vision_provider,
        "extraction_mode": extraction_mode,
        "alignment_threshold": alignment_threshold,
        "sync_window": sync_window,
        # Sync-specific stats
        "moments_analyzed": 0,
        "verified_moments": 0,
        "rejected_moments": 0,
        "contamination_prevented": [],
    }

    def run_synchronized_training():
        """Background task for synchronized training"""
        try:
            from .ml.training_pipeline import SmartMoneyKnowledgeBase, SYNC_LEARNING_AVAILABLE

            if not SYNC_LEARNING_AVAILABLE:
                _sync_training_status[job_id]["status"] = "error"
                _sync_training_status[job_id]["error"] = "Synchronized learning not available. Check dependencies."
                return

            _sync_training_status[job_id]["status"] = "loading"
            _sync_training_status[job_id]["message"] = "Loading playlist..."

            # Load playlist
            playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"
            if not playlist_file.exists():
                _sync_training_status[job_id]["status"] = "error"
                _sync_training_status[job_id]["error"] = f"Playlist not found: {playlist_id}"
                return

            with open(playlist_file) as f:
                playlist_data = json.load(f)

            videos = playlist_data.get("videos", [])
            video_ids = [v.get("video_id") for v in videos if v.get("video_id")]

            # Load transcripts
            transcripts = []
            for vid in video_ids:
                transcript_file = TRANSCRIPTS_DIR / f"{vid}.json"
                if transcript_file.exists():
                    try:
                        with open(transcript_file) as f:
                            transcript = json.load(f)
                            if transcript.get('full_text'):
                                transcripts.append(transcript)
                    except Exception as e:
                        pass

            if not transcripts:
                _sync_training_status[job_id]["status"] = "error"
                _sync_training_status[job_id]["error"] = "No transcripts found for playlist"
                return

            _sync_training_status[job_id]["total"] = len(transcripts)
            _sync_training_status[job_id]["status"] = "training"
            _sync_training_status[job_id]["message"] = "Starting synchronized audio-visual training..."

            # Progress callback
            def progress_callback(current, total, message):
                _sync_training_status[job_id]["progress"] = current
                _sync_training_status[job_id]["total"] = total
                _sync_training_status[job_id]["message"] = message

            # Initialize and run synchronized training
            kb = SmartMoneyKnowledgeBase(str(DATA_DIR))

            results = kb.train_synchronized(
                transcripts=transcripts,
                vision_provider=vision_provider,
                max_frames_per_video=max_frames,
                extraction_mode=extraction_mode,
                alignment_threshold=alignment_threshold,
                sync_window=sync_window,
                progress_callback=progress_callback
            )

            # Update status with results
            sync_stats = results.get('synchronization', {})
            _sync_training_status[job_id]["moments_analyzed"] = sync_stats.get('total_moments_analyzed', 0)
            _sync_training_status[job_id]["verified_moments"] = sync_stats.get('verified_moments', 0)
            _sync_training_status[job_id]["rejected_moments"] = sync_stats.get('rejected_moments', 0)
            _sync_training_status[job_id]["verification_rate"] = sync_stats.get('verification_rate', 0)
            _sync_training_status[job_id]["concepts_verified"] = sync_stats.get('concepts_verified', [])

            # Track rejection reasons (contamination prevention)
            rejection_reasons = sync_stats.get('rejection_reasons', {})
            _sync_training_status[job_id]["contamination_prevented"] = [
                f"{reason}: {count}" for reason, count in rejection_reasons.items()
            ]

            # Save
            kb.save()

            _sync_training_status[job_id]["status"] = "completed"
            _sync_training_status[job_id]["completed_at"] = datetime.utcnow().isoformat()
            _sync_training_status[job_id]["message"] = (
                f"Synchronized training complete! "
                f"{sync_stats.get('verified_moments', 0)} moments verified, "
                f"{sync_stats.get('rejected_moments', 0)} contaminated samples rejected."
            )
            _sync_training_status[job_id]["results"] = {
                "transcripts_processed": results.get('n_transcripts', 0),
                "verification_rate": f"{sync_stats.get('verification_rate', 0):.1%}",
                "concepts_verified": sync_stats.get('concepts_verified', []),
            }

        except Exception as e:
            _sync_training_status[job_id]["status"] = "error"
            _sync_training_status[job_id]["error"] = str(e)
            _sync_training_status[job_id]["traceback"] = traceback.format_exc()

    # Start background training
    background_tasks.add_task(run_synchronized_training)

    return {
        "job_id": job_id,
        "status": "started",
        "message": f"Synchronized training started for playlist {playlist_id}",
        "description": "STATE-OF-THE-ART training that verifies audio matches visual before learning",
        "vision_provider": vision_provider,
        "extraction_mode": extraction_mode,
        "alignment_threshold": alignment_threshold,
        "sync_window": sync_window,
        "status_url": f"/api/ml/train/synchronized/status/{job_id}",
        "stream_url": f"/api/ml/train/synchronized/stream/{job_id}"
    }


@app.get("/api/ml/train/synchronized/status/{job_id}")
async def get_synchronized_training_status(job_id: str):
    """Get status of a synchronized training job"""
    if job_id not in _sync_training_status:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _sync_training_status[job_id]


@app.get("/api/ml/train/synchronized/stream/{job_id}")
async def stream_synchronized_training_status(job_id: str):
    """SSE stream for synchronized training progress"""
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        while True:
            if job_id in _sync_training_status:
                status = _sync_training_status[job_id]
                yield {"data": json.dumps(status)}

                if status.get("status") in ["completed", "error"]:
                    break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/api/ml/synchronized/knowledge")
async def get_synchronized_knowledge(concept: str = Query(None, description="Optional: get knowledge for specific concept")):
    """
    Get VERIFIED knowledge that passed audio-visual alignment tests.

    This knowledge is guaranteed accurate - what was said matches what was shown.
    """
    kb = get_knowledge_base()
    if not kb:
        raise HTTPException(status_code=500, detail="Knowledge base not loaded")

    try:
        verified = kb.get_verified_knowledge(concept)
        return {
            "status": "success",
            "has_synchronized_learning": bool(getattr(kb, 'synchronized_knowledge', {})),
            "verified_knowledge": verified
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "has_synchronized_learning": False
        }


@app.get("/api/ml/vision/status")
async def get_vision_capabilities():
    """
    Check if synchronized learning (visual training) is available.
    NOTE: Standalone vision training is deprecated. This now checks for synchronized learning capabilities.
    """
    # Check if Ollama is available for local vision (FREE!)
    ollama_available = False
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_available = True
    except:
        pass

    # Check for synchronized knowledge instead of vision_knowledge
    kb = get_knowledge_base()
    sync_knowledge = getattr(kb, 'synchronized_knowledge', {}) if kb else {}
    has_sync = getattr(kb, 'has_synchronized_learning', False) if kb else False

    supported_providers = []
    if ollama_available:
        supported_providers.append("local")
    if os.environ.get("ANTHROPIC_API_KEY"):
        supported_providers.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        supported_providers.append("openai")

    # Count patterns from synchronized knowledge
    patterns_learned = len(sync_knowledge) if sync_knowledge else 0

    return {
        "vision_available": ollama_available or bool(supported_providers),
        "has_vision_knowledge": has_sync,  # Now uses synchronized learning status
        "patterns_learned": patterns_learned,
        "visual_concepts": patterns_learned,
        "videos_with_vision": 0,  # Will be updated when sync learning tracks this
        "supported_providers": supported_providers,
        "ollama_available": ollama_available,
        "training_mode": "synchronized",  # Indicate we use synchronized learning
        "requirements": {
            "local": "Ollama running locally (FREE!) - ollama.ai",
            "anthropic": "ANTHROPIC_API_KEY environment variable",
            "openai": "OPENAI_API_KEY environment variable"
        },
        "note": "Using Synchronized Learning (Text + Vision + Verification)"
    }


@app.get("/api/ml/vision/patterns/{pattern_type}")
async def get_visual_pattern_examples(pattern_type: str):
    """
    Get visual examples of a specific Smart Money pattern type.
    Now uses synchronized knowledge for verified visual examples.
    """
    kb = get_knowledge_base()
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    # Try to get from synchronized knowledge first
    sync_knowledge = getattr(kb, 'synchronized_knowledge', {})

    # Map pattern type to concept name
    pattern_map = {
        'fvg': 'fair_value_gap',
        'fair_value_gap': 'fair_value_gap',
        'order_block': 'order_block',
        'ob': 'order_block',
        'breaker': 'breaker_block',
        'breaker_block': 'breaker_block',
    }
    concept_name = pattern_map.get(pattern_type.lower(), pattern_type.lower())

    if concept_name in sync_knowledge:
        vk = sync_knowledge[concept_name]
        examples = vk.get('visual_evidence', []) if isinstance(vk, dict) else []
        return {
            "pattern_type": pattern_type,
            "examples": examples[:20],
            "count": len(examples),
            "source": "synchronized_learning",
            "verified": True
        }

    # Fallback to old method
    examples = kb.get_visual_pattern_examples(pattern_type) if hasattr(kb, 'get_visual_pattern_examples') else []

    available_patterns = list(sync_knowledge.keys()) if sync_knowledge else []

    return {
        "pattern_type": pattern_type,
        "examples": examples[:20] if examples else [],
        "count": len(examples) if examples else 0,
        "message": f"No verified visual examples found for '{pattern_type}'. Run synchronized training first.",
        "available_patterns": available_patterns
    }


@app.get("/api/ml/vision/teaching-moments")
async def get_teaching_moments(
    concept: Optional[str] = Query(None, description="Filter by concept (e.g., 'FVG', 'Order Block')")
):
    """
    Get key teaching moments from synchronized learning.
    These are verified moments where audio matches visual evidence.
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
    """
    Get comprehensive visual knowledge learned from synchronized training.
    NOTE: Now returns data from synchronized learning (Text + Vision + Verification).
    """
    from .ml.ml_pattern_engine import get_ml_engine

    # Get ML engine knowledge
    try:
        ml_engine = get_ml_engine()
        ml_summary = ml_engine.get_knowledge_summary()
    except Exception as e:
        ml_summary = {'status': 'error', 'patterns_learned': [], 'patterns_not_learned': []}

    # Also check synchronized knowledge
    kb = get_knowledge_base()
    has_sync = getattr(kb, 'has_synchronized_learning', False) if kb else False
    sync_knowledge = getattr(kb, 'synchronized_knowledge', {}) if kb else {}

    # Check if we have any trained knowledge
    if ml_summary.get('status') != 'trained' or not ml_summary.get('patterns_learned'):
        return {
            "has_vision_knowledge": has_sync,
            "message": "Run synchronized training for verified visual learning: /api/ml/train/synchronized/{playlist_id}",
            "patterns_not_learned": ml_summary.get('patterns_not_learned', []),
            "has_synchronized_learning": has_sync,
            "synchronized_concepts": list(sync_knowledge.keys()) if sync_knowledge else []
        }

    # Build detailed pattern info
    pattern_details = []
    for p in ml_summary.get('patterns_learned', []):
        pattern_details.append({
            'type': p.get('type'),
            'frequency': p.get('frequency', 0),
            'confidence': p.get('confidence', 0),
            'has_teaching': p.get('has_teaching', False),
            'has_visual': p.get('has_visual', False),
            'verified': p.get('type') in sync_knowledge  # Mark if verified by sync learning
        })

    return {
        "has_vision_knowledge": True,
        "patterns_learned": len(pattern_details),
        "pattern_details": pattern_details,
        "patterns_not_learned": ml_summary.get('patterns_not_learned', []),
        "videos_with_vision": ml_summary.get('total_videos', 0),
        "total_frames_analyzed": ml_summary.get('total_frames', 0),
        "chart_frames": ml_summary.get('chart_frames', 0),
        "visual_concepts": len(pattern_details),
        "key_teaching_moments_count": sum(1 for p in pattern_details if p.get('has_teaching', False)),
        "training_sources": ml_summary.get('training_sources', []),
        "last_trained": ml_summary.get('last_trained'),
        "has_synchronized_learning": has_sync,
        "synchronized_concepts": list(sync_knowledge.keys()) if sync_knowledge else [],
        "training_mode": "synchronized" if has_sync else "legacy_vision"
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

        # Count expert-trained patterns
        patterns_learned = summary.get('patterns_learned', [])
        expert_count = 0
        if hasattr(engine, 'knowledge_base') and engine.knowledge_base:
            for p in engine.knowledge_base.patterns_learned.values():
                if p.learned_traits.get('expert_trained', False):
                    expert_count += 1

        return {
            "status": summary.get('status', 'unknown'),
            "message": summary.get('message', ''),
            "patterns_learned": patterns_learned,
            "patterns_not_learned": summary.get('patterns_not_learned', []),
            "training_stats": {
                "total_videos": summary.get('total_videos', 0),
                "total_frames_analyzed": summary.get('total_frames', 0),
                "chart_frames": summary.get('chart_frames', 0),
                "last_trained": summary.get('last_trained'),
            },
            "training_method": {
                "primary": "Claude Code expert analysis",
                "fallback": "MLX-VLM (Qwen2-VL-2B)",
                "expert_trained_patterns": expert_count,
                "total_patterns": len(patterns_learned),
            },
            "training_sources": summary.get('training_sources', []),
            "usage_info": {
                "note": "Live charts ONLY detect patterns from 'patterns_learned' list.",
                "to_learn_more": "Train new playlists using Claude Code for expert-quality knowledge.",
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "patterns_learned": [],
            "patterns_not_learned": ["fvg", "order_block", "breaker_block", "market_structure"],
        }


# ============================================================================
# CLAUDE CODE TRAINING (RECOMMENDED - EXPERT QUALITY)
# ============================================================================

@app.post("/api/ml/train/claude-code/{video_id}")
async def prepare_for_claude_code_training(
    video_id: str,
    force: bool = Query(False, description="Force re-preprocessing")
):
    """
    Prepare a video for Claude Code expert training.

    This is the RECOMMENDED training method because:
    1. Claude Code has expert ICT/SMC knowledge
    2. Can analyze charts with human-level understanding
    3. Produces high-quality, verified training data

    Workflow:
    1. Call this endpoint to prepare video (download, frames, transcript)
    2. Use the returned instructions to complete training with Claude Code
    3. Claude Code writes the knowledge_base.json and summary.md files
    4. Restart backend to load the new knowledge

    Returns:
        Instructions for Claude Code to complete the training
    """
    from .ml.audio_first_learning import AudioFirstTrainer

    try:
        learner = AudioFirstTrainer()

        # Build URL from video_id
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Prepare for Claude Code (Phase 0-3)
        result = learner.prepare_for_claude_code(
            url=url,
            force_preprocess=force
        )

        return {
            "status": "ready_for_claude_code",
            "video_id": video_id,
            "message": "Video prepared. Use Claude Code to complete training.",
            "instructions": result.get('claude_code_instructions', {}),
            "preprocessing": result.get('preprocessing', {}),
            "training": result.get('training', {}),
            "next_steps": [
                "1. Read the transcript file",
                "2. View the selected key frames",
                "3. Extract ICT/SMC concepts from the video",
                "4. Write knowledge_base.json with expert analysis",
                "5. Write summary.md with human-readable summary",
                "6. Restart backend to load new knowledge"
            ]
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Preparation failed: {str(e)}\n{traceback.format_exc()}")


@app.get("/api/ml/train/claude-code/pending")
async def get_pending_claude_code_training():
    """
    Get list of videos pending Claude Code training completion.

    These are videos that have been prepared (Phase 0-3) but are waiting
    for Claude Code to complete the expert analysis (Phase 4-5).
    """
    from .ml.audio_first_learning import AudioFirstTrainer

    try:
        learner = AudioFirstTrainer()
        pending = learner.get_pending_claude_code_videos()

        return {
            "status": "success",
            "pending_count": len(pending),
            "pending_videos": pending
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "pending_videos": []
        }


@app.post("/api/ml/train/claude-code/complete/{video_id}")
async def mark_claude_code_training_complete(video_id: str):
    """
    Mark a video's Claude Code training as complete.

    Call this after Claude Code has written:
    - {video_id}_knowledge_base.json
    - {video_id}_knowledge_summary.md

    This clears the "pending" marker and reloads the ML engine.
    """
    from pathlib import Path
    from .ml.ml_pattern_engine import get_ml_engine

    try:
        # Check if knowledge base was created
        kb_path = Path(DATA_DIR) / "audio_first_training" / f"{video_id}_knowledge_base.json"
        if not kb_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Knowledge base not found: {kb_path}. Complete training first."
            )

        # Clear pending marker if it exists
        pending_path = Path(DATA_DIR) / "audio_first_training" / f"{video_id}_claude_code_pending.json"
        if pending_path.exists():
            pending_path.unlink()

        # Reload ML engine to pick up new knowledge
        ml_engine = get_ml_engine()
        ml_engine.reload_knowledge()

        # Get updated summary
        summary = ml_engine.get_knowledge_summary()

        return {
            "status": "completed",
            "video_id": video_id,
            "message": "Training complete! ML engine reloaded with new knowledge.",
            "knowledge_loaded": True,
            "patterns_learned": summary.get('patterns_learned', []),
            "total_patterns": len(summary.get('patterns_learned', []))
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                        # Try Binance Futures API first, fallback to CoinGecko if banned
                        current_price = None
                        try:
                            response = await client.get(
                                f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={binance_symbol}",
                                timeout=3.0
                            )
                            if response.status_code == 200:
                                data = response.json()
                                current_price = float(data['price'])
                        except:
                            pass

                        # Fallback to CoinGecko if Binance fails
                        if current_price is None:
                            coin_id_map = {
                                'BTCUSDT': 'bitcoin', 'ETHUSDT': 'ethereum', 'SOLUSDT': 'solana',
                                'XRPUSDT': 'ripple', 'DOGEUSDT': 'dogecoin', 'ADAUSDT': 'cardano',
                                'AVAXUSDT': 'avalanche-2', 'DOTUSDT': 'polkadot', 'LINKUSDT': 'chainlink',
                                'MATICUSDT': 'matic-network'
                            }
                            coin_id = coin_id_map.get(binance_symbol)
                            if coin_id:
                                try:
                                    response = await client.get(
                                        f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd",
                                        timeout=5.0
                                    )
                                    if response.status_code == 200:
                                        data = response.json()
                                        current_price = float(data[coin_id]['usd'])
                                except:
                                    pass

                        if current_price is not None:

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
    limit: int = Query(100, description="Number of candles"),
    end_time: Optional[int] = Query(None, description="End time (Unix timestamp in seconds) - fetch candles before this time for historical scrollback")
):
    """Get OHLCV data for live charting - uses Binance Futures for crypto

    If end_time is provided, fetches historical data ending at that timestamp.
    This enables infinite scrollback when user pans left on the chart.
    """
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
                params = {
                    'symbol': binance_symbol,
                    'interval': binance_interval,
                    'limit': limit
                }
                # Add endTime for historical scrollback (Binance uses milliseconds)
                if end_time:
                    params['endTime'] = end_time * 1000

                response = await client.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params=params,
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
            df = market_service.get_ohlcv(symbol, timeframe, limit=limit, end_time=end_time)

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
# Hedge Fund Level API Endpoints
# ============================================================================

# Import hedge fund components
try:
    from .ml.hedge_fund_ml import (
        get_pattern_grader,
        get_edge_tracker,
        get_historical_validator,
        get_mtf_analyzer,
        PatternGrade,
    )
    HEDGE_FUND_AVAILABLE = True
except ImportError:
    HEDGE_FUND_AVAILABLE = False
    print("⚠️ Hedge fund ML components not available")


@app.get("/api/hedge-fund/status")
async def get_hedge_fund_status():
    """
    Get hedge fund components status.

    Shows which hedge fund level features are available and active.
    """
    if not HEDGE_FUND_AVAILABLE:
        return {
            "available": False,
            "message": "Hedge fund ML components not installed"
        }

    return {
        "available": True,
        "components": {
            "pattern_grader": True,
            "edge_tracker": True,
            "historical_validator": True,
            "mtf_analyzer": True,
        },
        "description": "All hedge fund level features are active",
        "features": [
            "Pattern Grading (A+ to F)",
            "Statistical Edge Tracking",
            "Historical Pattern Validation",
            "Multi-Timeframe Confluence Analysis"
        ]
    }


@app.get("/api/hedge-fund/edge-statistics")
async def get_edge_statistics(pattern_type: Optional[str] = None):
    """
    Get statistical edge data for patterns.

    This shows which patterns have positive expectancy (statistical edge)
    based on tracked trade outcomes.

    Args:
        pattern_type: Optional filter for specific pattern type (fvg, order_block, etc.)

    Returns:
        Edge statistics including win rate, expectancy, profit factor
    """
    if not HEDGE_FUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hedge fund components not available")

    try:
        edge_tracker = get_edge_tracker()

        if pattern_type:
            return edge_tracker.get_edge_summary(pattern_type)
        else:
            # Return statistics for all pattern types
            return {
                "all_patterns": edge_tracker.get_edge_summary(),
                "best_patterns": edge_tracker.get_best_patterns(min_signals=5),
                "description": "Patterns with positive expectancy are recommended for trading"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hedge-fund/grade-pattern")
async def grade_pattern(
    pattern_type: str,
    pattern_data: Dict,
    market_context: Dict
):
    """
    Grade a single pattern using hedge fund methodology.

    Args:
        pattern_type: Type of pattern (fvg, order_block, breaker, etc.)
        pattern_data: Pattern details (high, low, bias, timeframe, etc.)
        market_context: Current market context (bias, zone, structure, etc.)

    Returns:
        Pattern grade (A+ to F) with detailed reasoning
    """
    if not HEDGE_FUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hedge fund components not available")

    try:
        grader = get_pattern_grader()

        # Get historical stats for this pattern type
        edge_tracker = get_edge_tracker()
        edge_summary = edge_tracker.get_edge_summary(pattern_type)

        historical_stats = None
        if not edge_summary.get('no_data'):
            historical_stats = {
                'win_rate': float(edge_summary.get('win_rate', '0%').rstrip('%')) / 100,
                'fill_rate': 0.7,
            }

        graded = grader.grade_pattern(
            pattern_type=pattern_type,
            pattern_data=pattern_data,
            market_context=market_context,
            historical_stats=historical_stats
        )

        return graded.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hedge-fund/analyze-confluence")
async def analyze_confluence(
    primary_pattern: Dict,
    primary_tf: str,
    all_tf_patterns: Dict[str, List[Dict]]
):
    """
    Analyze multi-timeframe confluence for a pattern.

    Args:
        primary_pattern: The main pattern being analyzed
        primary_tf: Timeframe of the primary pattern (M5, M15, H1, H4, D1)
        all_tf_patterns: Patterns found on all timeframes {tf: [patterns]}

    Returns:
        Confluence analysis with recommendation (STRONG/MODERATE/WEAK/AVOID)
    """
    if not HEDGE_FUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hedge fund components not available")

    try:
        mtf_analyzer = get_mtf_analyzer()

        result = mtf_analyzer.analyze_confluence(
            primary_pattern=primary_pattern,
            primary_tf=primary_tf,
            all_tf_patterns=all_tf_patterns
        )

        return {
            "pattern_type": result.pattern_type,
            "primary_timeframe": result.primary_timeframe,
            "confluence_score": result.confluence_score,
            "aligned_timeframes": result.aligned_timeframes,
            "conflicting_timeframes": result.conflicting_timeframes,
            "confluence_factors": result.confluence_factors,
            "recommendation": result.recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hedge-fund/record-trade")
async def record_trade_outcome(
    pattern_type: str,
    outcome: str,
    rr_achieved: float = 0.0,
    session: str = "",
    day_of_week: str = ""
):
    """
    Record a trade outcome for edge tracking.

    This is the feedback loop that makes the ML learn from real trades:
    - Records win/loss/breakeven outcomes
    - Updates statistical edge calculations
    - Improves pattern recommendations over time

    Args:
        pattern_type: Type of pattern traded (fvg, order_block, etc.)
        outcome: Trade result ('win', 'loss', 'breakeven')
        rr_achieved: Risk-reward ratio achieved (e.g., 2.5 for 2.5:1)
        session: Trading session ('asian', 'london', 'new_york')
        day_of_week: Day of the week ('Monday', 'Tuesday', etc.)

    Returns:
        Updated edge statistics for this pattern type
    """
    if not HEDGE_FUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hedge fund components not available")

    if outcome not in ['win', 'loss', 'breakeven']:
        raise HTTPException(status_code=400, detail="Outcome must be 'win', 'loss', or 'breakeven'")

    try:
        edge_tracker = get_edge_tracker()

        edge_tracker.record_trade(
            pattern_type=pattern_type,
            outcome=outcome,
            rr_achieved=rr_achieved,
            session=session,
            day_of_week=day_of_week
        )

        # Return updated statistics
        return {
            "recorded": True,
            "pattern_type": pattern_type,
            "outcome": outcome,
            "updated_stats": edge_tracker.get_edge_summary(pattern_type)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hedge-fund/best-patterns")
async def get_best_patterns(min_signals: int = 10):
    """
    Get the best performing patterns based on tracked outcomes.

    These are patterns with positive expectancy (statistical edge).
    The more trades tracked, the more reliable these statistics become.

    Args:
        min_signals: Minimum number of signals required to be included (default: 10)

    Returns:
        List of patterns with positive edge, sorted by expectancy
    """
    if not HEDGE_FUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hedge fund components not available")

    try:
        edge_tracker = get_edge_tracker()
        best = edge_tracker.get_best_patterns(min_signals=min_signals)

        return {
            "best_patterns": best,
            "min_signals_required": min_signals,
            "description": "Patterns are ranked by expectancy (expected R per trade)",
            "recommendation": "Focus on patterns at the top of this list for highest edge"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Backtesting & ML Quant Endpoints (Tier 1-5)
# ============================================================================

@app.post("/api/backtest/{symbol}")
async def run_backtest(
    symbol: str,
    timeframe: str = Query("D1", description="Timeframe: M5,M15,M30,H1,H4,D1,W1"),
    lookback_days: int = Query(365, description="Days of history to backtest"),
    background_tasks: BackgroundTasks = None,
):
    """
    Run pattern backtest on historical data.
    Tests all Smart Money patterns and tracks forward outcomes.
    """
    try:
        from .ml.backtester import get_backtester
        backtester = get_backtester()
        result = backtester.backtest_patterns(symbol, timeframe, lookback_days)
        backtester.save_results(result)
        return backtester.to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/results/{symbol}")
async def get_backtest_results(
    symbol: str,
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
):
    """Get saved backtest results for a symbol."""
    try:
        from .ml.backtester import get_backtester
        backtester = get_backtester()
        results = backtester.get_saved_results(symbol, timeframe)
        return {"symbol": symbol, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/performance")
async def get_backtest_performance():
    """Get aggregate backtest performance across all symbols and patterns."""
    try:
        from .ml.backtester import get_backtester
        backtester = get_backtester()
        performance = backtester.get_aggregate_performance()
        return {"performance": performance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-cache/info")
async def get_data_cache_info():
    """Get info about cached market data files."""
    try:
        from .ml.data_cache import get_data_cache
        cache = get_data_cache()
        return {"cache_files": cache.get_cache_info()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-cache/refresh/{symbol}")
async def refresh_data_cache(
    symbol: str,
    timeframe: str = Query("D1", description="Timeframe to refresh"),
):
    """Force refresh cached data for a symbol."""
    try:
        from .ml.data_cache import get_data_cache
        cache = get_data_cache()
        df = cache.get_ohlcv(symbol, timeframe, force_refresh=True)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df),
            "start": str(df.index[0]) if len(df) > 0 else None,
            "end": str(df.index[-1]) if len(df) > 0 else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/{symbol}")
async def optimize_parameters(
    symbol: str,
    timeframe: str = Query("D1", description="Timeframe to optimize"),
):
    """Run walk-forward parameter optimization (Tier 2)."""
    try:
        from .ml.parameter_optimizer import get_optimizer
        optimizer = get_optimizer()
        result = optimizer.optimize_thresholds(symbol, timeframe)
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="Parameter optimizer not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/optimize/params/{symbol}")
async def get_optimized_params(
    symbol: str,
    timeframe: str = Query("D1"),
):
    """Get saved optimized parameters for a symbol."""
    try:
        from .ml.parameter_optimizer import get_optimizer
        optimizer = get_optimizer()
        params = optimizer.get_best_params(symbol, timeframe)
        return {"symbol": symbol, "timeframe": timeframe, "params": params}
    except ImportError:
        raise HTTPException(status_code=503, detail="Parameter optimizer not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train-classifier/{symbol}")
async def train_ml_classifier(
    symbol: str,
    timeframe: str = Query("D1"),
):
    """Train ML classifiers on backtest data (Tier 3)."""
    try:
        from .ml.ml_models import get_classifier
        classifier = get_classifier()
        result = classifier.train(symbol, timeframe)
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="ML classifiers not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/classifier/status")
async def get_classifier_status():
    """Get status of trained ML classifiers (Tier 3)."""
    try:
        from .ml.ml_models import get_classifier
        classifier = get_classifier()
        return classifier.get_status()
    except ImportError:
        raise HTTPException(status_code=503, detail="ML classifiers not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train-deep/{symbol}")
async def train_deep_models(
    symbol: str,
    timeframe: str = Query("D1"),
):
    """Train deep learning models (Tier 4)."""
    try:
        from .ml.deep_models import get_deep_model
        model = get_deep_model()
        result = model.train(symbol, timeframe)
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="Deep models not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quant/regime/{symbol}")
async def get_market_regime(
    symbol: str,
    timeframe: str = Query("D1"),
):
    """Get current market regime detection (Tier 5)."""
    try:
        from .ml.quant_engine import get_quant_engine
        engine = get_quant_engine()
        regime = engine.detect_regime(symbol, timeframe)
        return regime
    except ImportError:
        raise HTTPException(status_code=503, detail="Quant engine not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quant/correlation")
async def get_correlation_matrix(
    symbols: str = Query("BTCUSDT,ETHUSDT,SPY,GOLD,DXY", description="Comma-separated symbols"),
    timeframe: str = Query("D1"),
    lookback_days: int = Query(90),
):
    """Get multi-asset correlation matrix (Tier 5)."""
    try:
        from .ml.quant_engine import get_quant_engine
        engine = get_quant_engine()
        symbol_list = [s.strip() for s in symbols.split(',')]
        result = engine.get_correlation_matrix(symbol_list, timeframe, lookback_days)
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="Quant engine not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quant/signal/{symbol}")
async def get_quant_signal(
    symbol: str,
    timeframe: str = Query("D1"),
):
    """Get full quant signal combining all tiers (Tier 5)."""
    try:
        from .ml.quant_engine import get_quant_engine
        engine = get_quant_engine()
        signal = engine.generate_signal(symbol, timeframe)
        return signal
    except ImportError:
        raise HTTPException(status_code=503, detail="Quant engine not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Tier 6: Genuine ML Price Predictor
# ============================================================================

@app.post("/api/ml/predict-price/{symbol}")
async def predict_price(
    symbol: str,
    timeframe: str = Query("D1"),
):
    """Generate forward price direction predictions using trained ML models."""
    try:
        from .ml.price_predictor import get_price_predictor
        predictor = get_price_predictor()
        result = predictor.predict_symbol(symbol.upper(), timeframe)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/predictions/{symbol}")
async def get_predictions(
    symbol: str,
    limit: int = Query(20),
):
    """Get recent predictions and outcomes for a symbol."""
    try:
        from .database import db
        predictions = db.get_recent_predictions(symbol.upper(), limit)
        return {"symbol": symbol.upper(), "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train-predictor/{symbol}")
async def train_predictor(
    symbol: str,
    timeframe: str = Query("D1"),
):
    """Train ML prediction models for a symbol (walk-forward validated)."""
    try:
        from .ml.price_predictor import get_price_predictor
        predictor = get_price_predictor()
        result = predictor.train_symbol(symbol.upper(), timeframe)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/predictor/performance")
async def get_predictor_performance(
    symbol: str = Query(None),
    lookback_days: int = Query(90),
):
    """Get live accuracy metrics from resolved predictions."""
    try:
        from .ml.price_predictor import get_price_predictor
        predictor = get_price_predictor()
        result = predictor.get_performance(
            symbol.upper() if symbol else None, lookback_days
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/predictor/status")
async def get_predictor_status():
    """Get status of all trained prediction models."""
    try:
        from .ml.price_predictor import get_price_predictor
        predictor = get_price_predictor()
        return predictor.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/predictor/resolve")
async def resolve_predictions():
    """Resolve pending predictions against actual prices."""
    try:
        from .ml.price_predictor import get_price_predictor
        predictor = get_price_predictor()
        result = predictor.resolve_predictions()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/video-knowledge/status")
async def get_video_knowledge_status(
    playlist_id: str = Query("all", description="Playlist ID to scope video knowledge")
):
    """Get video knowledge index status - concepts, teaching depth, co-occurrence."""
    try:
        from .ml.playlist_registry import PlaylistRegistry
        vk = PlaylistRegistry.get_video_knowledge(playlist_id)
        return vk.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train-pattern-quality")
async def train_pattern_quality(symbol: str = 'BTCUSDT', timeframe: str = 'D1'):
    """Train the pattern quality model using video-learned knowledge + OHLCV features."""
    try:
        from .ml.pattern_quality_model import get_pattern_quality_model
        pqm = get_pattern_quality_model()
        result = pqm.train(symbol, timeframe)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/pattern-quality/status")
async def get_pattern_quality_status():
    """Get pattern quality model status and feature importance breakdown."""
    try:
        from .ml.pattern_quality_model import get_pattern_quality_model
        pqm = get_pattern_quality_model()
        return pqm.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quant/dashboard")
async def get_quant_dashboard():
    """Get full quant dashboard with all tier data."""
    try:
        dashboard = {"tiers": {}}

        # Tier 1: Backtest performance
        try:
            from .ml.backtester import get_backtester
            backtester = get_backtester()
            dashboard["tiers"]["tier1_backtest"] = {
                "available": True,
                "details": "Pattern outcome tracking",
            }
        except Exception:
            dashboard["tiers"]["tier1_backtest"] = {"available": False}

        # Tier 2: Optimized params
        try:
            from .ml.parameter_optimizer import get_optimizer
            optimizer = get_optimizer()
            dashboard["tiers"]["tier2_optimizer"] = {
                "available": True,
                "details": "Threshold optimization",
            }
        except Exception:
            dashboard["tiers"]["tier2_optimizer"] = {"available": False}

        # Tier 3: ML classifiers
        try:
            from .ml.ml_models import get_classifier
            classifier = get_classifier()
            status = classifier.get_status()
            dashboard["tiers"]["tier3_classifier"] = {
                "available": True,
                "details": {"models_trained": status.get("models_trained", 0)},
            }
        except Exception:
            dashboard["tiers"]["tier3_classifier"] = {"available": False}

        # Tier 4: Deep models
        try:
            from .ml.deep_models import get_deep_model
            model = get_deep_model()
            dashboard["tiers"]["tier4_deep"] = {
                "available": True,
                "details": "MLP sequence models",
            }
        except Exception:
            dashboard["tiers"]["tier4_deep"] = {"available": False}

        # Tier 5: Quant engine
        try:
            from .ml.quant_engine import get_quant_engine
            engine = get_quant_engine()
            dashboard["tiers"]["tier5_quant"] = {
                "available": True,
                "details": "Regime + correlation + risk",
            }
        except Exception:
            dashboard["tiers"]["tier5_quant"] = {"available": False}

        # Tier 6: Genuine ML Price Predictor
        try:
            from .ml.price_predictor import get_price_predictor
            predictor = get_price_predictor()
            status = predictor.get_status()
            dashboard["tiers"]["tier6_predictor"] = {
                "available": True,
                "details": {
                    "models_trained": status.get("total_models", 0),
                    "pending_predictions": status.get("pending_predictions", 0),
                },
            }
        except Exception:
            dashboard["tiers"]["tier6_predictor"] = {"available": False}

        # Video Knowledge Integration
        try:
            from .ml.video_knowledge import get_video_knowledge
            vk = get_video_knowledge()
            vk_status = vk.get_status()
            dashboard["video_knowledge"] = {
                "loaded": vk_status.get("loaded", False),
                "concepts": vk_status.get("concepts", 0),
                "videos_trained": vk_status.get("videos", 0),
                "features_active": 22 if vk_status.get("loaded") else 0,
            }
        except Exception:
            dashboard["video_knowledge"] = {"loaded": False, "concepts": 0}

        # Pattern Quality Model
        try:
            from .ml.pattern_quality_model import get_pattern_quality_model
            pqm = get_pattern_quality_model()
            dashboard["pattern_quality"] = {
                "trained": pqm.is_trained,
                "video_importance": round(pqm.training_result.video_importance, 4) if pqm.training_result else None,
            }
        except Exception:
            dashboard["pattern_quality"] = {"trained": False}

        return dashboard

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
