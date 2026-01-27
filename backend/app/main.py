"""
Smart Money AI Trading System - FastAPI Application

Main entry point for the backend API.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
import json
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
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
    from .models.concept import ICT_CONCEPT_TAXONOMY

    concepts = []
    for category, data in ICT_CONCEPT_TAXONOMY.items():
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
        "categories": list(ICT_CONCEPT_TAXONOMY.keys())
    }


@app.get("/api/concepts/{category}")
async def get_concepts_by_category(category: str):
    """Get concepts for a specific category"""
    from .models.concept import ICT_CONCEPT_TAXONOMY

    if category not in ICT_CONCEPT_TAXONOMY:
        raise HTTPException(status_code=404, detail="Category not found")

    data = ICT_CONCEPT_TAXONOMY[category]
    return {
        "category": category,
        "name": data.get("name"),
        "concepts": data.get("concepts", [])
    }


@app.get("/api/concepts/search")
async def search_concepts(q: str = Query(..., description="Search query")):
    """Search for concepts by name or keyword"""
    from .models.concept import ICT_CONCEPT_TAXONOMY

    query = q.lower()
    matches = []

    for category, data in ICT_CONCEPT_TAXONOMY.items():
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
            from .ml.training_pipeline import ICTKnowledgeBase
            _knowledge_base = ICTKnowledgeBase(str(DATA_DIR))
            _knowledge_base.load()
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {e}")
    return _knowledge_base


def get_signal_generator():
    """Lazy load signal generator"""
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
    """Lazy load technical analyzer"""
    global _analyzer
    if _analyzer is None:
        try:
            from .ml.technical_analysis import FullICTAnalysis
            _analyzer = FullICTAnalysis()
        except Exception as e:
            print(f"Warning: Could not load analyzer: {e}")
    return _analyzer


@app.get("/api/signals/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    timeframes: str = Query("H1,H4,D1", description="Comma-separated timeframes")
):
    """
    Analyze a symbol using Smart Money methodology.
    Returns detected concepts, market structure, and potential signal.
    """
    try:
        from .services.free_market_data import FreeMarketDataService

        market_service = FreeMarketDataService()
        analyzer = get_analyzer()
        kb = get_knowledge_base()
        sig_gen = get_signal_generator()

        tf_list = [tf.strip() for tf in timeframes.split(",")]

        # Fetch market data
        market_data = {}
        for tf in tf_list:
            df = market_service.get_ohlcv(symbol, tf, limit=200)
            if df is not None and not df.empty:
                market_data[tf] = df

        if not market_data:
            raise HTTPException(status_code=404, detail=f"No market data available for {symbol}")

        # Run Smart Money analysis
        analyses = {}
        detected_concepts = []

        for tf, df in market_data.items():
            if analyzer:
                analysis = analyzer.analyze(df, tf)
                analyses[tf] = analysis
                detected_concepts.extend(analysis.get('detected_concepts', []))

        detected_concepts = list(set(detected_concepts))

        # Generate signal
        signal = None
        if sig_gen and market_data:
            signal_obj = sig_gen.generate_signal(
                symbol=symbol,
                market_data=market_data,
                detected_concepts=detected_concepts,
                context={'is_fresh': True}
            )
            if signal_obj:
                signal = {
                    'direction': signal_obj.direction,
                    'strength': signal_obj.strength,
                    'confidence': signal_obj.confidence,
                    'concepts': signal_obj.concepts,
                    'entry_zone': signal_obj.entry_zone,
                    'stop_loss': signal_obj.stop_loss,
                    'take_profit': signal_obj.take_profit,
                    'risk_reward': signal_obj.risk_reward,
                    'reasoning': signal_obj.reasoning,
                    'confluence_score': signal_obj.confluence_score,
                }

        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'timeframes': list(market_data.keys()),
            'detected_concepts': detected_concepts,
            'analyses': {
                tf: {
                    'bias': a.get('market_structure', {}).get('bias', 'neutral'),
                    'zone': a.get('premium_discount', {}).get('zone', 'neutral'),
                    'order_blocks': len(a.get('order_blocks', {}).get('bullish', [])) + len(a.get('order_blocks', {}).get('bearish', [])),
                    'fvgs': len(a.get('fair_value_gaps', {}).get('bullish', [])) + len(a.get('fair_value_gaps', {}).get('bearish', [])),
                    'summary': a.get('summary', ''),
                }
                for tf, a in analyses.items()
            },
            'signal': signal,
        }

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals/quick/{symbol}")
async def quick_signal(symbol: str):
    """
    Get a quick signal for a symbol (single timeframe analysis).
    Faster than full analysis.
    """
    try:
        from .services.free_market_data import FreeMarketDataService

        market_service = FreeMarketDataService()
        analyzer = get_analyzer()

        # Get H1 data
        df = market_service.get_ohlcv(symbol, 'H1', limit=100)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Quick analysis
        if analyzer:
            analysis = analyzer.analyze(df, 'H1')

            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'bias': analysis.get('market_structure', {}).get('bias', 'neutral'),
                'zone': analysis.get('premium_discount', {}).get('zone', 'neutral'),
                'concepts': analysis.get('detected_concepts', []),
                'summary': analysis.get('summary', 'Analysis unavailable'),
                'current_price': float(df['close'].iloc[-1]),
            }
        else:
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'bias': 'neutral',
                'zone': 'neutral',
                'concepts': [],
                'summary': 'Analyzer not available',
                'current_price': float(df['close'].iloc[-1]),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/status")
async def get_ml_status():
    """Get ML model status and learning progress"""
    kb = get_knowledge_base()

    if kb:
        progress = kb.get_learning_progress()
        return {
            'status': 'operational',
            'knowledge_base_loaded': True,
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
        from .ml.training_pipeline import ICTKnowledgeBase

        kb = ICTKnowledgeBase(str(DATA_DIR))

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

    return {
        "status": "operational",
        "playlists_loaded": len(list(PLAYLISTS_DIR.glob("*.json"))) if PLAYLISTS_DIR.exists() else 0,
        "total_videos": total_videos,
        "transcripts_ready": transcript_count,
        "transcription_progress": f"{transcript_count}/{total_videos}" if total_videos else "0/0",
        "concepts_loaded": 40,  # From taxonomy
        "ml_trained": ml_status.get('n_transcripts_processed', 0) > 0,
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
        from .ml.pattern_recognition import ICTPatternRecognizer

        market_service = FreeMarketDataService()
        recognizer = ICTPatternRecognizer()

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
        from .ml.price_predictor import ICTPricePredictor

        market_service = FreeMarketDataService()
        predictor = ICTPricePredictor()

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
        from .ml.pattern_recognition import ICTPatternRecognizer

        market_service = FreeMarketDataService()
        chart_gen = ICTChartGenerator()

        df = market_service.get_ohlcv(symbol, timeframe, limit=100)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Get patterns
        patterns = None
        if with_patterns:
            recognizer = ICTPatternRecognizer()
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
