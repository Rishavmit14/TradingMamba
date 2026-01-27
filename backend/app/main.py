"""
ICT AI Trading System - FastAPI Application

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
    title="TradingMamba - ICT AI Trading System",
    description="AI-powered trading signal system based on ICT methodology",
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
    """Get all ICT playlists"""
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
# ICT Concepts
# ============================================================================

@app.get("/api/concepts")
async def get_concepts():
    """Get all ICT concepts from taxonomy"""
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
    Analyze a symbol using ICT methodology.
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

        # Run ICT analysis
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
    """Query detailed information about an ICT concept from the knowledge base"""
    kb = get_knowledge_base()

    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    result = kb.query_concept(concept_name)

    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])

    return result


@app.post("/api/ml/predict")
async def predict_concepts(text: str = Query(..., description="Text to analyze")):
    """Predict ICT concepts in given text"""
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

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("🚀 TradingMamba API starting...")
    print(f"📁 Data directory: {DATA_DIR}")

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLAYLISTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("👋 TradingMamba API shutting down...")
